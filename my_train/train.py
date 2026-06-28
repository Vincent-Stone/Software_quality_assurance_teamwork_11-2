import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
from models import get_mnist_model, get_cifar10_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'auto_LiRPA'))
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def _infer_num_classes(model):
    """从模型最后一层动态推断类别数"""
    if hasattr(model, 'layers') and len(model.layers) > 0:
        return model.layers[-1].out_features
    elif hasattr(model, 'fc4'):
        return model.fc4.out_features
    elif hasattr(model, 'fc6'):
        return model.fc6.out_features
    else:
        # 遍历所有子模块查找最后一个Linear层
        linear_modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if linear_modules:
            return linear_modules[-1].out_features
    return 10


class VerificationLoss(nn.Module):
    def __init__(self, epsilon, device, lambda_weight=1.0, method='CROWN-IBP', model=None, data_shape=None):
        super(VerificationLoss, self).__init__()
        self.epsilon = epsilon
        self.device = device
        # lambda_weight 固定为超参数，不再是可训练参数，避免训练不稳定
        self.lambda_weight = lambda_weight
        # 关键修复：alpha-CROWN/CROWN-Optimized 不支持梯度传播，训练时必须使用支持梯度的方法
        train_method = _map_bound_method(method)
        if train_method in ['CROWN-Optimized', 'alpha-crown']:
            print(f"Warning: method '{method}' does not support gradient propagation for training. "
                  f"Falling back to 'CROWN-IBP' for robustness loss computation.")
            self.method = 'CROWN-IBP'
        else:
            self.method = train_method

        # 关键修复：创建BoundedModule包装模型，所有前向传播和边界计算都通过它进行
        # 这样可以避免model(data)和BoundedModule.compute_bounds混用导致的inplace operation错误
        self.bounded_model = None
        self.num_classes = 10
        if model is not None and data_shape is not None:
            self.set_model(model, data_shape)

    def set_model(self, model, data_shape):
        """在模型创建后设置BoundedModule包装器"""
        if self.bounded_model is None:
            dummy_input = torch.zeros((1,) + data_shape[1:], device=self.device)
            bound_opts = {
                'conv_mode': 'patches',
                'deterministic': True,
            }
            self.bounded_model = BoundedModule(model, dummy_input, bound_opts=bound_opts, device=self.device)
            self.num_classes = _infer_num_classes(model)

    def forward_output(self, data):
        """通过BoundedModule进行标准前向传播，用于计算分类损失"""
        if self.bounded_model is None:
            raise RuntimeError("bounded_model not initialized. Call set_model() first.")
        return self.bounded_model(data)

    def compute_verification_bound(self, data, target):
        if self.bounded_model is None:
            raise RuntimeError("bounded_model not initialized. Call set_model() first.")

        batch_size = data.shape[0]
        num_classes = self.num_classes

        ptb = PerturbationLpNorm(norm=np.inf, eps=self.epsilon)
        bounded_data = BoundedTensor(data, ptb)

        try:
            # 构造正确的分类margin矩阵C
            # margin = logit_target - mean(logits)
            # 鲁棒性要求：margin的下界 > 0
            # C矩阵定义：目标类别系数为 (num_classes-1)/num_classes，其他为 -1/num_classes
            c = torch.full((batch_size, num_classes), -1.0 / num_classes, device=self.device)
            for i in range(batch_size):
                target_class = target[i].item() if isinstance(target[i], torch.Tensor) else target[i]
                c[i, target_class] = (num_classes - 1.0) / num_classes

            lb, ub = self.bounded_model.compute_bounds(
                x=(bounded_data,), C=c.unsqueeze(1), method=self.method, bound_upper=False
            )
            # 关键修复：移除 .detach()，让支持梯度的边界方法（IBP/CROWN/CROWN-IBP）能正常传播梯度到模型参数
            lower_bounds = lb.squeeze()
        except Exception as e:
            print(f"Warning: compute_bounds failed: {e}")
            lower_bounds = torch.full((batch_size,), -float('inf'), device=self.device)

        return lower_bounds

    def forward(self, data, target, classification_loss):
        lower_bounds = self.compute_verification_bound(data, target)

        valid_mask = torch.isfinite(lower_bounds)

        if valid_mask.sum() > 0:
            valid_bounds = lower_bounds[valid_mask]

            # 核心修复：直接使用margin下界的负值作为损失
            # 当 margin_lower_bound > 0 时，模型在 epsilon 球内是鲁棒的，loss = 0
            # 当 margin_lower_bound <= 0 时，产生惩罚，迫使模型增大margin
            verification_loss = torch.clamp(-valid_bounds, min=0).mean()

            weighted_verification_loss = self.lambda_weight * verification_loss
        else:
            weighted_verification_loss = torch.tensor(0.0, device=self.device)

        total_loss = classification_loss + weighted_verification_loss

        return total_loss, weighted_verification_loss.detach(), lower_bounds


class WCIMLoss(nn.Module):
    """
    Worst-Case Interval Margin (WCIM) Loss
    
    将输入扰动区域[L∞球]对应的输出边界超矩形映射为跨所有类别的
    最坏情况margin下界，构建可导的鲁棒性损失函数。
    
    核心创新：
    1. 利用完整的[lbound, ubound]区间信息，而非仅使用下界
    2. 对每个样本计算跨所有类别对的最坏情况margin:
       m_t = lb_t - mean(ub_{j!=t})
    3. 取min_t(m_t)作为该样本的鲁棒性得分
    
    数学表达:
        lb_t: 类别t的logit下界
        ub_t: 类别t的logit上界
        margin_t = lb_t - (1/(K-1)) * sum_{j!=t}(ub_j)
        WCIM = min_t(margin_t)
        Loss = ReLU(-WCIM)
    """

    def __init__(self, epsilon, device, lambda_weight=1.0, method='CROWN-IBP', model=None, data_shape=None):
        super(WCIMLoss, self).__init__()
        self.epsilon = epsilon
        self.device = device
        self.lambda_weight = lambda_weight
        
        train_method = _map_bound_method(method)
        if train_method in ['CROWN-Optimized', 'alpha-crown']:
            print(f"Warning: method '{method}' does not support gradient propagation for training. "
                  f"Falling back to 'CROWN-IBP' for WCIM loss computation.")
            self.method = 'CROWN-IBP'
        else:
            self.method = train_method

        self.bounded_model = None
        self.num_classes = 10
        if model is not None and data_shape is not None:
            self.set_model(model, data_shape)

    def set_model(self, model, data_shape):
        """在模型创建后设置BoundedModule包装器"""
        if self.bounded_model is None:
            dummy_input = torch.zeros((1,) + data_shape[1:], device=self.device)
            bound_opts = {
                'conv_mode': 'patches',
                'deterministic': True,
            }
            self.bounded_model = BoundedModule(model, dummy_input, bound_opts=bound_opts, device=self.device)
            self.num_classes = _infer_num_classes(model)

    def forward_output(self, data):
        """通过BoundedModule进行标准前向传播"""
        if self.bounded_model is None:
            raise RuntimeError("bounded_model not initialized. Call set_model() first.")
        return self.bounded_model(data)

    def compute_wcim_bounds(self, data, target):
        """
        计算WCIM损失所需的双向边界。
        
        Returns:
            lower_bounds: shape (batch_size, num_classes) - 每个类别logit的下界
            upper_bounds: shape (batch_size, num_classes) - 每个类别logit的上界
        """
        if self.bounded_model is None:
            raise RuntimeError("bounded_model not initialized. Call set_model() first.")

        batch_size = data.shape[0]
        num_classes = self.num_classes

        ptb = PerturbationLpNorm(norm=np.inf, eps=self.epsilon)
        bounded_data = BoundedTensor(data, ptb)

        try:
            lb, ub = self.bounded_model.compute_bounds(
                x=(bounded_data,), method=self.method, 
                bound_lower=True, bound_upper=True
            )
            
            lower_bounds = lb.squeeze()
            upper_bounds = ub.squeeze()
            
        except Exception as e:
            print(f"Warning: compute_bounds failed: {e}")
            lower_bounds = torch.full((batch_size, num_classes), -float('inf'), device=self.device)
            upper_bounds = torch.full((batch_size, num_classes), float('inf'), device=self.device)

        return lower_bounds, upper_bounds

    def forward(self, data, target, classification_loss):
        """
        计算WCIM损失并返回总损失。
        
        WCIM损失计算步骤：
        1. 对每个样本的每个类别t，计算:
           margin_t = lb_t - mean(ub_{j!=t})
        2. 对每个样本，取min_t(margin_t)作为该样本的最坏情况margin
        3. WCIM损失 = mean(relu(-worst_margin))
        """
        lower_bounds, upper_bounds = self.compute_wcim_bounds(data, target)
        
        batch_size = lower_bounds.shape[0]
        num_classes = lower_bounds.shape[1]
        
        valid_mask_lb = torch.isfinite(lower_bounds)
        valid_mask_ub = torch.isfinite(upper_bounds)
        valid_mask = valid_mask_lb & valid_mask_ub
        
        if valid_mask.sum() > 0:
            lb_valid = lower_bounds[valid_mask].view(-1, num_classes)
            ub_valid = upper_bounds[valid_mask].view(-1, num_classes)
            
            sum_ub = ub_valid.sum(dim=1, keepdim=True)
            mean_ub_others = (sum_ub - ub_valid) / (num_classes - 1)
            
            margin_per_class = lb_valid - mean_ub_others
            worst_margin = margin_per_class.min(dim=1)[0]
            
            wcim_loss = torch.clamp(-worst_margin, min=0).mean()
            weighted_wcim_loss = self.lambda_weight * wcim_loss
        else:
            weighted_wcim_loss = torch.tensor(0.0, device=self.device)

        total_loss = classification_loss + weighted_wcim_loss

        return total_loss, weighted_wcim_loss.detach(), lower_bounds, upper_bounds


def _map_bound_method(method_name):
    """映射命令行参数到auto_LiRPA的method参数"""
    mapping = {
        'IBP': 'IBP',
        'CROWN': 'backward',
        'CROWN-IBP': 'CROWN-IBP',
        'alpha-CROWN': 'CROWN-Optimized',
        'CROWN-Optimized': 'CROWN-Optimized',
    }
    return mapping.get(method_name, method_name)


def train_model_with_ab_crown(dataset: str, model_type: str, epochs: int = 10, 
                               batch_size: int = 128, lr: float = 0.0001, save_path: str = None,
                               epsilon: float = 0.1, lambda_weight: float = 10.0,
                               use_ab_crown: bool = True, bound_method: str = 'CROWN-Optimized',
                               loss_type: str = 'standard', warmup_epochs: int = 3):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    print(f"Using robustness method: {bound_method}, epsilon={epsilon}, lambda_weight={lambda_weight}, loss_type={loss_type}, warmup_epochs={warmup_epochs}")
    
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        model = get_mnist_model(model_type)
    
    elif dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        model = get_cifar10_model(model_type)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    grad_clip = 5.0  # 梯度裁剪阈值，防止鲁棒性训练中的梯度爆炸
    
    if use_ab_crown:
        sample_data, _ = next(iter(train_loader))
        data_shape = sample_data.shape
        
        if loss_type == 'wcim':
            bound_loss_fn = WCIMLoss(epsilon=epsilon, device=device, lambda_weight=lambda_weight, method=bound_method)
            print(f"Using WCIM (Worst-Case Interval Margin) Loss with {bound_method}")
        else:
            bound_loss_fn = VerificationLoss(epsilon=epsilon, device=device, lambda_weight=lambda_weight, method=bound_method)
            print(f"Using Standard Verification Loss with {bound_method}")
        
        bound_loss_fn.set_model(model, data_shape)
    else:
        bound_loss_fn = None
    
    print(f"Training {model_type} model on {dataset} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_classification_loss = 0
        total_verification_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if use_ab_crown and bound_loss_fn is not None and epoch >= warmup_epochs:
                output = bound_loss_fn.forward_output(data)
                classification_loss = criterion(output, target)
                
                if loss_type == 'wcim':
                    loss, verification_loss, lower_bounds, upper_bounds = bound_loss_fn(data, target, classification_loss)
                else:
                    loss, verification_loss, lower_bounds = bound_loss_fn(data, target, classification_loss)
                
                if (batch_idx + 1) % 50 == 0:
                    valid_bounds = lower_bounds[torch.isfinite(lower_bounds)]
                    if len(valid_bounds) > 0:
                        print(f'  Batch {batch_idx+1}: Class Loss: {classification_loss.item():.4f}, '
                              f'Verify Loss: {verification_loss.item():.4f}, '
                              f'Bound Mean: {valid_bounds.mean().item():.4f}, '
                              f'λ: {bound_loss_fn.lambda_weight:.4f}')
            else:
                if use_ab_crown and bound_loss_fn is not None:
                    output = bound_loss_fn.forward_output(data)
                else:
                    output = model(data)
                classification_loss = criterion(output, target)
                loss = classification_loss
                verification_loss = torch.tensor(0.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # print(f"  [Debug] Optimizer step done", flush=True)
            
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_verification_loss += verification_loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {total_loss/(batch_idx+1):.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        avg_class_loss = total_classification_loss / len(train_loader)
        avg_verify_loss = total_verification_loss / len(train_loader) if use_ab_crown else 0
        print(f'Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, Test Acc: {test_acc:.2f}%, '
              f'Avg Class Loss: {avg_class_loss:.4f}, Avg Verify Loss: {avg_verify_loss:.4f}')
    
    if save_path is None:
        save_path = f'../robustness_verification/test_models/{dataset}_{model_type}_{bound_method}_{loss_type}.pt'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model


def train_model(dataset: str, model_type: str, epochs: int = 10, 
                batch_size: int = 128, lr: float = 0.01, save_path: str = None):
    
    device = torch.device('cpu')
    print(f"Training on device: {device}")
    
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        model = get_mnist_model(model_type)
    
    elif dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        model = get_cifar10_model(model_type)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    grad_clip = 5.0
    
    print(f"Training {model_type} model on {dataset} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {total_loss/(batch_idx+1):.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, Test Acc: {test_acc:.2f}%')
    
    if save_path is None:
        save_path = f'../robustness_verification/test_models/{dataset}_{model_type}.pt'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model


def train_two_stage(dataset: str, model_type: str, 
                    epochs_stage1: int = 5, epochs_stage2: int = 5,
                    batch_size: int = 128, lr_stage1: float = 0.001, lr_stage2: float = 1e-5,
                    epsilon: float = 0.1, lambda_weight: float = 10.0,
                    bound_method: str = 'CROWN-IBP', save_path: str = None):
    """
    两阶段训练策略：
    阶段1: 标准训练快速收敛到较好的初始权重
    阶段2: 使用鲁棒性损失（CROWN-IBP）微调，提升模型的鲁棒性边界
    
    参数:
        dataset: 数据集名称 ('MNIST' 或 'CIFAR10')
        model_type: 模型类型 ('simple', 'medium', 'deep')
        epochs_stage1: 第一阶段训练轮数
        epochs_stage2: 第二阶段训练轮数
        batch_size: 批量大小
        lr_stage1: 第一阶段学习率
        lr_stage2: 第二阶段学习率（通常更小）
        epsilon: 扰动半径
        lambda_weight: 鲁棒性损失权重
        bound_method: 边界计算方法
        save_path: 模型保存路径
    """
    print("\n" + "="*60)
    print("开始两阶段训练")
    print("="*60)
    
    # ========== 第一阶段：标准训练 ==========
    print(f"\n{'='*60}")
    print(f"阶段 1/2: 标准训练 ({epochs_stage1} epochs)")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        model = get_mnist_model(model_type)
    elif dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        model = get_cifar10_model(model_type)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_stage1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs_stage1//2), gamma=0.5)
    grad_clip = 5.0
    
    print(f"Training {model_type} model on {dataset} for {epochs_stage1} epochs (Stage 1)...")
    
    best_test_acc = 0.0
    
    for epoch in range(epochs_stage1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'  Epoch: {epoch+1}/{epochs_stage1}, Batch: {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {total_loss/(batch_idx+1):.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        # 评估
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        best_test_acc = max(best_test_acc, test_acc)
        print(f'  [Stage 1] Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, '
              f'Test Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)')
    
    print(f"\n阶段1完成 - 最佳测试准确率: {best_test_acc:.2f}%")
    
    # ========== 第二阶段：鲁棒性训练 ==========
    print(f"\n{'='*60}")
    print(f"阶段 2/2: 鲁棒性训练 ({epochs_stage2} epochs, {bound_method})")
    print(f"{'='*60}")
    print(f"使用较小的学习率 {lr_stage2} 进行微调...")
    
    # 重新初始化优化器，使用更小的学习率
    optimizer = optim.Adam(model.parameters(), lr=lr_stage2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs_stage2//2), gamma=0.5)
    
    # 创建鲁棒性损失函数
    bound_loss_fn = VerificationLoss(
        epsilon=epsilon, 
        device=device, 
        lambda_weight=lambda_weight, 
        method=_map_bound_method(bound_method)
    )
    
    print(f"Training with {bound_method} robustness for {epochs_stage2} epochs (Stage 2)...")
    
    for epoch in range(epochs_stage2):
        model.train()
        total_loss = 0
        total_classification_loss = 0
        total_verification_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            classification_loss = criterion(output, target)
            
            # 计算鲁棒性损失（临时切换到eval模式禁用Dropout）
            model.eval()
            loss, verification_loss, lower_bounds = bound_loss_fn(model, data, target, classification_loss)
            model.train()
            
            if (batch_idx + 1) % 50 == 0:
                valid_bounds = lower_bounds[torch.isfinite(lower_bounds)]
                if len(valid_bounds) > 0:
                    print(f'  Batch {batch_idx+1}: Class Loss: {classification_loss.item():.4f}, '
                          f'Verify Loss: {verification_loss.item():.4f}, '
                          f'Bound Mean: {valid_bounds.mean().item():.4f}, '
                          f'λ: {bound_loss_fn.lambda_weight:.4f}')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_verification_loss += verification_loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'  Epoch: {epoch+1}/{epochs_stage2}, Batch: {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {total_loss/(batch_idx+1):.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        # 评估
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        avg_class_loss = total_classification_loss / len(train_loader)
        avg_verify_loss = total_verification_loss / len(train_loader)
        print(f'  [Stage 2] Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, '
              f'Test Acc: {test_acc:.2f}%, '
              f'Avg Class Loss: {avg_class_loss:.4f}, Avg Verify Loss: {avg_verify_loss:.4f}')
    
    # 保存模型
    if save_path is None:
        save_path = f'../robustness_verification/test_models/{dataset}_{model_type}_two_stage.pt'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n两阶段训练完成！模型保存到: {save_path}")
    print(f"最终测试准确率: {test_acc:.2f}%")
    print("="*60 + "\n")
    
    return model


def load_model(dataset: str, model_type: str, model_path: str = None):
    if dataset == 'MNIST':
        model = get_mnist_model(model_type)
    elif dataset == 'CIFAR10':
        model = get_cifar10_model(model_type)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Model loaded from {model_path}")
    else:
        print("Warning: Model path not found, using randomly initialized model")
    
    model.eval()
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--model', type=str, default='tiny', help='Model size: tiny (fastest), simple, medium, deep')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bound-method', type=str, default='none',
                        choices=['none', 'IBP', 'CROWN', 'CROWN-IBP', 'alpha-CROWN'],
                        help='Bound computation method for robustness training')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Perturbation epsilon for robustness')
    parser.add_argument('--lambda-weight', type=float, default=1.0, help='Weight for verification loss')
    parser.add_argument('--two-stage', action='store_true', 
                        help='Use two-stage training (standard + robustness fine-tuning)')
    parser.add_argument('--epochs-stage1', type=int, default=5, 
                        help='Number of epochs for stage 1 (standard training)')
    parser.add_argument('--epochs-stage2', type=int, default=5, 
                        help='Number of epochs for stage 2 (robustness training)')
    parser.add_argument('--lr-stage2', type=float, default=0.0001, 
                        help='Learning rate for stage 2 (usually smaller)')
    args = parser.parse_args()
    
    if args.two_stage:
        # 两阶段训练模式
        train_two_stage(
            args.dataset, args.model,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            batch_size=args.batch_size,
            lr_stage1=args.lr,
            lr_stage2=args.lr_stage2,
            epsilon=args.epsilon,
            lambda_weight=args.lambda_weight,
            bound_method=args.bound_method if args.bound_method != 'none' else 'CROWN-IBP'
        )
    elif args.bound_method != 'none':
        # 单阶段鲁棒训练模式
        train_model_with_ab_crown(
            args.dataset, args.model, args.epochs, args.batch_size, args.lr,
            epsilon=args.epsilon, lambda_weight=args.lambda_weight, use_ab_crown=True,
            bound_method=_map_bound_method(args.bound_method)
        )
    else:
        # 标准训练模式
        train_model(args.dataset, args.model, args.epochs, args.batch_size, args.lr)
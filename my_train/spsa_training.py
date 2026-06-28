"""
混合训练策略：CROWN-IBP + alpha-CROWN SPSA微调

使用梯度无关的SPSA（Simultaneous Perturbation Stochastic Approximation）方法，
在alpha-CROWN边界下进行模型参数优化，从而解决alpha-CROWN无法反向传播梯度的问题。

算法流程：
1. Phase 1: 使用CROWN-IBP进行快速梯度训练，获得一个良好初始化的模型
2. Phase 2: 使用SPSA方法，利用alpha-CROWN边界作为损失函数进行微调

SPSA核心公式：
g_hat_i = [L(θ + c·Δ) - L(θ - c·Δ)] / (2·c·Δ_i)
其中 Δ 是随机扰动向量（±1），c 是扰动尺度

优化改进：
1. 梯度裁剪与动量项
2. 参数正则化（权重衰减）
3. 数据预加载与高效缓存
4. 改进的学习率调度策略
5. 完善的监控与日志系统
6. 异常检测与警报机制
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import psutil
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'auto_LiRPA'))

from models import get_mnist_model
from train import train_model_with_ab_crown
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class SPSAOptimizer:
    """
    改进版SPSA（Simultaneous Perturbation Stochastic Approximation）优化器
    
    改进点：
    1. 梯度裁剪（gradient clipping）- 限制梯度范数
    2. 动量项（momentum）- 加速收敛
    3. 权重衰减（weight decay）- 正则化
    4. 自适应扰动尺度 - 根据损失波动调整
    5. 梯度平滑 - 使用移动平均减少噪声
    6. 参数更新裁剪 - 限制单次参数更新幅度
    """
    
    def __init__(self, parameters, a: float = 0.001, c: float = 0.001, 
                 alpha: float = 0.602, gamma: float = 0.101,
                 momentum: float = 0.0, weight_decay: float = 1e-4,
                 grad_clip: float = 1.0, adaptive_scale: bool = True,
                 param_clip: float = 0.01, smoothing_factor: float = 0.9):
        """
        参数说明：
        - a: 学习率衰减系数（推荐 0.001）
        - c: 扰动尺度衰减系数（推荐 0.001）
        - alpha: 学习率衰减指数 (推荐 0.602)
        - gamma: 扰动尺度衰减指数 (推荐 0.101)
        - momentum: 动量系数（SPSA通常不使用动量，推荐0.0）
        - weight_decay: 权重衰减系数
        - grad_clip: 梯度裁剪阈值（严格限制）
        - adaptive_scale: 是否自适应调整扰动尺度
        - param_clip: 参数更新裁剪阈值
        - smoothing_factor: 梯度平滑因子
        """
        self.parameters = list(parameters)
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.adaptive_scale = adaptive_scale
        self.param_clip = param_clip
        self.smoothing_factor = smoothing_factor
        self.k = 0
        
        self.momentum_buffer = [torch.zeros_like(p.data) for p in self.parameters]
        self.smoothed_grad_buffer = [torch.zeros_like(p.data) for p in self.parameters]
        
        self.grad_norm_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.loss_diff_history = deque(maxlen=50)
        
        self.total_params = sum(p.numel() for p in self.parameters)
    
    def get_step_size(self):
        return self.a / (self.k + 1) ** self.alpha
    
    def get_perturbation_scale(self):
        c_k = self.c / (self.k + 1) ** self.gamma
        
        if self.adaptive_scale and len(self.loss_history) > 20:
            loss_std = np.std(list(self.loss_history))
            if loss_std > 1.0:
                c_k = min(c_k, c_k * 0.3)
            elif loss_std > 0.5:
                c_k = min(c_k, c_k * 0.5)
            elif loss_std < 0.05:
                c_k = max(c_k, c_k * 1.2)
        
        if len(self.loss_diff_history) > 20:
            diff_std = np.std(list(self.loss_diff_history))
            if diff_std > 10.0:
                c_k = min(c_k, c_k * 0.5)
        
        c_k = max(c_k, 1e-6)
        
        return c_k
    
    def perturb(self):
        perturbations = []
        for param in self.parameters:
            delta = torch.sign(torch.randn_like(param))
            perturbations.append(delta)
        return perturbations
    
    def compute_grad_norm(self, grad_estimate: float, perturbations: List[torch.Tensor]) -> float:
        """计算梯度估计的范数（按参数数量归一化）"""
        total_norm = 0.0
        count = 0
        for delta in perturbations:
            total_norm += (delta ** 2).sum().item()
            count += delta.numel()
        return abs(grad_estimate) * np.sqrt(total_norm / max(count, 1))
    
    def step(self, loss_plus: float, loss_minus: float, perturbations: List[torch.Tensor]) -> Dict:
        """
        根据两次损失评估更新参数
        
        Returns:
            统计信息字典
        """
        self.k += 1
        a_k = self.get_step_size()
        c_k = self.get_perturbation_scale()
        
        loss_plus_clipped = max(min(loss_plus, 1e5), -1e5)
        loss_minus_clipped = max(min(loss_minus, 1e5), -1e5)
        
        loss_diff = loss_plus_clipped - loss_minus_clipped
        
        max_diff = max(1.0, c_k * 1000)
        loss_diff = max(min(loss_diff, max_diff), -max_diff)
        
        self.loss_diff_history.append(loss_diff)
        
        grad_estimate = loss_diff / (2 * c_k)
        
        max_grad = self.grad_clip * 10
        grad_estimate = max(min(grad_estimate, max_grad), -max_grad)
        
        grad_norm = self.compute_grad_norm(grad_estimate, perturbations)
        
        if grad_norm > self.grad_clip:
            grad_estimate = grad_estimate * (self.grad_clip / grad_norm)
        
        self.grad_norm_history.append(grad_norm)
        self.loss_history.append((loss_plus_clipped + loss_minus_clipped) / 2)
        
        for param, delta, momentum, smoothed_grad in zip(
                self.parameters, perturbations, self.momentum_buffer, self.smoothed_grad_buffer):
            
            raw_grad = grad_estimate * delta
            
            smoothed_grad.mul_(self.smoothing_factor)
            smoothed_grad.add_(raw_grad * (1 - self.smoothing_factor))
            
            momentum.mul_(self.momentum)
            momentum.add_(smoothed_grad)
            
            update = a_k * momentum
            
            if self.weight_decay > 0:
                update.add_(self.weight_decay * param.data)
            
            if self.param_clip > 0:
                update_norm = update.norm()
                if update_norm > self.param_clip:
                    update.mul_(self.param_clip / update_norm)
            
            param.data.add_(update)
        
        return {
            'step_size': a_k,
            'perturbation_scale': c_k,
            'grad_norm': grad_norm,
            'loss_diff': loss_diff,
            'avg_loss': (loss_plus + loss_minus) / 2
        }
    
    def get_stats(self) -> Dict:
        """获取优化器统计信息"""
        return {
            'current_step': self.k,
            'current_step_size': self.get_step_size(),
            'current_perturbation_scale': self.get_perturbation_scale(),
            'avg_grad_norm': np.mean(list(self.grad_norm_history)) if self.grad_norm_history else 0,
            'avg_loss': np.mean(list(self.loss_history)) if self.loss_history else 0,
            'avg_loss_diff': np.mean(list(self.loss_diff_history)) if self.loss_diff_history else 0
        }


class PreloadedDataset(torch.utils.data.Dataset):
    """
    预加载数据集到内存，减少IO等待时间
    
    改进点：
    1. 所有数据一次性加载到内存
    2. 支持随机数据增强
    3. 支持缓存机制
    """
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
        self.data = []
        self.targets = []
        
        for i in range(len(dataset)):
            item = dataset[i]
            self.data.append(item[0])
            self.targets.append(item[1])
        
        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)
        
        print(f"Preloaded {len(self)} samples into memory")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            data = self.transform(data)
        
        return data, target


class TrainingMonitor:
    """
    训练监控与日志系统
    
    功能：
    1. 实时跟踪损失值、准确率、梯度范数
    2. 监控GPU/CPU资源使用率
    3. 检测异常（损失爆炸、梯度消失等）
    4. 生成训练日志
    """
    
    def __init__(self, log_interval: int = 10, alert_thresholds: Dict = None):
        self.log_interval = log_interval
        self.alert_thresholds = alert_thresholds or {
            'loss_increase_ratio': 2.0,
            'loss_threshold': 100.0,
            'grad_norm_threshold': 100.0,
            'accuracy_drop_ratio': 0.5
        }
        
        self.start_time = time.time()
        self.loss_history = []
        self.accuracy_history = []
        self.grad_norm_history = []
        self.resource_usage_history = []
        
        self.last_alert_time = 0
        self.alert_cooldown = 60
        
        self.log_file = None
    
    def set_log_file(self, filepath: str):
        self.log_file = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(f"Training started at {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n")
    
    def log(self, message: str, level: str = 'INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_line + "\n")
    
    def get_resource_usage(self) -> Dict:
        """获取当前资源使用情况"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info['gpu_utilization'] = torch.cuda.utilization()
            gpu_info['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**2
            gpu_info['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1024**2
            gpu_info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**2
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            **gpu_info
        }
    
    def check_anomaly(self, loss: float, grad_norm: float = None, accuracy: float = None) -> List[str]:
        """检测训练异常"""
        alerts = []
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.alert_cooldown:
            return alerts
        
        if loss > self.alert_thresholds['loss_threshold']:
            alerts.append(f"Loss explosion detected: {loss:.4f} > threshold {self.alert_thresholds['loss_threshold']}")
        
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            avg_recent = np.mean(recent_losses)
            if avg_recent > 0 and loss > avg_recent * self.alert_thresholds['loss_increase_ratio']:
                alerts.append(f"Loss increased by {self.alert_thresholds['loss_increase_ratio']}x: {loss:.4f} vs avg {avg_recent:.4f}")
        
        if grad_norm is not None and grad_norm > self.alert_thresholds['grad_norm_threshold']:
            alerts.append(f"Gradient norm too large: {grad_norm:.4f} > threshold {self.alert_thresholds['grad_norm_threshold']}")
        
        if accuracy is not None and len(self.accuracy_history) > 5:
            recent_accs = self.accuracy_history[-5:]
            avg_recent_acc = np.mean(recent_accs)
            if avg_recent_acc > 0 and accuracy < avg_recent_acc * self.alert_thresholds['accuracy_drop_ratio']:
                alerts.append(f"Accuracy dropped significantly: {accuracy:.2f}% vs avg {avg_recent_acc:.2f}%")
        
        if alerts:
            self.last_alert_time = current_time
            for alert in alerts:
                self.log(alert, level='ALERT')
        
        return alerts
    
    def record(self, loss: float, accuracy: float = None, grad_norm: float = None, 
               step: int = None, epoch: int = None, additional_info: Dict = None):
        """记录训练指标"""
        self.loss_history.append(loss)
        
        if accuracy is not None:
            self.accuracy_history.append(accuracy)
        
        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)
        
        self.resource_usage_history.append(self.get_resource_usage())
        
        self.check_anomaly(loss, grad_norm, accuracy)
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        return {
            'total_time': time.time() - self.start_time,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else None,
            'min_loss': np.min(self.loss_history) if self.loss_history else None,
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
            'avg_accuracy': np.mean(self.accuracy_history) if self.accuracy_history else None,
            'avg_grad_norm': np.mean(self.grad_norm_history) if self.grad_norm_history else None,
            'total_steps': len(self.loss_history)
        }


class AlphaCROWNSPSATrainer:
    """
    使用alpha-CROWN边界和SPSA进行训练（优化版）
    
    改进点：
    1. 改进的BoundedModule管理
    2. 优化的损失计算
    3. 支持混合损失（分类损失 + 鲁棒性损失）
    4. 完善的训练监控
    """
    
    def __init__(self, model, epsilon: float = 0.1, device='cpu', 
                 lambda_weight: float = 1.0, log_dir: str = './logs'):
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.lambda_weight = lambda_weight
        
        dummy_input = torch.zeros(1, 1, 28, 28, device=device)
        bound_opts = {'conv_mode': 'patches', 'deterministic': True}
        self.bounded_model = BoundedModule(model, dummy_input, bound_opts=bound_opts, device=device)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.monitor = TrainingMonitor(log_interval=10)
        log_path = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        self.monitor.set_log_file(log_path)
    
    def compute_alpha_crown_loss(self, data: torch.Tensor, target: torch.Tensor) -> float:
        """
        使用alpha-CROWN边界计算鲁棒性损失
        
        Returns:
            损失值（标量）
        """
        ptb = PerturbationLpNorm(norm=float('inf'), eps=self.epsilon)
        bounded_data = BoundedTensor(data, ptb)
        
        try:
            lb, ub = self.bounded_model.compute_bounds(
                x=(bounded_data,), 
                method='CROWN-Optimized',
                bound_lower=True,
                bound_upper=True
            )
            
            batch_size = lb.shape[0]
            num_classes = lb.shape[1]
            
            sum_ub = ub.sum(dim=1, keepdim=True)
            mean_ub_others = (sum_ub - ub) / (num_classes - 1)
            margin_per_class = lb - mean_ub_others
            worst_margin = margin_per_class.min(dim=1)[0]
            wcim_loss = torch.clamp(-worst_margin, min=0).mean()
            
            loss_value = float(wcim_loss.item())
            
            if loss_value > 100:
                loss_value = 100
            elif loss_value < 0:
                loss_value = 0
            
            return loss_value
        
        except Exception as e:
            self.monitor.log(f"compute_bounds failed: {e}", level='WARNING')
            return float('inf')
    
    def compute_standard_loss(self, data: torch.Tensor, target: torch.Tensor) -> float:
        """计算标准分类损失"""
        output = self.model(data)
        loss = self.criterion(output, target)
        return float(loss.item())
    
    def evaluate(self, test_loader) -> Tuple[float, float]:
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / total
        
        return accuracy, avg_loss
    
    def spsa_fine_tune(self, train_loader, test_loader, 
                       epochs: int = 10, batch_size: int = 32,
                       a: float = 0.001, c: float = 0.001,
                       momentum: float = 0.0, weight_decay: float = 1e-4,
                       grad_clip: float = 1.0, param_clip: float = 0.01,
                       smoothing_factor: float = 0.9,
                       verbose: bool = True) -> List[Dict]:
        """
        使用改进版SPSA进行alpha-CROWN边界微调
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 微调轮数
            batch_size: 批大小（用于SPSA评估）
            a: SPSA学习率系数（推荐 0.001）
            c: SPSA扰动系数（推荐 0.001）
            momentum: 动量系数（SPSA通常不使用动量，推荐0.0）
            weight_decay: 权重衰减系数
            grad_clip: 梯度裁剪阈值（严格限制，推荐1.0）
            param_clip: 参数更新裁剪阈值（限制单次参数更新幅度）
            smoothing_factor: 梯度平滑因子（减少噪声，推荐0.9）
            verbose: 是否打印进度
        
        Returns:
            训练历史记录
        """
        optimizer = SPSAOptimizer(
            self.model.parameters(), 
            a=a, c=c,
            momentum=momentum,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            param_clip=param_clip,
            smoothing_factor=smoothing_factor
        )
        
        history = []
        start_time = time.time()
        total_batches = len(train_loader)
        
        for epoch in range(epochs):
            self.model.train()
            total_alpha_crown_loss = 0
            total_standard_loss = 0
            batches_processed = 0
            
            epoch_start_time = time.time()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                if data.size(0) < batch_size:
                    continue
                
                data = data[:batch_size]
                target = target[:batch_size]
                
                original_params = [param.data.clone() for param in self.model.parameters()]
                
                perturbations = optimizer.perturb()
                c_k = optimizer.get_perturbation_scale()
                
                for param, delta in zip(self.model.parameters(), perturbations):
                    param.data.add_(c_k * delta)
                
                loss_plus = self.compute_alpha_crown_loss(data, target)
                
                if loss_plus == float('inf'):
                    for param, orig in zip(self.model.parameters(), original_params):
                        param.data.copy_(orig)
                    continue
                
                for param, orig, delta in zip(self.model.parameters(), original_params, perturbations):
                    param.data.copy_(orig)
                    param.data.sub_(c_k * delta)
                
                loss_minus = self.compute_alpha_crown_loss(data, target)
                
                if loss_minus == float('inf'):
                    for param, orig in zip(self.model.parameters(), original_params):
                        param.data.copy_(orig)
                    continue
                
                for param, orig in zip(self.model.parameters(), original_params):
                    param.data.copy_(orig)
                
                opt_stats = optimizer.step(loss_plus, loss_minus, perturbations)
                
                std_loss = self.compute_standard_loss(data, target)
                
                total_alpha_crown_loss += loss_plus
                total_standard_loss += std_loss
                batches_processed += 1
                
                self.monitor.record(
                    loss=loss_plus,
                    grad_norm=opt_stats['grad_norm'],
                    step=optimizer.k,
                    epoch=epoch
                )
                
                if verbose and batches_processed % self.monitor.log_interval == 0:
                    elapsed = time.time() - start_time
                    batch_percent = (batch_idx + 1) / total_batches * 100
                    
                    print(f"  Epoch {epoch+1}/{epochs} [{batch_percent:.1f}%] "
                          f"Batch {batches_processed}: "
                          f"Alpha-CROWN Loss={loss_plus:.4f}, "
                          f"Standard Loss={std_loss:.4f}, "
                          f"Grad Norm={opt_stats['grad_norm']:.4f}, "
                          f"Step Size={opt_stats['step_size']:.6f}, "
                          f"Time={elapsed:.2f}s")
            
            epoch_time = time.time() - epoch_start_time
            test_acc, test_loss = self.evaluate(test_loader)
            avg_alpha_crown_loss = total_alpha_crown_loss / batches_processed if batches_processed > 0 else 0
            
            optimizer_stats = optimizer.get_stats()
            
            history.append({
                'epoch': epoch + 1,
                'alpha_crown_loss': avg_alpha_crown_loss,
                'standard_loss': total_standard_loss / batches_processed if batches_processed > 0 else 0,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'spsa_step': optimizer.k,
                'epoch_time': epoch_time,
                'avg_grad_norm': optimizer_stats['avg_grad_norm'],
                'current_step_size': optimizer_stats['current_step_size'],
                'current_perturb_scale': optimizer_stats['current_perturbation_scale']
            })
            
            self.monitor.record(
                loss=avg_alpha_crown_loss,
                accuracy=test_acc,
                epoch=epoch
            )
            
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}:")
                print(f"  Alpha-CROWN Loss: {avg_alpha_crown_loss:.4f}")
                print(f"  Standard Loss: {history[-1]['standard_loss']:.4f}")
                print(f"  Test Accuracy: {test_acc:.2f}%")
                print(f"  Test Loss: {test_loss:.4f}")
                avg_grad_norm = optimizer_stats.get('avg_grad_norm', 0)
                print(f"  Avg Grad Norm: {avg_grad_norm:.4f}" if avg_grad_norm else "  Avg Grad Norm: N/A")
                print(f"  Epoch Time: {epoch_time:.2f}s")
                print(f"  Total Time: {time.time() - start_time:.2f}s")
        
        summary = self.monitor.get_summary()
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Training Time: {summary['total_time']:.2f}s")
        print(f"Final Alpha-CROWN Loss: {summary['final_loss']:.4f}")
        print(f"Average Alpha-CROWN Loss: {summary['avg_loss']:.4f}")
        print(f"Minimum Alpha-CROWN Loss: {summary['min_loss']:.4f}")
        print(f"Final Test Accuracy: {summary['final_accuracy']:.2f}%")
        avg_grad_norm = summary.get('avg_grad_norm', 0)
        print(f"Average Gradient Norm: {avg_grad_norm:.4f}" if avg_grad_norm else "Average Gradient Norm: N/A")
        
        return history


def create_data_loaders(dataset: str = 'MNIST', batch_size: int = 32, 
                        preload: bool = True, num_workers: int = 2) -> Tuple:
    """
    创建优化的数据加载器
    
    改进点：
    1. 支持数据预加载到内存
    2. 使用适当的num_workers
    3. 支持高效的数据处理
    """
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
    
    if preload:
        train_dataset = PreloadedDataset(train_dataset)
        test_dataset = PreloadedDataset(test_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_with_spsa(
    dataset: str = 'MNIST',
    model_type: str = 'simple',
    ibp_epochs: int = 3,
    spsa_epochs: int = 5,
    epsilon: float = 0.1,
    lambda_weight: float = 1.0,
    spsa_a: float = 0.001,
    spsa_c: float = 0.001,
    spsa_momentum: float = 0.0,
    spsa_weight_decay: float = 1e-4,
    spsa_grad_clip: float = 1.0,
    spsa_param_clip: float = 0.01,
    spsa_smoothing: float = 0.9,
    batch_size: int = 32,
    save_path: str = None,
    log_dir: str = './logs'
) -> Dict:
    """
    完整的混合训练流程（优化版）
    
    Args:
        dataset: 数据集名称
        model_type: 模型类型
        ibp_epochs: CROWN-IBP训练轮数
        spsa_epochs: SPSA微调轮数
        epsilon: 扰动范围
        lambda_weight: 鲁棒性损失权重
        spsa_a: SPSA学习率系数（推荐 0.001）
        spsa_c: SPSA扰动系数（推荐 0.001）
        spsa_momentum: SPSA动量系数（推荐 0.0）
        spsa_weight_decay: SPSA权重衰减系数
        spsa_grad_clip: SPSA梯度裁剪阈值（推荐 1.0）
        spsa_param_clip: 参数更新裁剪阈值（推荐 0.01）
        spsa_smoothing: 梯度平滑因子（推荐 0.9）
        batch_size: 批大小
        save_path: 模型保存路径
        log_dir: 日志目录
    
    Returns:
        训练结果字典
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Phase 1: CROWN-IBP Training ({ibp_epochs} epochs)")
    print(f"{'='*70}")
    
    base_model_path = save_path.replace('.pt', '_ibp.pt') if save_path else None
    
    model = train_model_with_ab_crown(
        dataset=dataset,
        model_type=model_type,
        epochs=ibp_epochs,
        batch_size=64,
        lr=0.001,
        epsilon=epsilon,
        lambda_weight=lambda_weight,
        use_ab_crown=True,
        bound_method='CROWN-IBP',
        loss_type='wcim',
        warmup_epochs=2,
        save_path=base_model_path
    )
    
    print(f"\n{'='*70}")
    print(f"Phase 2: SPSA Fine-tuning with alpha-CROWN ({spsa_epochs} epochs)")
    print(f"{'='*70}")
    
    train_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=batch_size,
        preload=True,
        num_workers=2
    )
    
    spsa_trainer = AlphaCROWNSPSATrainer(
        model, 
        epsilon=epsilon, 
        device=device,
        lambda_weight=lambda_weight,
        log_dir=log_dir
    )
    
    spsa_history = spsa_trainer.spsa_fine_tune(
        train_loader, test_loader,
        epochs=spsa_epochs,
        batch_size=batch_size // 2,
        a=spsa_a,
        c=spsa_c,
        momentum=spsa_momentum,
        weight_decay=spsa_weight_decay,
        grad_clip=spsa_grad_clip,
        param_clip=spsa_param_clip,
        smoothing_factor=spsa_smoothing
    )
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"\nFinal model saved to {save_path}")
    
    return {
        'model': model,
        'ibp_model_path': base_model_path,
        'final_model_path': save_path,
        'spsa_history': spsa_history,
        'ibp_epochs': ibp_epochs,
        'spsa_epochs': spsa_epochs,
        'epsilon': epsilon,
        'lambda_weight': lambda_weight
    }


def run_comparison_experiment(
    dataset: str = 'MNIST',
    model_type: str = 'simple',
    ibp_epochs: int = 3,
    spsa_epochs: int = 5,
    epsilon: float = 0.1,
    results_dir: str = './results'
):
    """
    对比实验：CROWN-IBP vs CROWN-IBP + SPSA
    
    Returns:
        对比结果DataFrame
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*70}")
    print(f"Comparison Experiment: IBP vs IBP+SPSA")
    print(f"{'='*70}")
    
    print(f"\n--- Training IBP-only model ---")
    ibp_model_path = os.path.join(results_dir, f'{dataset}_{model_type}_ibp_only_{timestamp}.pt')
    ibp_result = train_model_with_ab_crown(
        dataset=dataset,
        model_type=model_type,
        epochs=ibp_epochs + spsa_epochs,
        batch_size=64,
        lr=0.001,
        epsilon=epsilon,
        lambda_weight=1.0,
        use_ab_crown=True,
        bound_method='CROWN-IBP',
        loss_type='wcim',
        warmup_epochs=2,
        save_path=ibp_model_path
    )
    
    print(f"\n--- Training IBP+SPSA model ---")
    spsa_model_path = os.path.join(results_dir, f'{dataset}_{model_type}_ibp_spsa_{timestamp}.pt')
    spsa_result = train_with_spsa(
        dataset=dataset,
        model_type=model_type,
        ibp_epochs=ibp_epochs,
        spsa_epochs=spsa_epochs,
        epsilon=epsilon,
        save_path=spsa_model_path
    )
    
    from experiments import ABCrownVerifier, load_dataset
    
    results = []
    
    for model_name, model_path in [('IBP-only', ibp_model_path), ('IBP+SPSA', spsa_model_path)]:
        print(f"\n{'='*50}")
        print(f"Verifying {model_name} model")
        print(f"{'='*50}")
        
        model = get_mnist_model(model_type)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        verifier = ABCrownVerifier(model, dataset, timeout=30)
        samples = load_dataset(dataset, sample_size=30)
        
        df = verifier.verify_batch(samples, epsilon, method='alpha-crown')
        
        safe_count = (df['status'] == 'safe-incomplete').sum() + (df['status'] == 'safe').sum()
        unsafe_count = (df['status'] == 'unsafe-pgd').sum()
        robust_acc = safe_count / len(df) * 100
        avg_margin = df['margin'].mean() if df['margin'].notna().any() else 0.0
        
        print(f"  Safe: {safe_count}, Unsafe: {unsafe_count}, Robust Acc: {robust_acc:.2f}%")
        print(f"  Average Margin: {avg_margin:.4f}")
        
        results.append({
            'model_name': model_name,
            'robust_accuracy': robust_acc,
            'safe_count': int(safe_count),
            'unsafe_count': int(unsafe_count),
            'avg_margin': float(avg_margin),
            'total_samples': len(df),
            'epsilon': epsilon
        })
    
    results_df = pd.DataFrame(results)
    results_path = os.path.join(results_dir, f'spsa_comparison_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    print(f"\n{'='*70}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*70}")
    for _, row in results_df.iterrows():
        print(f"{row['model_name']}:")
        print(f"  Robust Accuracy: {row['robust_accuracy']:.2f}%")
        print(f"  Safe/Unsafe: {row['safe_count']}/{row['unsafe_count']}")
        print(f"  Average Margin: {row['avg_margin']:.4f}")
    
    improvement = results_df[results_df['model_name'] == 'IBP+SPSA']['robust_accuracy'].values[0] - \
                  results_df[results_df['model_name'] == 'IBP-only']['robust_accuracy'].values[0]
    print(f"\nSPSA Improvement: {improvement:+.2f}%")
    
    return results_df


def run_validation_experiments(
    dataset: str = 'MNIST',
    model_type: str = 'simple',
    ibp_epochs: int = 3,
    spsa_epochs: int = 5,
    epsilon: float = 0.1,
    num_runs: int = 3,
    results_dir: str = './results'
) -> pd.DataFrame:
    """
    运行多次独立验证实验，评估训练稳定性与收敛性
    
    Args:
        dataset: 数据集名称
        model_type: 模型类型
        ibp_epochs: CROWN-IBP训练轮数
        spsa_epochs: SPSA微调轮数
        epsilon: 扰动范围
        num_runs: 独立实验次数
        results_dir: 结果保存目录
    
    Returns:
        实验结果DataFrame
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*70}")
    print(f"Validation Experiments: {num_runs} Independent Runs")
    print(f"{'='*70}")
    
    all_results = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*70}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*70}")
        
        run_start_time = time.time()
        
        save_path = os.path.join(results_dir, f'{dataset}_{model_type}_run{run_idx+1}_{timestamp}.pt')
        
        try:
            result = train_with_spsa(
                dataset=dataset,
                model_type=model_type,
                ibp_epochs=ibp_epochs,
                spsa_epochs=spsa_epochs,
                epsilon=epsilon,
                save_path=save_path
            )
            
            spsa_history = result['spsa_history']
            
            final_alpha_crown_loss = spsa_history[-1]['alpha_crown_loss']
            final_test_acc = spsa_history[-1]['test_accuracy']
            total_time = sum(h['epoch_time'] for h in spsa_history)
            
            loss_decrease = spsa_history[0]['alpha_crown_loss'] - final_alpha_crown_loss
            loss_decrease_ratio = loss_decrease / spsa_history[0]['alpha_crown_loss'] if spsa_history[0]['alpha_crown_loss'] > 0 else 0
            
            all_results.append({
                'run': run_idx + 1,
                'final_alpha_crown_loss': final_alpha_crown_loss,
                'initial_alpha_crown_loss': spsa_history[0]['alpha_crown_loss'],
                'loss_decrease': loss_decrease,
                'loss_decrease_ratio': loss_decrease_ratio,
                'final_test_accuracy': final_test_acc,
                'total_training_time': total_time,
                'avg_epoch_time': total_time / spsa_epochs,
                'model_path': save_path
            })
            
            print(f"\nRun {run_idx + 1} completed:")
            print(f"  Final Alpha-CROWN Loss: {final_alpha_crown_loss:.4f}")
            print(f"  Loss Decrease: {loss_decrease:.4f} ({loss_decrease_ratio:.2%})")
            print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
            print(f"  Total Training Time: {total_time:.2f}s")
            
        except Exception as e:
            print(f"\nRun {run_idx + 1} failed: {e}")
            all_results.append({
                'run': run_idx + 1,
                'final_alpha_crown_loss': float('nan'),
                'initial_alpha_crown_loss': float('nan'),
                'loss_decrease': float('nan'),
                'loss_decrease_ratio': float('nan'),
                'final_test_accuracy': float('nan'),
                'total_training_time': float('nan'),
                'avg_epoch_time': float('nan'),
                'model_path': None,
                'error': str(e)
            })
    
    results_df = pd.DataFrame(all_results)
    
    summary_path = os.path.join(results_dir, f'validation_summary_{timestamp}.csv')
    results_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY ({num_runs} runs)")
    print(f"{'='*70}")
    print(results_df)
    
    valid_runs = results_df.dropna()
    if len(valid_runs) > 0:
        print(f"\nStatistics:")
        print(f"  Average Final Alpha-CROWN Loss: {valid_runs['final_alpha_crown_loss'].mean():.4f} ± {valid_runs['final_alpha_crown_loss'].std():.4f}")
        print(f"  Average Loss Decrease Ratio: {valid_runs['loss_decrease_ratio'].mean():.2%} ± {valid_runs['loss_decrease_ratio'].std():.2%}")
        print(f"  Average Final Test Accuracy: {valid_runs['final_test_accuracy'].mean():.2f}% ± {valid_runs['final_test_accuracy'].std():.2f}%")
        print(f"  Average Training Time: {valid_runs['total_training_time'].mean():.2f}s ± {valid_runs['total_training_time'].std():.2f}s")
    
    return results_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SPSA Training with alpha-CROWN (Optimized)')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset')
    parser.add_argument('--model', type=str, default='simple', help='Model type')
    parser.add_argument('--ibp_epochs', type=int, default=3, help='IBP training epochs')
    parser.add_argument('--spsa_epochs', type=int, default=5, help='SPSA fine-tuning epochs')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Perturbation epsilon')
    parser.add_argument('--spsa_a', type=float, default=0.001, help='SPSA learning rate coefficient')
    parser.add_argument('--spsa_c', type=float, default=0.001, help='SPSA perturbation coefficient')
    parser.add_argument('--spsa_momentum', type=float, default=0.0, help='SPSA momentum')
    parser.add_argument('--spsa_weight_decay', type=float, default=1e-4, help='SPSA weight decay')
    parser.add_argument('--spsa_grad_clip', type=float, default=1.0, help='SPSA gradient clip')
    parser.add_argument('--spsa_param_clip', type=float, default=0.01, help='SPSA parameter update clip')
    parser.add_argument('--spsa_smoothing', type=float, default=0.9, help='SPSA gradient smoothing factor')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--compare', action='store_true', help='Run comparison experiment')
    parser.add_argument('--validate', action='store_true', help='Run validation experiments')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of validation runs')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    
    args = parser.parse_args()
    
    if args.validate:
        run_validation_experiments(
            dataset=args.dataset,
            model_type=args.model,
            ibp_epochs=args.ibp_epochs,
            spsa_epochs=args.spsa_epochs,
            epsilon=args.epsilon,
            num_runs=args.num_runs,
            results_dir=args.results_dir
        )
    elif args.compare:
        run_comparison_experiment(
            dataset=args.dataset,
            model_type=args.model,
            ibp_epochs=args.ibp_epochs,
            spsa_epochs=args.spsa_epochs,
            epsilon=args.epsilon,
            results_dir=args.results_dir
        )
    else:
        train_with_spsa(
            dataset=args.dataset,
            model_type=args.model,
            ibp_epochs=args.ibp_epochs,
            spsa_epochs=args.spsa_epochs,
            epsilon=args.epsilon,
            spsa_a=args.spsa_a,
            spsa_c=args.spsa_c,
            spsa_momentum=args.spsa_momentum,
            spsa_weight_decay=args.spsa_weight_decay,
            spsa_grad_clip=args.spsa_grad_clip,
            spsa_param_clip=args.spsa_param_clip,
            spsa_smoothing=args.spsa_smoothing,
            batch_size=args.batch_size,
            log_dir=args.log_dir
        )
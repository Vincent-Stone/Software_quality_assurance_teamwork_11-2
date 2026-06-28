"""
基于auto_LiRPA的鲁棒性训练脚本，使用SPSA方法计算alpha-CROWN边界梯度

功能说明：
1. 继承simple_training.py的代码架构和参数配置
2. 实现SPSA（Simultaneous Perturbation Stochastic Approximation）梯度估计方法
3. 将SPSA梯度计算与alpha-CROWN边界计算整合到训练流程中
4. 支持两阶段训练策略：CROWN-IBP快速初始化 + SPSA+alpha-CROWN微调

SPSA核心原理：
    g_hat_i = [L(θ + c·Δ) - L(θ - c·Δ)] / (2·c·Δ_i)
    其中 Δ 是随机扰动向量（每个元素为±1），c 是扰动尺度

与alpha-CROWN的整合逻辑：
    1. 使用alpha-CROWN（CROWN-Optimized）计算输出边界
    2. 基于边界计算鲁棒性损失（WCIM损失）
    3. 使用SPSA估计损失相对于模型参数的梯度
    4. 使用估计的梯度更新模型参数
"""

import time
import random
import multiprocessing
import argparse
import numpy as np
import os
import sys
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler

from robustness_verification.models import get_mnist_model, get_cifar10_model


parser = argparse.ArgumentParser()

# 基础训练参数（继承自simple_training.py）
parser.add_argument("--verify", action="store_true", help='verification mode, do not train')
parser.add_argument("--load", type=str, default="", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="MNIST", choices=["MNIST", "CIFAR"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--eps", type=float, default=0.1, help='Target training epsilon')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN", "CROWN-FAST"], help='method of bound analysis for Phase 1')
parser.add_argument("--model", type=str, default="simple", help='model name (tiny, simple, medium, deep)')
parser.add_argument("--num_epochs", type=int, default=100, help='number of total epochs')
parser.add_argument("--batch_size", type=int, default=128, help='batch size')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate for Phase 1')
parser.add_argument("--scheduler_name", type=str, default="SmoothedScheduler",
                    choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler", "FixedScheduler"], help='epsilon scheduler')
parser.add_argument("--scheduler_opts", type=str, default="start=3,length=60", help='options for epsilon scheduler')
parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options')
parser.add_argument("--conv_mode", type=str, choices=["matrix", "patches"], default="patches")
parser.add_argument("--save_model", type=str, default='')

# SPSA训练参数
parser.add_argument("--spsa_epochs", type=int, default=5, help='number of SPSA fine-tuning epochs')
parser.add_argument("--spsa_a", type=float, default=0.001, help='SPSA learning rate coefficient')
parser.add_argument("--spsa_c", type=float, default=0.001, help='SPSA perturbation coefficient')
parser.add_argument("--spsa_alpha", type=float, default=0.602, help='SPSA learning rate decay exponent')
parser.add_argument("--spsa_gamma", type=float, default=0.101, help='SPSA perturbation scale decay exponent')
parser.add_argument("--spsa_momentum", type=float, default=0.0, help='SPSA momentum coefficient')
parser.add_argument("--spsa_weight_decay", type=float, default=1e-4, help='SPSA weight decay')
parser.add_argument("--spsa_grad_clip", type=float, default=1.0, help='SPSA gradient clipping threshold')
parser.add_argument("--spsa_param_clip", type=float, default=0.01, help='SPSA parameter update clip')
parser.add_argument("--spsa_smoothing", type=float, default=0.9, help='SPSA gradient smoothing factor')
parser.add_argument("--spsa_batch_size", type=int, default=32, help='batch size for SPSA loss evaluation')
parser.add_argument("--use_spsa", action="store_true", help='Enable SPSA fine-tuning after Phase 1')

args = parser.parse_args()


class SPSAOptimizer:
    """
    SPSA（Simultaneous Perturbation Stochastic Approximation）优化器
    
    核心特点：
    1. 梯度无关的优化方法，只需两次函数评估即可估计梯度
    2. 适用于alpha-CROWN等无法直接反向传播梯度的边界计算方法
    3. 支持动量、权重衰减、梯度裁剪等优化技术
    
    SPSA梯度估计公式：
        g_hat_i = [L(θ + c·Δ) - L(θ - c·Δ)] / (2·c·Δ_i)
    
    参数更新公式：
        θ_{k+1} = θ_k - a_k · g_hat(θ_k)
    """
    
    def __init__(self, parameters, a: float = 0.001, c: float = 0.001, 
                 alpha: float = 0.602, gamma: float = 0.101,
                 momentum: float = 0.0, weight_decay: float = 1e-4,
                 grad_clip: float = 1.0, param_clip: float = 0.01,
                 smoothing_factor: float = 0.9):
        """
        参数说明：
        - a: 学习率衰减系数（推荐 0.001）
        - c: 扰动尺度衰减系数（推荐 0.001）
        - alpha: 学习率衰减指数（推荐 0.602）
        - gamma: 扰动尺度衰减指数（推荐 0.101）
        - momentum: 动量系数（SPSA通常不使用动量，推荐0.0）
        - weight_decay: 权重衰减系数
        - grad_clip: 梯度裁剪阈值
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
        """计算当前步长 a_k = a / (k+1)^alpha"""
        return self.a / (self.k + 1) ** self.alpha
    
    def get_perturbation_scale(self):
        """计算当前扰动尺度 c_k = c / (k+1)^gamma，支持自适应调整"""
        c_k = self.c / (self.k + 1) ** self.gamma
        
        if len(self.loss_history) > 20:
            loss_std = np.std(list(self.loss_history))
            if loss_std > 1.0:
                c_k = min(c_k, c_k * 0.3)
            elif loss_std > 0.5:
                c_k = min(c_k, c_k * 0.5)
            elif loss_std < 0.05:
                c_k = max(c_k, c_k * 1.2)
        
        c_k = max(c_k, 1e-6)
        return c_k
    
    def perturb(self):
        """生成随机扰动向量 Δ（每个元素为±1）"""
        perturbations = []
        for param in self.parameters:
            delta = torch.sign(torch.randn_like(param))
            perturbations.append(delta)
        return perturbations
    
    def compute_grad_norm(self, grad_estimate: float, perturbations) -> float:
        """计算梯度估计的范数（按参数数量归一化）"""
        total_norm = 0.0
        count = 0
        for delta in perturbations:
            total_norm += (delta ** 2).sum().item()
            count += delta.numel()
        return abs(grad_estimate) * np.sqrt(total_norm / max(count, 1))
    
    def step(self, loss_plus: float, loss_minus: float, perturbations):
        """
        根据两次损失评估更新参数
        
        Args:
            loss_plus: L(θ + c·Δ)，参数正扰动后的损失
            loss_minus: L(θ - c·Δ)，参数负扰动后的损失
            perturbations: 扰动向量列表
            
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
    
    def get_stats(self):
        """获取优化器统计信息"""
        return {
            'current_step': self.k,
            'current_step_size': self.get_step_size(),
            'current_perturbation_scale': self.get_perturbation_scale(),
            'avg_grad_norm': np.mean(list(self.grad_norm_history)) if self.grad_norm_history else 0,
            'avg_loss': np.mean(list(self.loss_history)) if self.loss_history else 0,
            'avg_loss_diff': np.mean(list(self.loss_diff_history)) if self.loss_diff_history else 0
        }


class AlphaCROWNSPSALoss:
    """
    使用alpha-CROWN边界计算鲁棒性损失，并通过SPSA方法估计梯度
    
    损失计算方法：WCIM（Worst-Case Interval Margin）
        margin_t = lb_t - mean(ub_{j!=t})
        worst_margin = min_t(margin_t)
        loss = ReLU(-worst_margin)
    
    与SPSA的整合方式：
        1. 使用alpha-CROWN（CROWN-Optimized）计算输出边界
        2. 基于边界计算WCIM损失
        3. 保存原始参数，对参数进行正负扰动
        4. 在扰动后的参数上重新计算损失
        5. 使用SPSA公式估计梯度
    """
    
    def __init__(self, model, epsilon, device, bound_opts=None):
        """
        Args:
            model: PyTorch模型
            epsilon: 扰动范围
            device: 计算设备
            bound_opts: BoundedModule配置选项
        """
        self.model = model
        self.epsilon = epsilon
        self.device = device
        
        if bound_opts is None:
            bound_opts = {'conv_mode': 'patches', 'deterministic': True}
        
        dummy_input = torch.randn(1, 1, 28, 28).to(device) if args.data == 'MNIST' else \
                      torch.randn(1, 3, 32, 32).to(device)
        self.bounded_model = BoundedModule(model, dummy_input, bound_opts=bound_opts, device=device)
        
        self.criterion = CrossEntropyLoss()
    
    def compute_wcim_loss(self, data, target):
        """
        计算WCIM（Worst-Case Interval Margin）损失
        
        Args:
            data: 输入数据，shape (batch_size, channels, height, width)
            target: 目标标签，shape (batch_size,)
        
        Returns:
            WCIM损失值（标量）
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
            print(f"[Warning] compute_bounds failed: {e}")
            return float('inf')
    
    def compute_standard_loss(self, data, target):
        """计算标准分类损失"""
        output = self.model(data)
        loss = self.criterion(output, target)
        return float(loss.item())


def TrainPhase1(model, t, loader, eps_scheduler, norm, train, opt, bound_type, device):
    """
    Phase 1: 使用CROWN-IBP进行快速梯度训练
    
    继承自simple_training.py的Train函数，用于快速初始化模型参数
    """
    num_class = 10
    meter = MultiAverageMeter()
    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model.eval()
        eps_scheduler.eval()

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        
        batch_method = "robust"
        if eps < 1e-20:
            batch_method = "natural"
        if train:
            opt.zero_grad()
        
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), num_class - 1, num_class))
        
        if norm == np.inf:
            data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / loader.std).view(1,-1,1,1), data_max)
            data_lb = torch.max(data - (eps / loader.std).view(1,-1,1,1), data_min)
        else:
            data_ub = data_lb = data

        if device == 'cuda' and list(model.parameters())[0].is_cuda:
            data, labels, c = data.cuda(), labels.cuda(), c.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        if norm > 0:
            ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        elif norm == 0:
            ptb = PerturbationLpNorm(norm=0, eps=eps_scheduler.get_max_eps(), ratio=eps_scheduler.get_eps()/eps_scheduler.get_max_eps())
        x = BoundedTensor(data, ptb)

        output = model(x)
        regular_ce = CrossEntropyLoss()(output, labels)
        meter.update('CE', regular_ce.item(), x.size(0))
        meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / x.size(0), x.size(0))

        if batch_method == "robust":
            if bound_type == "IBP":
                lb, ub = model.compute_bounds(IBP=True, C=c, method=None)
            elif bound_type == "CROWN":
                lb, ub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
            elif bound_type == "CROWN-IBP":
                factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
                ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
                if factor < 1e-5:
                    lb = ilb
                else:
                    clb, cub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                    lb = clb * factor + ilb * (1 - factor)
            elif bound_type == "CROWN-FAST":
                lb, ub = model.compute_bounds(IBP=True, C=c, method=None)
                lb, ub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)

            lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
        
        if batch_method == "robust":
            loss = robust_ce
        elif batch_method == "natural":
            loss = regular_ce
        
        if train:
            loss.backward()
            eps_scheduler.update_loss(loss.item() - regular_ce.item())
            opt.step()
        
        meter.update('Loss', loss.item(), data.size(0))
        if batch_method != "natural":
            meter.update('Robust_CE', robust_ce.item(), data.size(0))
            meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)
        
        if i % 50 == 0 and train:
            print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
    
    print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
    return meter


def TrainPhase2(model, train_loader, test_loader, device, args):
    """
    Phase 2: 使用SPSA方法进行alpha-CROWN边界微调
    
    核心流程：
    1. 初始化SPSA优化器
    2. 对每个batch：
       a. 保存原始参数
       b. 生成随机扰动Δ
       c. 计算L(θ + c·Δ)和L(θ - c·Δ)
       d. 使用SPSA公式估计梯度
       e. 更新参数
    3. 定期评估模型性能
    """
    print("\n" + "="*70)
    print(f"Phase 2: SPSA Fine-tuning with alpha-CROWN ({args.spsa_epochs} epochs)")
    print("="*70)
    
    criterion = CrossEntropyLoss()
    spsa_loss_fn = AlphaCROWNSPSALoss(model, args.eps, device)
    
    optimizer = SPSAOptimizer(
        model.parameters(),
        a=args.spsa_a,
        c=args.spsa_c,
        alpha=args.spsa_alpha,
        gamma=args.spsa_gamma,
        momentum=args.spsa_momentum,
        weight_decay=args.spsa_weight_decay,
        grad_clip=args.spsa_grad_clip,
        param_clip=args.spsa_param_clip,
        smoothing_factor=args.spsa_smoothing
    )
    
    history = []
    start_time = time.time()
    
    for epoch in range(args.spsa_epochs):
        model.train()
        total_alpha_crown_loss = 0
        total_standard_loss = 0
        batches_processed = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            if data.size(0) < args.spsa_batch_size:
                continue
            
            data = data[:args.spsa_batch_size]
            target = target[:args.spsa_batch_size]
            
            original_params = [param.data.clone() for param in model.parameters()]
            
            perturbations = optimizer.perturb()
            c_k = optimizer.get_perturbation_scale()
            
            for param, delta in zip(model.parameters(), perturbations):
                param.data.add_(c_k * delta)
            
            loss_plus = spsa_loss_fn.compute_wcim_loss(data, target)
            
            if loss_plus == float('inf'):
                for param, orig in zip(model.parameters(), original_params):
                    param.data.copy_(orig)
                continue
            
            for param, orig, delta in zip(model.parameters(), original_params, perturbations):
                param.data.copy_(orig)
                param.data.sub_(c_k * delta)
            
            loss_minus = spsa_loss_fn.compute_wcim_loss(data, target)
            
            if loss_minus == float('inf'):
                for param, orig in zip(model.parameters(), original_params):
                    param.data.copy_(orig)
                continue
            
            for param, orig in zip(model.parameters(), original_params):
                param.data.copy_(orig)
            
            opt_stats = optimizer.step(loss_plus, loss_minus, perturbations)
            
            std_loss = spsa_loss_fn.compute_standard_loss(data, target)
            
            total_alpha_crown_loss += loss_plus
            total_standard_loss += std_loss
            batches_processed += 1
            
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{args.spsa_epochs}, Batch {batch_idx}: "
                      f"Alpha-CROWN Loss={loss_plus:.4f}, "
                      f"Standard Loss={std_loss:.4f}, "
                      f"Grad Norm={opt_stats['grad_norm']:.4f}, "
                      f"Step Size={opt_stats['step_size']:.6f}, "
                      f"Perturb Scale={opt_stats['perturbation_scale']:.6f}, "
                      f"Time={elapsed:.2f}s")
        
        epoch_time = time.time() - epoch_start_time
        
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += data.size(0)
        
        test_acc = 100. * test_correct / test_total
        avg_test_loss = test_loss / test_total
        avg_alpha_crown_loss = total_alpha_crown_loss / batches_processed if batches_processed > 0 else 0
        
        optimizer_stats = optimizer.get_stats()
        
        history.append({
            'epoch': epoch + 1,
            'alpha_crown_loss': avg_alpha_crown_loss,
            'standard_loss': total_standard_loss / batches_processed if batches_processed > 0 else 0,
            'test_accuracy': test_acc,
            'test_loss': avg_test_loss,
            'spsa_step': optimizer.k,
            'epoch_time': epoch_time,
            'avg_grad_norm': optimizer_stats['avg_grad_norm'],
            'current_step_size': optimizer_stats['current_step_size'],
            'current_perturb_scale': optimizer_stats['current_perturbation_scale']
        })
        
        print(f"\n  [Phase 2] Epoch {epoch + 1}/{args.spsa_epochs}:")
        print(f"    Alpha-CROWN Loss: {avg_alpha_crown_loss:.4f}")
        print(f"    Standard Loss: {history[-1]['standard_loss']:.4f}")
        print(f"    Test Accuracy: {test_acc:.2f}%")
        print(f"    Test Loss: {avg_test_loss:.4f}")
        print(f"    Avg Grad Norm: {optimizer_stats['avg_grad_norm']:.4f}")
        print(f"    Epoch Time: {epoch_time:.2f}s")
        print(f"    Total Time: {time.time() - start_time:.2f}s")
    
    print(f"\n{'='*70}")
    print("PHASE 2 TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total Training Time: {time.time() - start_time:.2f}s")
    print(f"Final Alpha-CROWN Loss: {history[-1]['alpha_crown_loss']:.4f}")
    print(f"Final Test Accuracy: {history[-1]['test_accuracy']:.2f}%")
    
    return history


def main(args):
    """主函数：执行完整的两阶段训练流程"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.data == 'MNIST':
        model_ori = get_mnist_model(args.model)
        dummy_input = torch.randn(2, 1, 28, 28)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
    else:
        model_ori = get_cifar10_model(args.model)
        dummy_input = torch.randn(2, 3, 32, 32)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        train_data = datasets.CIFAR10("./data", train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize]))
        test_data = datasets.CIFAR10("./data", train=False, download=True, 
                transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    
    if args.data == 'MNIST':
        train_data.mean = test_data.mean = torch.tensor([0.1307])
        train_data.std = test_data.std = torch.tensor([0.3081])
    elif args.data == 'CIFAR':
        train_data.mean = test_data.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        train_data.std = test_data.std = torch.tensor([0.2023, 0.1994, 0.2010])

    if args.load:
        state_dict = torch.load(args.load, map_location=device, weights_only=True)
        model_ori.load_state_dict(state_dict)
        print(f"Pretrained model loaded from {args.load}")

    bound_opts = {'activation_bound_option': args.bound_opts, 'conv_mode': args.conv_mode}
    model = BoundedModule(model_ori, dummy_input, bound_opts=bound_opts, device=device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)
    
    print("Model structure: \n", str(model_ori))

    if args.verify:
        eps_scheduler = FixedScheduler(args.eps)
        with torch.no_grad():
            TrainPhase1(model, 1, test_data, eps_scheduler, norm, False, None, args.bound_type, device)
    else:
        timer = 0.0
        
        # ========== Phase 1: CROWN-IBP快速训练 ==========
        print("\n" + "="*70)
        print(f"Phase 1: CROWN-IBP Training ({args.num_epochs} epochs)")
        print("="*70)
        
        for t in range(1, args.num_epochs + 1):
            if eps_scheduler.reached_max_eps():
                lr_scheduler.step()
            print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
            start_time = time.time()
            TrainPhase1(model, t, train_data, eps_scheduler, norm, True, opt, args.bound_type, device)
            epoch_time = time.time() - start_time
            timer += epoch_time
            print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            print("Evaluating...")
            with torch.no_grad():
                TrainPhase1(model, t, test_data, eps_scheduler, norm, False, None, args.bound_type, device)
        
        # 保存Phase 1模型
        if args.save_model:
            ibp_model_path = args.save_model.replace('.pt', '_ibp.pt')
            torch.save({'state_dict': model_ori.state_dict(), 'epoch': args.num_epochs}, ibp_model_path)
            print(f"\nPhase 1 model saved to {ibp_model_path}")
        
        # ========== Phase 2: SPSA微调（可选） ==========
        if args.use_spsa:
            TrainPhase2(model_ori, train_data, test_data, device, args)
        
        # 保存最终模型
        if args.save_model:
            torch.save({'state_dict': model_ori.state_dict(), 'epoch': args.num_epochs + args.spsa_epochs}, args.save_model)
            print(f"\nFinal model saved to {args.save_model}")


if __name__ == "__main__":
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    main(args)
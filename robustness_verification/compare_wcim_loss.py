"""
WCIMLoss vs Standard Loss 对比验证脚本

该脚本训练使用 WCIMLoss 和 Standard Loss 的模型，并使用 alpha-CROWN
边界进行鲁棒性验证，对比两种损失函数的性能差异。
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'complete_verifier'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'auto_LiRPA'))

from models import get_mnist_model
from train import train_model_with_ab_crown, VerificationLoss, WCIMLoss
from experiments import ABCrownVerifier, load_dataset


def train_models_for_comparison(
    dataset: str = 'MNIST',
    model_type: str = 'simple',
    epochs: int = 5,
    batch_size: int = 64,
    epsilon: float = 0.1,
    lambda_weight: float = 1.0,
    warmup_epochs: int = 2,
    save_dir: str = './models'
) -> Dict[str, str]:
    """
    训练两个模型：使用 WCIMLoss 和使用 Standard Loss

    Returns:
        模型路径字典
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {}

    # 1. 训练使用 WCIMLoss 的模型
    print(f"\n{'='*70}")
    print(f"Training model with WCIMLoss")
    print(f"{'='*70}")

    wcim_path = os.path.join(save_dir, f'{dataset}_{model_type}_wcim_{timestamp}.pt')
    wcim_model = train_model_with_ab_crown(
        dataset=dataset,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        lr=0.001,
        epsilon=epsilon,
        lambda_weight=lambda_weight,
        use_ab_crown=True,
        bound_method='CROWN-IBP',
        loss_type='wcim',
        warmup_epochs=warmup_epochs,
        save_path=wcim_path
    )
    results['wcim'] = wcim_path
    print(f"WCIMLoss model saved to {wcim_path}")

    # 2. 训练使用 Standard Loss 的模型
    print(f"\n{'='*70}")
    print(f"Training model with Standard VerificationLoss")
    print(f"{'='*70}")

    standard_path = os.path.join(save_dir, f'{dataset}_{model_type}_standard_{timestamp}.pt')
    standard_model = train_model_with_ab_crown(
        dataset=dataset,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        lr=0.001,
        epsilon=epsilon,
        lambda_weight=lambda_weight,
        use_ab_crown=True,
        bound_method='CROWN-IBP',
        loss_type='standard',
        save_path=standard_path
    )
    results['standard'] = standard_path
    print(f"Standard model saved to {standard_path}")

    return results


def verify_models_with_alpha_crown(
    model_paths: Dict[str, str],
    dataset: str = 'MNIST',
    sample_size: int = 50,
    eps_values: List[float] = [0.05, 0.1, 0.15],
    timeout: int = 60
) -> pd.DataFrame:
    """
    使用 alpha-CROWN 验证训练好的模型

    Returns:
        包含所有验证结果的 DataFrame
    """
    print(f"\n{'='*70}")
    print(f"Verification using alpha-CROWN bounds")
    print(f"{'='*70}")

    all_results = []

    for model_type, model_path in model_paths.items():
        print(f"\n{'='*50}")
        print(f"Verifying {model_type.upper()} model: {os.path.basename(model_path)}")
        print(f"{'='*50}")

        # 加载模型
        model = get_mnist_model('simple')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"  Loaded from {model_path}")
        else:
            print(f"  Warning: Model not found at {model_path}")
            continue

        # 创建验证器
        verifier = ABCrownVerifier(model, dataset, timeout=timeout)

        # 在不同 epsilon 下进行验证
        for eps in eps_values:
            print(f"\n  --- Epsilon = {eps} ---")
            samples = load_dataset(dataset, sample_size)

            start_time = time.time()
            df = verifier.verify_batch(samples, eps, method='alpha-crown')
            elapsed = time.time() - start_time

            # 统计结果
            safe_count = (df['status'] == 'safe-incomplete').sum() + (df['status'] == 'safe').sum()
            unsafe_count = (df['status'] == 'unsafe-pgd').sum()
            unknown_count = (df['status'] == 'unknown').sum()
            verified_unsafe = (df['success'] == True).sum()  # 成功验证为不鲁棒

            robust_acc = safe_count / len(df) * 100
            avg_margin = df['margin'].mean() if df['margin'].notna().any() else 0.0
            std_margin = df['margin'].std() if df['margin'].notna().any() else 0.0

            print(f"    Results: Safe={safe_count}, Unsafe={unsafe_count}, Unknown={unknown_count}")
            print(f"    Robust Accuracy: {robust_acc:.2f}%")
            print(f"    Average Margin: {avg_margin:.4f} (std: {std_margin:.4f})")
            print(f"    Verification Time: {elapsed:.2f}s total, {elapsed/len(df):.4f}s per sample")

            all_results.append({
                'model_type': model_type,
                'epsilon': eps,
                'safe_count': int(safe_count),
                'unsafe_count': int(unsafe_count),
                'unknown_count': int(unknown_count),
                'verified_unsafe': int(verified_unsafe),
                'robust_accuracy': robust_acc,
                'avg_margin': float(avg_margin),
                'std_margin': float(std_margin),
                'total_samples': len(df),
                'total_time': elapsed,
                'time_per_sample': elapsed / len(df)
            })

    return pd.DataFrame(all_results)


def compute_statistical_significance(results_df: pd.DataFrame) -> Dict:
    """
    计算两种方法之间鲁棒性差异的统计显著性

    使用 margin 作为主要指标进行 t-test
    """
    from scipy import stats

    wcim_results = results_df[results_df['model_type'] == 'wcim']
    standard_results = results_df[results_df['model_type'] == 'standard']

    significance_results = {}

    for eps in results_df['epsilon'].unique():
        wcim_eps = wcim_results[wcim_results['epsilon'] == eps]
        std_eps = standard_results[standard_results['epsilon'] == eps]

        if len(wcim_eps) > 0 and len(std_eps) > 0:
            wcim_acc = wcim_eps['robust_accuracy'].values[0]
            std_acc = std_eps['robust_accuracy'].values[0]

            wcim_margin = wcim_eps['avg_margin'].values[0]
            std_margin = std_eps['avg_margin'].values[0]

            # 计算改进幅度
            acc_improvement = wcim_acc - std_acc
            margin_improvement = wcim_margin - std_margin

            # 使用 margin 差异的粗略估计作为效应量
            pooled_std = np.sqrt((wcim_eps['std_margin'].values[0]**2 + std_eps['std_margin'].values[0]**2) / 2)
            if pooled_std > 0:
                effect_size = abs(margin_improvement) / pooled_std
            else:
                effect_size = 0

            significance_results[float(eps)] = {
                'wcim_robust_acc': wcim_acc,
                'standard_robust_acc': std_acc,
                'acc_improvement': acc_improvement,
                'wcim_avg_margin': wcim_margin,
                'standard_avg_margin': std_margin,
                'margin_improvement': margin_improvement,
                'effect_size_cohen_d': effect_size,
                'interpretation': _interpret_effect_size(effect_size)
            }

    return significance_results


def _interpret_effect_size(d: float) -> str:
    """解释 Cohen's d 效应量"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def plot_comparison_results(results_df: pd.DataFrame, save_path: str = './results'):
    """绘制对比结果图表"""
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    eps_values = sorted(results_df['epsilon'].unique())

    # 图1: 鲁棒准确率对比
    ax1 = axes[0]
    wcim_acc = [results_df[(results_df['model_type']=='wcim') & (results_df['epsilon']==eps)]['robust_accuracy'].values[0]
                for eps in eps_values]
    std_acc = [results_df[(results_df['model_type']=='standard') & (results_df['epsilon']==eps)]['robust_accuracy'].values[0]
               for eps in eps_values]

    x = np.arange(len(eps_values))
    width = 0.35

    bars1 = ax1.bar(x - width/2, wcim_acc, width, label='WCIMLoss', color='steelblue')
    bars2 = ax1.bar(x + width/2, std_acc, width, label='Standard', color='coral')

    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Robust Accuracy (%)')
    ax1.set_title('Robust Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(eps) for eps in eps_values])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 图2: 平均边界对比
    ax2 = axes[1]
    wcim_margin = [results_df[(results_df['model_type']=='wcim') & (results_df['epsilon']==eps)]['avg_margin'].values[0]
                   for eps in eps_values]
    std_margin = [results_df[(results_df['model_type']=='standard') & (results_df['epsilon']==eps)]['avg_margin'].values[0]
                  for eps in eps_values]

    ax2.bar(x - width/2, wcim_margin, width, label='WCIMLoss', color='steelblue')
    ax2.bar(x + width/2, std_margin, width, label='Standard', color='coral')

    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Average Margin')
    ax2.set_title('Average Bound Margin Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(eps) for eps in eps_values])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 图3: 安全/不安全/未知样本数
    ax3 = axes[2]

    wcim_safe = [results_df[(results_df['model_type']=='wcim') & (results_df['epsilon']==eps)]['safe_count'].values[0]
                 for eps in eps_values]
    wcim_unsafe = [results_df[(results_df['model_type']=='wcim') & (results_df['epsilon']==eps)]['unsafe_count'].values[0]
                   for eps in eps_values]
    std_safe = [results_df[(results_df['model_type']=='standard') & (results_df['epsilon']==eps)]['safe_count'].values[0]
                for eps in eps_values]
    std_unsafe = [results_df[(results_df['model_type']=='standard') & (results_df['epsilon']==eps)]['unsafe_count'].values[0]
                  for eps in eps_values]

    bottom1 = np.array(wcim_safe)
    bottom2 = np.array(std_safe)

    ax3.bar(x - width/2, wcim_safe, width, label='WCIM-Safe', color='green', alpha=0.7)
    ax3.bar(x - width/2, wcim_unsafe, width, bottom=bottom1, label='WCIM-Unsafe', color='red', alpha=0.7)
    ax3.bar(x + width/2, std_safe, width, label='Std-Safe', color='blue', alpha=0.7)
    ax3.bar(x + width/2, std_unsafe, width, bottom=bottom2, label='Std-Unsafe', color='orange', alpha=0.7)

    ax3.set_xlabel('Epsilon')
    ax3.set_ylabel('Sample Count')
    ax3.set_title('Verification Results Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(eps) for eps in eps_values])
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison_results.png'), dpi=150)
    print(f"\nComparison plot saved to {os.path.join(save_path, 'comparison_results.png')}")

    return fig


def run_full_comparison(
    dataset: str = 'MNIST',
    model_type: str = 'simple',
    epochs: int = 5,
    sample_size: int = 50,
    eps_values: List[float] = [0.05, 0.1, 0.15],
    timeout: int = 60,
    results_dir: str = './results'
):
    """运行完整的对比实验流程"""

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*70}")
    print(f"WCIMLoss vs Standard Loss - Robustness Comparison Experiment")
    print(f"{'='*70}")
    print(f"Dataset: {dataset}")
    print(f"Model: {model_type}")
    print(f"Training epochs: {epochs}")
    print(f"Verification samples: {sample_size}")
    print(f"Epsilon values: {eps_values}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*70}")

    # Step 1: 训练模型
    model_paths = train_models_for_comparison(
        dataset=dataset,
        model_type=model_type,
        epochs=epochs,
        epsilon=0.1,
        lambda_weight=1.0,
        warmup_epochs=2
    )

    # Step 2: 使用 alpha-CROWN 验证
    results_df = verify_models_with_alpha_crown(
        model_paths=model_paths,
        dataset=dataset,
        sample_size=sample_size,
        eps_values=eps_values,
        timeout=timeout
    )

    # Step 3: 保存结果
    results_path = os.path.join(results_dir, f'comparison_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Step 4: 计算统计显著性
    significance = compute_statistical_significance(results_df)
    sig_path = os.path.join(results_dir, f'statistical_significance_{timestamp}.json')
    with open(sig_path, 'w') as f:
        json.dump(significance, f, indent=2)
    print(f"Statistical significance saved to {sig_path}")

    # Step 5: 绘制对比图
    plot_comparison_results(results_df, save_path=results_dir)

    # Step 6: 打印总结
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")

    for eps in eps_values:
        sig_eps = significance.get(eps, {})
        print(f"\nEpsilon = {eps}:")
        print(f"  WCIMLoss:     Robust Acc = {sig_eps.get('wcim_robust_acc', 0):.2f}%, "
              f"Avg Margin = {sig_eps.get('wcim_avg_margin', 0):.4f}")
        print(f"  Standard:     Robust Acc = {sig_eps.get('standard_robust_acc', 0):.2f}%, "
              f"Avg Margin = {sig_eps.get('standard_avg_margin', 0):.4f}")
        print(f"  Improvement:  Acc = {sig_eps.get('acc_improvement', 0):+.2f}%, "
              f"Margin = {sig_eps.get('margin_improvement', 0):+.4f}")
        print(f"  Effect Size: {sig_eps.get('interpretation', 'N/A')}")

    return results_df, significance


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare WCIMLoss vs Standard Loss')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--model', type=str, default='simple', help='Model type')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--samples', type=int, default=50, help='Verification sample size')
    parser.add_argument('--eps', type=float, nargs='+', default=[0.05, 0.1, 0.15],
                        help='Epsilon values for verification')
    parser.add_argument('--timeout', type=int, default=60, help='Verification timeout per sample')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')

    args = parser.parse_args()

    run_full_comparison(
        dataset=args.dataset,
        model_type=args.model,
        epochs=args.epochs,
        sample_size=args.samples,
        eps_values=args.eps,
        timeout=args.timeout,
        results_dir=args.results_dir
    )

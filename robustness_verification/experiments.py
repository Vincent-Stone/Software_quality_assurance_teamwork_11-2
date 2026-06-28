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
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'complete_verifier'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'auto_LiRPA'))

from models import get_mnist_model, get_cifar10_model
from api import ABCrownSolver, VerificationSpec, input_vars, output_vars, ConfigBuilder


class ABCrownVerifier:
    """使用真正的αβ-CROWN进行模型鲁棒性验证"""
    
    def __init__(self, model, dataset_name: str, device='cpu', timeout=60):
        self.model = model
        self.dataset_name = dataset_name
        self.device = device
        self.timeout = timeout
        
        if dataset_name == 'MNIST':
            self.input_shape = (1, 1, 28, 28)
            self.num_classes = 10
        elif dataset_name == 'CIFAR10':
            self.input_shape = (1, 3, 32, 32)
            self.num_classes = 10
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.model.eval()
    
    def verify_sample(self, image: torch.Tensor, label: int, epsilon: float, method: str = 'alpha-crown') -> Dict:
        """验证单个样本的鲁棒性"""
        
        image_np = image.squeeze().numpy()
        
        center = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)
        
        if self.dataset_name == 'MNIST':
            mean = np.array([0.1307])
            std = np.array([0.3081])
        else:
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
        
        lower = center - epsilon
        upper = center + epsilon
        lower = torch.clamp(lower, 0, 1)
        upper = torch.clamp(upper, 0, 1)
        
        target_class = label
        
        x_vars = input_vars(self.input_shape)
        y_vars = output_vars(self.num_classes)
        
        clauses = []
        for c in range(self.num_classes):
            if c == target_class:
                continue
            C = torch.zeros(1, self.num_classes, dtype=torch.float32)
            C[0, target_class] = 1
            C[0, c] = -1
            rhs = torch.zeros(1, dtype=torch.float32)
            clauses.append((C, rhs))
        
        try:
            spec = VerificationSpec.build_from_input_bounds(lower, upper, clauses)
        except Exception as e:
            print(f"Warning: Failed to build spec: {e}")
            return {'status': 'error', 'margin': None, 'time': 0}
        
        config = ConfigBuilder.from_defaults().to_dict()
        config['general']['device'] = 'cpu'
        config['general']['deterministic'] = True
        config['general']['complete_verifier'] = 'skip'
        config['general']['enable_incomplete_verification'] = True
        config['bab']['timeout'] = self.timeout
        config['solver']['bound_prop_method'] = method
        
        try:
            solver = ABCrownSolver(spec, self.model, config=config, name='verify')
            result = solver.solve()
            
            reference = result.reference
            if 'global_lb' in reference:
                lb = reference['global_lb']
                if hasattr(lb, 'item'):
                    margin = lb.item()
                else:
                    margin = float(lb) if len(lb) > 0 else None
            else:
                margin = None
            
            return {
                'status': result.status,
                'success': result.success,
                'margin': margin,
                'time': result.stats.get('elapsed', 0) if result.stats else 0
            }
            
        except Exception as e:
            print(f"Warning: Verification failed: {e}")
            return {'status': 'error', 'margin': None, 'time': 0, 'error': str(e)}
    
    def verify_batch(self, samples: List[Tuple[torch.Tensor, int]], epsilon: float, method: str = 'alpha-crown') -> pd.DataFrame:
        """批量验证样本"""
        
        results = []
        
        for idx, (image, label) in enumerate(samples):
            start_time = time.time()
            result = self.verify_sample(image, label, epsilon, method=method)
            elapsed = time.time() - start_time
            
            results.append({
                'idx': idx,
                'label': label,
                'status': result['status'],
                'success': result.get('success', False),
                'margin': result['margin'],
                'time': elapsed
            })
            
            if (idx + 1) % 10 == 0:
                verified_count = sum(1 for r in results if r['status'].startswith('safe'))
                print(f"  Processed {idx+1}/{len(samples)}, Verified safe: {verified_count}")
        
        return pd.DataFrame(results)


def load_dataset(dataset_name: str, sample_size: int = 100, start_idx: int = 0):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
    
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
    
    indices = list(range(start_idx, min(start_idx + sample_size, len(test_dataset))))
    samples = [test_dataset[i] for i in indices]
    
    return samples


def run_verification_experiment(model, dataset_name: str, sample_size: int, 
                                  eps_values: List[float], timeout: int = 60) -> pd.DataFrame:
    """使用真正的αβ-CROWN运行验证实验"""
    
    print(f"\n{'='*60}")
    print(f"Running αβ-CROWN verification on {dataset_name}")
    print(f"{'='*60}")
    
    verifier = ABCrownVerifier(model, dataset_name, timeout=timeout)
    
    results = []
    
    for eps in eps_values:
        print(f"\n--- Epsilon = {eps} ---")
        
        samples = load_dataset(dataset_name, sample_size)
        
        start_time = time.time()
        df = verifier.verify_batch(samples, eps)
        elapsed = time.time() - start_time
        
        safe_count = (df['status'] == 'safe-incomplete').sum() + (df['status'] == 'safe').sum()
        unsafe_count = (df['status'] == 'unsafe-pgd').sum()
        unknown_count = (df['status'] == 'unknown').sum()
        
        robust_acc = safe_count / len(df) * 100
        avg_margin = df['margin'].mean() if df['margin'].notna().any() else None
        
        print(f"  Results: Safe={safe_count}, Unsafe={unsafe_count}, Unknown={unknown_count}")
        print(f"  Robust Accuracy: {robust_acc:.2f}%")
        print(f"  Average Time: {elapsed/len(df):.4f}s per sample")
        
        result = {
            'dataset': dataset_name,
            'epsilon': eps,
            'method': 'αβ-CROWN',
            'robust_accuracy': robust_acc,
            'avg_time': elapsed / len(df),
            'avg_margin': avg_margin,
            'total_samples': len(df),
            'safe_count': safe_count,
            'unsafe_count': unsafe_count,
            'unknown_count': unknown_count
        }
        results.append(result)
    
    return pd.DataFrame(results)


def run_ab_crown_vs_standard(model, dataset_name: str, sample_size: int, 
                              eps: float, timeout: int = 60) -> Dict:
    """对比αβ-CROWN和标准验证方法"""
    
    print(f"\n{'='*60}")
    print(f"Comparing αβ-CROWN with different configurations")
    print(f"{'='*60}")
    
    verifier = ABCrownVerifier(model, dataset_name, timeout=timeout)
    
    samples = load_dataset(dataset_name, sample_size)
    
    configs = [
        ('αβ-CROWN (alpha-crown)', 'alpha-crown'),
        ('αβ-CROWN (CROWN)', 'CROWN'),
    ]
    
    results = {}
    
    for name, method in configs:
        print(f"\nTesting {name}...")
        
        df = verifier.verify_batch(samples, eps, method=method)
        
        safe_count = (df['status'] == 'safe-incomplete').sum() + (df['status'] == 'safe').sum()
        robust_acc = safe_count / len(df) * 100
        
        results[name] = {
            'robust_accuracy': robust_acc,
            'avg_time': df['time'].mean(),
            'safe_count': safe_count,
            'unknown_count': (df['status'] == 'unknown').sum()
        }
        
        print(f"  {name}: Robust Acc = {robust_acc:.2f}%, Avg Time = {results[name]['avg_time']:.4f}s")
    
    return results


def plot_results(df: pd.DataFrame, save_path: str = None):
    """绘制验证结果"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        axes[0].plot(dataset_df['epsilon'], dataset_df['robust_accuracy'], 
                    marker='o', label=f'{dataset}')
    
    axes[0].set_xlabel('Epsilon')
    axes[0].set_ylabel('Robust Accuracy (%)')
    axes[0].set_title('Robust Accuracy vs Epsilon (αβ-CROWN)')
    axes[0].legend()
    axes[0].grid(True)
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        axes[1].plot(dataset_df['epsilon'], dataset_df['avg_time'], 
                    marker='s', label=f'{dataset}')
    
    axes[1].set_xlabel('Epsilon')
    axes[1].set_ylabel('Average Time (s)')
    axes[1].set_title('Verification Time vs Epsilon (αβ-CROWN)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.close()


def main():
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./data', exist_ok=True)
    
    print("=" * 60)
    print("Neural Network Robustness Verification with αβ-CROWN")
    print("=" * 60)
    
    eps_values = [0.01, 0.02, 0.03, 0.05, 0.1]
    sample_size = 20
    timeout = 30
    
    all_results = []
    
    for dataset_name in ['MNIST', 'CIFAR10']:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} dataset")
        print(f"{'='*60}")
        
        model_type = 'simple'
        model_path = f'./test_models/{dataset_name}_{model_type}.pt'
        
        if dataset_name == 'MNIST':
            model = get_mnist_model(model_type)
        else:
            model = get_cifar10_model(model_type)
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            print(f"Warning: Model not found at {model_path}, using random weights")
        
        model.eval()
        
        print(f"Loading {sample_size} samples from {dataset_name}...")
        
        df = run_verification_experiment(model, dataset_name, sample_size, eps_values, timeout)
        all_results.append(df)
        
        print(f"\nRunning ablation study on {dataset_name} with eps=0.03...")
        ablation_results = run_ab_crown_vs_standard(model, dataset_name, sample_size, 0.03, timeout)
        
        with open(f'./results/ablation_{dataset_name}.json', 'w') as f:
            json.dump(ablation_results, f, indent=2)
    
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv('./results/abcrown_verification_results.csv', index=False)
    print("\nResults saved to ./results/abcrown_verification_results.csv")
    
    plot_results(all_df, './results/abcrown_verification_plots.png')
    
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(all_df.to_string())
    
    print("\nExperiment completed!")


if __name__ == '__main__':
    main()
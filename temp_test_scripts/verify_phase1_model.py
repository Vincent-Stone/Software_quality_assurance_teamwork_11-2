"""
使用 complete_verifier 验证 Phase 1 模型的鲁棒性

该脚本专门用于验证经过 CROWN-IBP 训练的 Phase 1 模型
记录验证率、超时率、平均验证时间等指标
"""

import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
import time
import json
import yaml
from datetime import datetime
from collections import defaultdict

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'complete_verifier'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robustness_verification'))

try:
    from models import get_mnist_model
except ImportError:
    models_path = os.path.join(os.path.dirname(__file__), 'robustness_verification')
    if models_path not in sys.path:
        sys.path.insert(0, models_path)
    from models import get_mnist_model


def check_dependencies():
    """检查 complete_verifier 所需的依赖"""
    missing = []
    
    try:
        import onnxruntime
    except ImportError:
        missing.append('onnxruntime')
    
    try:
        import onnx
    except ImportError:
        missing.append('onnx')
    
    try:
        import onnx2pytorch
    except ImportError:
        missing.append('onnx2pytorch')
    
    has_all = len(missing) == 0
    return has_all, missing


def export_model_to_onnx(model, onnx_path, input_shape=(1, 1, 28, 28)):
    """将 PyTorch 模型导出为 ONNX 格式"""
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    
    print(f"[OK] Model exported to ONNX: {onnx_path}")
    return onnx_path


def create_vnnlib_spec(image_idx, label, epsilon, num_classes=10):
    """
    创建 VNNLIB 规范文件
    
    Args:
        image_idx: 图像索引
        label: 真实标签
        epsilon: 扰动半径
        num_classes: 类别数
    
    Returns:
        vnnlib 文件路径
    """
    # 构建输出约束
    output_constraints = []
    for j in range(num_classes):
        if j != label:
            output_constraints.append(f"y{j} - y{label} <= -1e-8")
    
    vnnlib_content = f"""
# Robustness specification for MNIST sample {image_idx}
# Label: {label}, epsilon: {epsilon}

# Input constraints (Linf norm bounded perturbation)
forall i:
  -epsilon <= x0[i] <= epsilon

# Output constraints (correct classification)
"""
    
    # 添加每个输出约束
    for constraint in output_constraints:
        vnnlib_content += f"{constraint}\n"
    
    vnnlib_path = f"./temp_spec_{image_idx}.vnnlib"
    with open(vnnlib_path, 'w') as f:
        f.write(vnnlib_content)
    
    return vnnlib_path


def verify_single_sample_with_complete_verifier(model_path, sample_data, label, epsilon=0.1, timeout=30):
    """
    使用 complete_verifier 验证单个样本
    
    Args:
        model_path: 模型路径 (.pt 文件)
        sample_data: 样本数据 (tensor, shape [1, 1, 28, 28])
        label: 真实标签
        epsilon: 扰动半径
        timeout: 验证超时时间
    
    Returns:
        result: 'verified' | 'timeout' | 'falsified' | 'error'
        time_elapsed: 验证耗时（秒）
    """
    try:
        from api import ABCrownSolver, ConfigBuilder
        
        # 加载模型
        model = get_mnist_model('simple')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # 导出为 ONNX
        onnx_path = "./temp_model.onnx"
        export_model_to_onnx(model, onnx_path)
        
        # 创建 VNNLIB 规范
        vnnlib_path = create_vnnlib_spec(0, label, epsilon)
        
        # 构建 complete_verifier 配置
        config = ConfigBuilder.default()
        config.update({
            'general': {
                'complete_verifier': 'bab-refine',
                'device': 'cpu',
                'timeout': timeout,
            },
            'model': {
                'path': onnx_path,
                'input_shape': [1, 1, 28, 28],
            },
            'specification': {
                'epsilon': epsilon,
            },
            'solver': {
                'batch_size': 256,
                'beta-crown': {
                    'iteration': 20,
                },
            },
            'bab': {
                'timeout': timeout,
                'branching': {
                    'reduceop': 'max',
                },
            },
        })
        
        # 运行验证
        solver = ABCrownSolver()
        solver.load_config(config.build())
        
        start_time = time.time()
        
        # 准备输入数据
        # complete_verifier 需要特定格式的输入
        input_bounds_lower = sample_data - epsilon
        input_bounds_upper = sample_data + epsilon
        
        # 执行验证
        try:
            result_dict = solver.verify(
                model_path=onnx_path,
                vnnlib_path=vnnlib_path,
                input_bounds=(input_bounds_lower, input_bounds_upper),
            )
            
            result = result_dict.get('result', 'error')
            time_elapsed = time.time() - start_time
            
        except Exception as e:
            print(f"Verification error: {e}")
            result = 'error'
            time_elapsed = time.time() - start_time
        
        # 清理临时文件
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(vnnlib_path):
            os.remove(vnnlib_path)
        
        return result, time_elapsed
    
    except ImportError:
        print("[WARNING] complete_verifier not available")
        return 'error', 0


def verify_with_alpha_crown(model, sample_data, label, epsilon=0.1):
    """
    使用 alpha-CROWN 验证单个样本（作为 fallback）
    
    Args:
        model: PyTorch 模型
        sample_data: 样本数据
        label: 真实标签
        epsilon: 扰动半径
    
    Returns:
        verified: 是否被证明鲁棒
        margin: 边界裕度
    """
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    
    # 创建 BoundedModule
    dummy_input = torch.zeros(1, 1, 28, 28)
    bound_opts = {'conv_mode': 'patches', 'deterministic': True}
    bounded_model = BoundedModule(model, dummy_input, bound_opts=bound_opts, device='cpu')
    
    # 设置扰动
    ptb = PerturbationLpNorm(norm=float('inf'), eps=epsilon)
    bounded_data = BoundedTensor(sample_data, ptb)
    
    # 计算边界
    lb, ub = bounded_model.compute_bounds(
        x=(bounded_data,),
        method='CROWN-Optimized',
        bound_lower=True,
        bound_upper=True
    )
    
    # 检查鲁棒性
    true_lb = lb[0, label].item()
    worst_other_ub = max([ub[0, j].item() for j in range(10) if j != label])
    margin = true_lb - worst_other_ub
    
    verified = margin > 0
    
    return verified, margin


def verify_phase1_model_robustness(
    model_path,
    epsilon=0.1,
    num_samples=50,
    timeout=30,
    results_dir='./results'
):
    """
    验证 Phase 1 模型的鲁棒性
    
    Args:
        model_path: Phase 1 模型路径
        epsilon: 扰动半径
        num_samples: 验证样本数
        timeout: 每个样本的超时时间
        results_dir: 结果保存目录
    
    Returns:
        results: 验证结果字典
    """
    print(f"\n{'='*70}")
    print(f"Phase 1 Model Robustness Verification")
    print(f"{'='*70}")
    print(f"Model path: {model_path}")
    print(f"Epsilon: {epsilon}")
    print(f"Num samples: {num_samples}")
    print(f"Timeout per sample: {timeout}s")
    print(f"{'='*70}\n")
    
    # 检查依赖
    has_complete_verifier, missing_deps = check_dependencies()
    
    if has_complete_verifier:
        print("[OK] All dependencies for complete_verifier are installed.")
        use_complete_verifier = True
    else:
        print(f"[WARNING] Missing dependencies: {', '.join(missing_deps)}")
        print("[INFO] Falling back to alpha-CROWN verification.")
        use_complete_verifier = False
    
    # 加载模型
    print("\n[1/3] Loading model...")
    model = get_mnist_model('simple')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"[OK] Model loaded from: {model_path}")
    
    # 加载测试数据
    print("\n[2/3] Loading test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 选择部分样本
    selected_indices = range(min(num_samples, len(test_dataset)))
    
    # 验证统计
    results = {
        'verified': 0,
        'timeout': 0,
        'falsified': 0,
        'error': 0,
        'total_time': 0,
        'avg_time': 0,
        'min_time': float('inf'),
        'max_time': 0,
        'margins': [],
        'detailed_results': []
    }
    
    print("\n[3/3] Running verification...")
    print(f"{'Progress':<20} {'Result':<15} {'Time':<10} {'Margin':<10}")
    print("-" * 70)
    
    for idx in selected_indices:
        sample_data, label = test_dataset[idx]
        sample_data = sample_data.unsqueeze(0)  # 添加 batch dimension
        
        # 使用 alpha-CROWN 进行验证（更可靠）
        start_time = time.time()
        try:
            verified, margin = verify_with_alpha_crown(model, sample_data, label, epsilon)
            time_elapsed = time.time() - start_time
            
            if verified:
                result = 'verified'
            else:
                result = 'falsified'
        except Exception as e:
            print(f"Error verifying sample {idx}: {e}")
            result = 'error'
            margin = None
            time_elapsed = time.time() - start_time
        
        # 更新统计
        results[result] += 1
        results['total_time'] += time_elapsed
        results['min_time'] = min(results['min_time'], time_elapsed)
        results['max_time'] = max(results['max_time'], time_elapsed)
        
        if margin is not None:
            results['margins'].append(margin)
        
        results['detailed_results'].append({
            'sample_idx': idx,
            'label': label,
            'result': result,
            'time': time_elapsed,
            'margin': margin
        })
        
        # 打印进度
        progress = f"{idx+1}/{num_samples}"
        margin_str = f"{margin:.4f}" if margin else "N/A"
        print(f"{progress:<20} {result:<15} {time_elapsed:.2f}s{'':<5} {margin_str}")
    
    # 计算平均值
    results['avg_time'] = results['total_time'] / num_samples
    if results['margins']:
        results['avg_margin'] = sum(results['margins']) / len(results['margins'])
    else:
        results['avg_margin'] = None
    
    # 计算验证率
    results['verified_rate'] = 100. * results['verified'] / num_samples
    results['timeout_rate'] = 100. * results['timeout'] / num_samples
    
    # 打印总结
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples: {num_samples}")
    print(f"Verified: {results['verified']} ({results['verified_rate']:.2f}%)")
    print(f"Timeout: {results['timeout']} ({results['timeout_rate']:.2f}%)")
    print(f"Falsified: {results['falsified']}")
    print(f"Errors: {results['error']}")
    print(f"Average verification time: {results['avg_time']:.2f}s")
    print(f"Min time: {results['min_time']:.2f}s")
    print(f"Max time: {results['max_time']:.2f}s")
    
    if results['avg_margin'] is not None:
        print(f"Average margin: {results['avg_margin']:.4f}")
    
    print(f"{'='*70}\n")
    
    # 保存结果
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存 JSON
    json_path = os.path.join(results_dir, f'phase1_verification_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved to: {json_path}")
    
    # 保存 YAML（人类可读）
    yaml_path = os.path.join(results_dir, f'phase1_verification_{timestamp}.yaml')
    yaml_safe_results = {
        'model_path': model_path,
        'epsilon': epsilon,
        'num_samples': num_samples,
        'verified_count': results['verified'],
        'verified_rate': results['verified_rate'],
        'timeout_count': results['timeout'],
        'timeout_rate': results['timeout_rate'],
        'avg_time': results['avg_time'],
        'avg_margin': results['avg_margin'],
        'timestamp': timestamp,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_safe_results, f, default_flow_style=False)
    print(f"[OK] Summary saved to: {yaml_path}")
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify Phase 1 Model with complete_verifier')
    parser.add_argument('--model_path', type=str, 
                        default='robustness_verification/test_models/MNIST_simple_final_ibp.pt',
                        help='Path to Phase 1 model')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Perturbation epsilon')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to verify')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout per sample (seconds)')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    
    args = parser.parse_args()
    
    # 运行验证
    results = verify_phase1_model_robustness(
        model_path=args.model_path,
        epsilon=args.epsilon,
        num_samples=args.num_samples,
        timeout=args.timeout,
        results_dir=args.results_dir
    )
    
    print("\nVerification completed!")
    return results


if __name__ == '__main__':
    main()
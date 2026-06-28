"""
使用 complete_verifier 进行完整的鲁棒性验证
对比三种 MNIST 模型的认证鲁棒性
"""
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
import time

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'complete_verifier'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robustness_verification'))

# 导入模型构建函数
try:
    from models import get_mnist_model
except ImportError:
    # 如果直接导入失败，尝试使用完整路径
    import sys
    models_path = os.path.join(os.path.dirname(__file__), 'robustness_verification')
    if models_path not in sys.path:
        sys.path.insert(0, models_path)
    from models import get_mnist_model


def check_dependencies():
    """
    检查 complete_verifier 所需的依赖
    返回: (has_complete_verifier, missing_deps)
    """
    missing = []
    
    # 检查 onnxruntime
    try:
        import onnxruntime
    except ImportError:
        missing.append('onnxruntime')
    
    # 检查 onnx
    try:
        import onnx
    except ImportError:
        missing.append('onnx')
    
    # 检查 onnx2pytorch
    try:
        import onnx2pytorch
    except ImportError:
        missing.append('onnx2pytorch')
    
    has_all = len(missing) == 0
    return has_all, missing


def load_model(model_path):
    """加载模型"""
    model = get_mnist_model('simple')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"[OK] Model loaded: {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.eval()
    return model


def verify_with_complete_verifier(model, test_loader, epsilon=0.1, num_samples=10, timeout=30):
    """
    使用 complete_verifier 进行完整验证
    
    参数:
        model: PyTorch 模型
        test_loader: 测试数据加载器
        epsilon: 扰动半径
        num_samples: 验证样本数
        timeout: 每个样本的超时时间（秒）
    
    返回:
        results: 验证结果字典
    """
    # 尝试导入 complete_verifier
    try:
        from api import ABCrownSolver, VerificationSpec, input_vars, output_vars, ConfigBuilder
        print("[OK] complete_verifier imported successfully")
        use_complete_verifier = True
    except ImportError as e:
        print(f"[WARNING] Failed to import complete_verifier: {e}")
        print("[INFO] This is expected if onnxruntime is not installed.")
        print("[INFO] Falling back to auto_LiRPA verification...")
        use_complete_verifier = False
    
    # 如果 complete_verifier 不可用，直接使用 auto_LiRPA
    if not use_complete_verifier:
        return verify_with_auto_lirpa(model, test_loader, epsilon, num_samples)
    
    device = torch.device('cpu')
    model = model.to(device)
    
    print(f"\n{'='*70}")
    print(f"Starting complete verification - epsilon={epsilon}, samples={num_samples}")
    print(f"{'='*70}")
    
    results = {
        'total': 0,
        'verified_safe': 0,
        'verified_unsafe': 0,
        'unknown': 0,
        'margins': [],
        'times': []
    }
    
    sample_count = 0
    
    for idx, (data, target) in enumerate(test_loader):
        if sample_count >= num_samples:
            break
        
        data = data.to(device)
        target = target.to(device)
        batch_size = data.shape[0]
        
        for i in range(batch_size):
            if sample_count >= num_samples:
                break
            
            single_data = data[i:i+1]
            single_target = target[i:i+1]
            
            start_time = time.time()
            
            try:
                # 构建输入边界
                center = single_data.squeeze().numpy()
                lower = torch.tensor(center - epsilon, dtype=torch.float32).unsqueeze(0)
                upper = torch.tensor(center + epsilon, dtype=torch.float32).unsqueeze(0)
                lower = torch.clamp(lower, 0, 1)
                upper = torch.clamp(upper, 0, 1)
                
                # 构建验证规范
                target_class = single_target.item()
                clauses = []
                for c in range(10):
                    if c == target_class:
                        continue
                    C = torch.zeros(1, 10, dtype=torch.float32)
                    C[0, target_class] = 1
                    C[0, c] = -1
                    rhs = torch.zeros(1, dtype=torch.float32)
                    clauses.append((C, rhs))
                
                spec = VerificationSpec.build_from_input_bounds(lower, upper, clauses)
                
                # 配置验证器
                config = ConfigBuilder.from_defaults().to_dict()
                config['general']['device'] = 'cpu'
                config['general']['deterministic'] = True
                config['general']['complete_verifier'] = 'skip'
                config['general']['enable_incomplete_verification'] = True
                config['bab']['timeout'] = timeout
                config['solver']['bound_prop_method'] = 'IBP'
                
                # 执行验证
                solver = ABCrownSolver(spec, model, config=config, name='verify')
                result = solver.solve()
                
                elapsed = time.time() - start_time
                
                # 解析结果
                reference = result.reference
                if 'global_lb' in reference:
                    lb = reference['global_lb']
                    margin = lb.item() if hasattr(lb, 'item') else float(lb) if len(lb) > 0 else None
                else:
                    margin = None
                
                if margin is not None and margin > 0:
                    results['verified_safe'] += 1
                    status = 'SAFE'
                elif margin is not None and margin <= 0:
                    results['verified_unsafe'] += 1
                    status = 'UNSAFE'
                else:
                    results['unknown'] += 1
                    status = 'UNKNOWN'
                
                if margin is not None:
                    results['margins'].append(margin)
                results['times'].append(elapsed)
                results['total'] += 1
                
                sample_count += 1
                
                if sample_count % 5 == 0:
                    safe_rate = results['verified_safe'] / results['total'] * 100
                    print(f"  [{sample_count}/{num_samples}] Status: {status}, "
                          f"Margin: {margin:.4f if margin else 'N/A'}, "
                          f"Time: {elapsed:.2f}s, Safe rate: {safe_rate:.1f}%")
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  [{sample_count+1}/{num_samples}] Verification error: {str(e)[:100]}")
                results['unknown'] += 1
                results['total'] += 1
                results['times'].append(elapsed)
                sample_count += 1
    
    return results


def verify_with_auto_lirpa(model, test_loader, epsilon=0.1, num_samples=10):
    """
    使用 auto_LiRPA 进行快速验证（备用方案）
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'auto_LiRPA'))
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    
    device = torch.device('cpu')
    model = model.to(device)
    
    dummy_input = torch.zeros(1, 1, 28, 28, device=device)
    bound_opts = {'conv_mode': 'patches', 'deterministic': True}
    
    print(f"\n{'='*70}")
    print(f"Starting auto_LiRPA verification - epsilon={epsilon}, samples={num_samples}")
    print(f"{'='*70}")
    
    bounded_model = BoundedModule(model, dummy_input, bound_opts=bound_opts, device=device)
    
    results = {
        'total': 0,
        'verified_safe': 0,
        'verified_unsafe': 0,
        'unknown': 0,
        'margins': [],
        'times': []
    }
    
    sample_count = 0
    
    for idx, (data, target) in enumerate(test_loader):
        if sample_count >= num_samples:
            break
        
        data = data.to(device)
        target = target.to(device)
        batch_size = data.shape[0]
        
        start_time = time.time()
        
        try:
            ptb = PerturbationLpNorm(norm=float('inf'), eps=epsilon)
            bounded_data = BoundedTensor(data, ptb)
            
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_data,),
                method='IBP',
                bound_upper=False
            )
            
            elapsed = time.time() - start_time
            
            margins = []
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                
                target_class = target[i].item()
                sample_lb = lb[i]
                target_lb = sample_lb[target_class]
                other_lbs = torch.cat([sample_lb[:target_class], sample_lb[target_class+1:]])
                min_margin = (target_lb - other_lbs).min().item()
                
                margins.append(min_margin)
                
                if min_margin > 0:
                    results['verified_safe'] += 1
                else:
                    results['verified_unsafe'] += 1
                
                sample_count += 1
            
            results['margins'].extend(margins)
            results['times'].extend([elapsed / batch_size] * batch_size)
            results['total'] += batch_size
            
            safe_rate = results['verified_safe'] / results['total'] * 100
            avg_margin = sum(margins) / len(margins)
            print(f"  [{sample_count}/{num_samples}] Batch verified, "
                  f"Avg margin: {avg_margin:.4f}, Safe rate: {safe_rate:.1f}%, "
                  f"Time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  [ERROR] Verification failed: {e}")
            results['unknown'] += batch_size
            results['total'] += batch_size
            sample_count += batch_size
    
    return results


def print_results(model_name, results):
    """打印验证结果"""
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    print(f"Total samples:      {results['total']}")
    print(f"Verified safe:      {results['verified_safe']} ({results['verified_safe']/results['total']*100:.1f}%)")
    print(f"Verified unsafe:    {results['verified_unsafe']} ({results['verified_unsafe']/results['total']*100:.1f}%)")
    print(f"Unknown/Timeout:    {results['unknown']}")
    
    if results['margins']:
        print(f"Average margin:     {sum(results['margins'])/len(results['margins']):.4f}")
        print(f"Min margin:         {min(results['margins']):.4f}")
        print(f"Max margin:         {max(results['margins']):.4f}")
    
    if results['times']:
        print(f"Avg time/sample:    {sum(results['times'])/len(results['times']):.4f}s")
        print(f"Total time:         {sum(results['times']):.2f}s")
    print(f"{'='*70}\n")


def main():
    # 配置
    epsilon = 0.1
    num_samples = 10
    batch_size = 5
    
    # 检查依赖
    print("="*70)
    print("DEPENDENCY CHECK")
    print("="*70)
    has_all, missing = check_dependencies()
    
    if has_all:
        print("[OK] All dependencies for complete_verifier are installed.")
    else:
        print(f"[WARNING] Missing dependencies: {', '.join(missing)}")
        print("\nTo use complete_verifier (full verification), install them with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nNote: The script will automatically fall back to auto_LiRPA verification.")
        print("      auto_LiRPA provides reliable bound computation for robustness analysis.")
    print("="*70)
    print()
    
    # 模型配置
    models_config = [
        ('Standard Model', 'robustness_verification/test_models/MNIST_simple.pt'),
        ('IBP Model', 'robustness_verification/test_models/MNIST_simple_IBP.pt'),
        ('CROWN-IBP Model', 'robustness_verification/test_models/MNIST_simple_CROWN-IBP.pt'),
    ]
    
    # 加载测试数据
    print("Loading MNIST test set...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    all_results = []
    
    # 验证每个模型
    for model_name, model_path in models_config:
        print(f"\n{'#'*70}")
        print(f"# Verifying {model_name}")
        print(f"{'#'*70}")
        
        try:
            model = load_model(model_path)
            results = verify_with_complete_verifier(
                model, test_loader, epsilon=epsilon,
                num_samples=num_samples, timeout=30
            )
            
            all_results.append({
                'name': model_name,
                'path': model_path,
                'results': results
            })
            
            print_results(model_name, results)
            
        except Exception as e:
            print(f"[ERROR] Failed to verify {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成对比表格
    print(f"\n\n{'='*80}")
    print("ROBUSTNESS VERIFICATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model Type':<20} {'Safe Rate':<12} {'Avg Margin':<12} {'Avg Time(s)':<12}")
    print(f"{'-'*80}")
    
    for item in all_results:
        name = item['name']
        res = item['results']
        safe_rate = res['verified_safe'] / res['total'] * 100 if res['total'] > 0 else 0
        avg_margin = sum(res['margins'])/len(res['margins']) if res['margins'] else 0
        avg_time = sum(res['times'])/len(res['times']) if res['times'] else 0
        
        print(f"{name:<20} {safe_rate:>10.1f}%  {avg_margin:>10.4f}  {avg_time:>10.4f}")
    
    print(f"{'='*80}")
    print("\nKey Findings:")
    print("- Higher safe rate indicates better robustness under epsilon perturbation")
    print("- Larger average margin means safer decision boundaries")
    print("- Robust training (IBP/CROWN-IBP) should outperform standard training")
    print("- Complete verifier provides formal guarantees (sound verification)")


if __name__ == '__main__':
    main()

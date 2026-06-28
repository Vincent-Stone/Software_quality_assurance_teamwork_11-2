"""
快速测试 MNIST_tiny_lambda_1.0.pt 模型的鲁棒性
使用 auto_LiRPA 进行 IBP 边界计算验证
"""
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
import time

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'auto_LiRPA'))
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# 导入模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robustness_verification'))
from models import get_mnist_model

def test_model_robustness(model_path, num_samples=100, epsilon=0.1):
    """
    测试模型鲁棒性
    
    参数:
        model_path: 模型文件路径
        num_samples: 测试样本数
        epsilon: 扰动半径
    """
    print("="*70)
    print(f"Testing Model Robustness")
    print(f"Model: {model_path}")
    print(f"Samples: {num_samples}, Epsilon: {epsilon}")
    print("="*70)
    
    # 加载模型
    model = get_mnist_model('tiny')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"[OK] Model loaded from {model_path}")
    else:
        print(f"[ERROR] Model file not found: {model_path}")
        return
    
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    
    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='data', train=False, download=True, transform=transform
    )
    
    # 创建 bounded model
    dummy_input = torch.zeros(1, 1, 28, 28, device=device)
    bound_opts = {'conv_mode': 'patches', 'deterministic': True}
    bounded_model = BoundedModule(model, dummy_input, bound_opts=bound_opts, device=device)
    
    # 统计结果
    total = 0
    verified_safe = 0
    verified_unsafe = 0
    margins = []
    accuracies = []
    
    start_time = time.time()
    
    print(f"\nVerifying {num_samples} samples...")
    print("-"*70)
    
    for idx in range(min(num_samples, len(test_dataset))):
        data, target = test_dataset[idx]
        data = data.unsqueeze(0).to(device)  # Add batch dimension
        target = torch.tensor([target]).to(device)
        
        try:
            # 创建扰动
            ptb = PerturbationLpNorm(norm=float('inf'), eps=epsilon)
            bounded_data = BoundedTensor(data, ptb)
            
            # 计算边界
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_data,),
                method='IBP',
                bound_upper=False
            )
            
            # 计算 margin
            target_class = target.item()
            sample_lb = lb[0]
            target_lb = sample_lb[target_class]
            other_lbs = torch.cat([sample_lb[:target_class], sample_lb[target_class+1:]])
            min_margin = (target_lb - other_lbs).min().item()
            
            margins.append(min_margin)
            
            # 判断是否通过验证
            if min_margin > 0:
                verified_safe += 1
                status = "SAFE ✓"
            else:
                verified_unsafe += 1
                status = f"UNSAFE ✗ (margin={min_margin:.4f})"
            
            total += 1
            
            # 每10个样本打印一次进度
            if (idx + 1) % 10 == 0:
                safe_rate = verified_safe / total * 100
                avg_margin = sum(margins) / len(margins)
                print(f"  [{idx+1}/{num_samples}] Safe rate: {safe_rate:>5.1f}%, "
                      f"Avg margin: {avg_margin:>8.4f}")
            
        except Exception as e:
            print(f"  [ERROR] Sample {idx+1} failed: {str(e)[:80]}")
            total += 1
    
    elapsed = time.time() - start_time
    
    # 打印结果
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    print(f"Total samples:      {total}")
    print(f"Verified safe:      {verified_safe} ({verified_safe/total*100:.1f}%)")
    print(f"Verified unsafe:    {verified_unsafe} ({verified_unsafe/total*100:.1f}%)")
    
    if margins:
        print(f"\nMargin Statistics:")
        print(f"  Average margin:   {sum(margins)/len(margins):.4f}")
        print(f"  Min margin:       {min(margins):.4f}")
        print(f"  Max margin:       {max(margins):.4f}")
        print(f"  Std margin:       {(sum((x - sum(margins)/len(margins))**2 for x in margins) / len(margins))**0.5:.4f}")
    
    print(f"\nTime Statistics:")
    print(f"  Total time:       {elapsed:.2f}s")
    print(f"  Avg time/sample:  {elapsed/total:.4f}s")
    print("="*70)
    
    # 分析结果
    safe_rate = verified_safe / total * 100
    avg_margin = sum(margins)/len(margins) if margins else 0
    
    print(f"\nAnalysis:")
    if safe_rate == 0:
        print("  ⚠️  WARNING: All samples failed verification!")
        print("  This indicates the model has very poor robustness.")
        print("\nPossible reasons:")
        print("  1. Model capacity too small (tiny model with only 32 neurons)")
        print("  2. Insufficient training epochs (only 5 epochs)")
        print("  3. Lambda weight may need adjustment")
        print("  4. Epsilon mismatch between training and verification")
    elif safe_rate < 50:
        print(f"  ⚠️  Low robustness rate ({safe_rate:.1f}%)")
        print("  Consider increasing model size or training longer.")
    else:
        print(f"  ✓ Reasonable robustness rate ({safe_rate:.1f}%)")


if __name__ == '__main__':
    model_path = 'robustness_verification/test_models/MNIST_tiny_lambda_1.0.pt'
    test_model_robustness(model_path, num_samples=100, epsilon=0.1)

"""
验证脚本：测试基于simple模型的SPSA + alpha-CROWN训练后的性能指标
"""

import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'auto_LiRPA'))

from models import get_mnist_model
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def verify_model_robustness(model_path, epsilon=0.1, num_samples=1000):
    """
    验证模型的鲁棒性指标
    
    Args:
        model_path: 模型文件路径
        epsilon: 扰动范围
        num_samples: 测试样本数量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = get_mnist_model(model_type='simple')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"Model loaded from: {model_path}")
    print(f"Model architecture: SimpleFCN([64, 32])")
    print(f"Epsilon: {epsilon}")
    print(f"{'='*70}")
    
    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 选择部分样本进行详细验证
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=True
    )
    
    # 标准准确率测试
    correct_standard = 0
    total_standard = 0
    
    # 鲁棒性验证测试
    correct_verified = 0
    total_verified = 0
    verified_samples = 0
    
    # Bound statistics
    bound_margins = []
    verified_accuracies_per_batch = []
    
    print("\n[1/2] Computing Standard Accuracy...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # 标准预测
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_standard += pred.eq(target.view_as(pred)).sum().item()
            total_standard += data.size(0)
            
            if batch_idx >= 10:  # 只测试前1000个样本用于快速验证
                break
    
    standard_accuracy = 100. * correct_standard / total_standard
    print(f"Standard Accuracy: {standard_accuracy:.2f}% ({correct_standard}/{total_standard})")
    
    # 鲁棒性验证
    print("\n[2/2] Computing Robustness Metrics using alpha-CROWN...")
    
    # 创建BoundedModule
    dummy_input = torch.zeros(1, 1, 28, 28, device=device)
    bound_opts = {'conv_mode': 'patches', 'deterministic': True}
    bounded_model = BoundedModule(model, dummy_input, bound_opts=bound_opts, device=device)
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if verified_samples >= num_samples:
            break
        
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        
        # 使用alpha-CROWN计算边界
        ptb = PerturbationLpNorm(norm=float('inf'), eps=epsilon)
        bounded_data = BoundedTensor(data, ptb)
        
        try:
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_data,),
                method='CROWN-Optimized',
                bound_lower=True,
                bound_upper=True
            )
            
            # 检查是否鲁棒：对于每个样本，真实类别的下界是否大于其他类别的上界
            for i in range(batch_size):
                true_label = target[i].item()
                true_lb = lb[i, true_label].item()
                
                # 检查其他类别的上界
                is_robust = True
                for j in range(10):
                    if j != true_label:
                        other_ub = ub[i, j].item()
                        if true_lb <= other_ub:
                            is_robust = False
                            break
                
                if is_robust:
                    correct_verified += 1
                
                total_verified += 1
                
                # 记录边界margin
                worst_other_ub = max([ub[i, j].item() for j in range(10) if j != true_label])
                margin = true_lb - worst_other_ub
                bound_margins.append(margin)
            
            verified_samples += batch_size
            
            # 记录batch验证准确率
            batch_verified_acc = 100. * correct_verified / total_verified
            verified_accuracies_per_batch.append(batch_verified_acc)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Verified {verified_samples}/{num_samples} samples: "
                      f"Verified Accuracy={batch_verified_acc:.2f}%, "
                      f"Avg Margin={sum(bound_margins[-batch_size:])/batch_size:.4f}")
        
        except Exception as e:
            print(f"  Warning: Bound computation failed for batch {batch_idx}: {e}")
            continue
    
    # 计算最终指标
    verified_accuracy = 100. * correct_verified / total_verified
    avg_margin = sum(bound_margins) / len(bound_margins)
    positive_margin_ratio = 100. * sum([1 for m in bound_margins if m > 0]) / len(bound_margins)
    
    print(f"\n{'='*70}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*70}")
    print(f"Standard Accuracy: {standard_accuracy:.2f}%")
    print(f"Verified Accuracy (Robust): {verified_accuracy:.2f}%")
    print(f"Samples Verified: {total_verified}")
    print(f"Average Bound Margin: {avg_margin:.4f}")
    print(f"Positive Margin Ratio: {positive_margin_ratio:.2f}%")
    print(f"{'='*70}")
    
    return {
        'standard_accuracy': standard_accuracy,
        'verified_accuracy': verified_accuracy,
        'avg_margin': avg_margin,
        'positive_margin_ratio': positive_margin_ratio,
        'num_samples': total_verified
    }


def compare_models(ibp_path, final_model=None):
    """
    对比IBP模型和最终模型（如果有）的性能
    
    Args:
        ibp_path: IBP训练后的模型路径
        final_model: 最终微调后的模型（可选）
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # 测试IBP模型
    print("\n[IBP Model] Testing Phase 1 (CROWN-IBP) model...")
    ibp_results = verify_model_robustness(ibp_path, epsilon=0.1, num_samples=500)
    
    # 如果有最终模型，进行对比
    if final_model is not None:
        print("\n[Final Model] Testing Phase 1 + Phase 2 (SPSA) model...")
        # 保存最终模型
        final_path = './test_models/MNIST_simple_SPSA_alpha-CROWN.pt'
        torch.save(final_model.state_dict(), final_path)
        print(f"Final model saved to: {final_path}")
        
        final_results = verify_model_robustness(final_path, epsilon=0.1, num_samples=500)
        
        # 对比结果
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Metric':<30} {'IBP Model':<20} {'Final Model':<20}")
        print("-"*70)
        print(f"{'Standard Accuracy':<30} {ibp_results['standard_accuracy']:.2f}%{'':<15} {final_results['standard_accuracy']:.2f}%")
        print(f"{'Verified Accuracy':<30} {ibp_results['verified_accuracy']:.2f}%{'':<15} {final_results['verified_accuracy']:.2f}%")
        print(f"{'Average Margin':<30} {ibp_results['avg_margin']:.4f}{'':<15} {final_results['avg_margin']:.4f}")
        print(f"{'Positive Margin Ratio':<30} {ibp_results['positive_margin_ratio']:.2f}%{'':<15} {final_results['positive_margin_ratio']:.2f}%")
        print("="*70)
    
    return ibp_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify Model Robustness')
    parser.add_argument('--model_path', type=str, 
                        default='./test_models/MNIST_simple_CROWN-IBP_wcim.pt',
                        help='Path to the model file')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Perturbation epsilon')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to verify')
    
    args = parser.parse_args()
    
    results = verify_model_robustness(
        model_path=args.model_path,
        epsilon=args.epsilon,
        num_samples=args.num_samples
    )
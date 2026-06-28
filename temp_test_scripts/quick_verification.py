"""
快速鲁棒性验证脚本 - 使用 auto_LiRPA 进行边界计算验证
对比三种模型的鲁棒性
"""
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
import time

# 添加 auto_LiRPA 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'auto_LiRPA'))
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# 添加 robustness_verification 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robustness_verification'))
from models import get_mnist_model


def load_model(model_path):
    """加载模型"""
    model = get_mnist_model('simple')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"[OK] 模型加载成功: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model.eval()
    return model


def verify_robustness(model, test_loader, epsilon=0.1, num_samples=10, method='IBP'):
    """
    使用 auto_LiRPA 验证模型鲁棒性
    
    参数:
        model: PyTorch 模型
        test_loader: 测试数据加载器
        epsilon: 扰动半径
        num_samples: 验证样本数
        method: 边界计算方法 (IBP/CROWN)
    
    返回:
        results: 验证结果字典
    """
    device = torch.device('cpu')
    model = model.to(device)
    
    # 创建 dummy input 用于构建 BoundedModule
    dummy_input = torch.zeros(1, 1, 28, 28, device=device)
    bound_opts = {
        'conv_mode': 'patches',
        'deterministic': True,
    }
    
    print(f"\n{'='*60}")
    print(f"开始验证 - 方法: {method}, ε={epsilon}, 样本数: {num_samples}")
    print(f"{'='*60}")
    
    bounded_model = BoundedModule(model, dummy_input, bound_opts=bound_opts, device=device)
    
    results = {
        'total': 0,
        'verified_safe': 0,
        'verified_unsafe': 0,
        'unknown': 0,
        'margins': [],
        'times': []
    }
    
    for idx, (data, target) in enumerate(test_loader):
        if idx >= num_samples:
            break
        
        data = data.to(device)
        target = target.to(device)
        
        start_time = time.time()
        
        try:
            batch_size = data.shape[0]
            num_classes = 10
            
            # 创建扰动
            ptb = PerturbationLpNorm(norm=float('inf'), eps=epsilon)
            bounded_data = BoundedTensor(data, ptb)
            
            # 为每个样本计算输出边界
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_data,),
                method=method,
                bound_upper=False
            )
            
            # lb, ub shape: (batch_size, num_classes)
            # 检查每个样本的正确类别是否是输出下界最大的类别
            margins = []
            
            for i in range(batch_size):
                target_class = target[i].item()
                # 获取该样本的下界
                sample_lb = lb[i]  # shape: (num_classes,)
                
                # 计算正确类别的下界与其他类别下界的差值
                target_lb = sample_lb[target_class]
                other_lbs = torch.cat([sample_lb[:target_class], sample_lb[target_class+1:]])
                min_margin = (target_lb - other_lbs).min().item()
                
                margins.append(min_margin)
                
                if min_margin > 0:
                    results['verified_safe'] += 1
                else:
                    results['verified_unsafe'] += 1
            
            elapsed = time.time() - start_time
            results['total'] += batch_size
            results['margins'].extend(margins)
            results['times'].append(elapsed)
            
            # 打印进度
            avg_margin = sum(margins) / len(margins)
            safe_count = results['verified_safe']
            print(f"  [{idx+1}/{num_samples}] Batch size: {batch_size}, "
                  f"安全: {safe_count}, "
                  f"平均边界: {avg_margin:.4f}, "
                  f"耗时: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  [{idx+1}/{num_samples}] 验证失败: {e}")
            results['unknown'] += batch_size
            results['total'] += batch_size
    
    return results


def print_results(model_name, results):
    """打印验证结果"""
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")
    print(f"总样本数:     {results['total']}")
    print(f"验证安全:     {results['verified_safe']} ({results['verified_safe']/results['total']*100:.1f}%)")
    print(f"验证不安全:   {results['verified_unsafe']} ({results['verified_unsafe']/results['total']*100:.1f}%)")
    print(f"未知/超时:    {results['unknown']}")
    
    if results['margins']:
        print(f"平均边界值:   {sum(results['margins'])/len(results['margins']):.4f}")
        print(f"最小边界值:   {min(results['margins']):.4f}")
        print(f"最大边界值:   {max(results['margins']):.4f}")
    
    if results['times']:
        print(f"平均验证时间: {sum(results['times'])/len(results['times']):.4f}s/样本")
        print(f"总验证时间:   {sum(results['times']):.2f}s")
    print(f"{'='*60}\n")


def main():
    # 配置
    epsilon = 0.1
    num_samples = 10
    batch_size = 5  # 小批量以加快验证
    
    # 定义三个模型路径
    models_config = [
        ('标准模型', 'robustness_verification/test_models/MNIST_simple.pt'),
        ('IBP 模型', 'robustness_verification/test_models/MNIST_simple_IBP.pt'),
        ('CROWN-IBP 模型', 'robustness_verification/test_models/MNIST_simple_CROWN-IBP.pt'),
    ]
    
    # 加载测试数据
    print("加载 MNIST 测试集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # 存储所有结果用于对比
    all_results = []
    
    # 对每个模型进行验证
    for model_name, model_path in models_config:
        print(f"\n{'#'*60}")
        print(f"# 验证 {model_name}")
        print(f"{'#'*60}")
        
        try:
            # 加载模型
            model = load_model(model_path)
            
            # 使用 IBP 方法快速验证
            results_ibp = verify_robustness(
                model, test_loader, epsilon=epsilon, 
                num_samples=num_samples, method='IBP'
            )
            
            all_results.append({
                'name': model_name,
                'path': model_path,
                'results': results_ibp
            })
            
            # 打印结果
            print_results(model_name, results_ibp)
            
        except Exception as e:
            print(f"[ERROR] 验证 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成对比表格
    print(f"\n\n{'='*80}")
    print("模型鲁棒性对比总结")
    print(f"{'='*80}")
    print(f"{'模型类型':<15} {'验证安全率':<12} {'平均边界':<12} {'平均时间(s)':<12}")
    print(f"{'-'*80}")
    
    for item in all_results:
        name = item['name']
        res = item['results']
        safe_rate = res['verified_safe'] / res['total'] * 100 if res['total'] > 0 else 0
        avg_margin = sum(res['margins'])/len(res['margins']) if res['margins'] else 0
        avg_time = sum(res['times'])/len(res['times']) if res['times'] else 0
        
        print(f"{name:<15} {safe_rate:>10.1f}%  {avg_margin:>10.4f}  {avg_time:>10.4f}")
    
    print(f"{'='*80}")
    print("\n分析结论:")
    print("- 验证安全率越高，说明模型在该扰动半径下越鲁棒")
    print("- 平均边界值越大，说明模型的决策边界越安全")
    print("- 鲁棒训练（IBP/CROWN-IBP）的模型应该比标准模型有更高的安全率和更大的边界")


if __name__ == '__main__':
    main()

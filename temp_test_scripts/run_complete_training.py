"""
完整模型训练流程脚本

确保训练过程中Phase 2能够顺利完成且不出现中断或错误
训练完成后，将最终模型保存至指定路径

监控指标：
- 损失值变化
- 准确率表现
- 资源使用情况
"""

import sys
import os
import time
import torch
from datetime import datetime

# 添加必要的路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'auto_LiRPA'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robustness_verification'))

def monitor_resources():
    """监控系统资源使用情况"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        return {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_info.rss / (1024 ** 2),
            'memory_vms_mb': memory_info.vms / (1024 ** 2)
        }
    except ImportError:
        return {'cpu_percent': 'N/A', 'memory_mb': 'N/A', 'memory_vms_mb': 'N/A'}


def main():
    print(f"\n{'='*70}")
    print(f"完整模型训练流程启动")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # 目标保存路径
    target_save_path = os.path.join(
        os.path.dirname(__file__),
        'robustness_verification',
        'models',
        'MNIST_simple_final_ibp.pt'
    )
    
    print(f"目标保存路径: {target_save_path}")
    print(f"当前目录: {os.getcwd()}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(target_save_path), exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 打印资源信息
    resources = monitor_resources()
    print(f"\n初始资源状态:")
    print(f"  CPU: {resources['cpu_percent']}%")
    print(f"  内存: {resources['memory_mb']:.2f} MB")
    
    try:
        # 导入训练函数
        from spsa_training import train_with_spsa
        
        # 执行完整训练流程
        result = train_with_spsa(
            dataset='MNIST',
            model_type='simple',
            ibp_epochs=3,
            spsa_epochs=1,
            epsilon=0.1,
            lambda_weight=1.0,
            spsa_a=0.001,
            spsa_c=0.001,
            spsa_momentum=0.0,
            spsa_weight_decay=1e-4,
            spsa_grad_clip=1.0,
            spsa_param_clip=0.01,
            spsa_smoothing=0.9,
            batch_size=32,
            save_path=target_save_path,
            log_dir='./logs'
        )
        
        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        
        # 打印最终资源状态
        resources = monitor_resources()
        print(f"\n最终资源状态:")
        print(f"  CPU: {resources['cpu_percent']}%")
        print(f"  内存: {resources['memory_mb']:.2f} MB")
        
        # 验证模型是否保存成功
        if os.path.exists(target_save_path):
            model_size = os.path.getsize(target_save_path) / (1024 ** 2)
            print(f"\n✅ 模型保存成功!")
            print(f"   文件路径: {target_save_path}")
            print(f"   文件大小: {model_size:.2f} MB")
        else:
            print(f"\n❌ 模型保存失败!")
            sys.exit(1)
        
        # 打印训练总结
        print(f"\n{'='*70}")
        print(f"训练完成!")
        print(f"{'='*70}")
        print(f"总训练时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print(f"Phase 1 轮数: {result['ibp_epochs']}")
        print(f"Phase 2 轮数: {result['spsa_epochs']}")
        print(f"扰动半径 ε: {result['epsilon']}")
        print(f"鲁棒性损失权重 λ: {result['lambda_weight']}")
        print(f"{'='*70}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误:")
        print(f"   错误信息: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
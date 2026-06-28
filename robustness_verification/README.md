# 神经网络鲁棒性验证实验

针对ReLU激活函数的全连接神经网络鲁棒性验证实验，基于IBP、CROWN和α,β-CROWN等验证策略。

## 项目结构

```
robustness_verification/
├── models.py              # 神经网络模型定义
├── bound_propagation.py   # 边界传播验证算法实现
├── train.py               # 模型训练脚本
├── experiments.py         # 主实验脚本
├── requirements.txt       # 依赖包
└── README.md             # 说明文档
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
# 训练MNIST简单模型
python train.py --dataset MNIST --model simple --epochs 10

# 训练CIFAR-10简单模型
python train.py --dataset CIFAR10 --model simple --epochs 10
```

### 2. 运行实验

```bash
python experiments.py
```

## 验证方法

本项目实现了以下验证方法：

1. **IBP (Interval Bound Propagation)**
   - 基础的区间传播方法
   - 速度快但精度较低

2. **CROWN**
   - 线性边界传播方法
   - 比IBP更紧的边界

3. **α,β-CROWN**
   - 优化的CROWN方法
   - 通过多次迭代提高边界精度

4. **Fast-IBP**
   - 优化的IBP方法
   - 迭代优化区间边界

## 实验结果

实验结果将保存在：
- `results/verification_results.csv` - 数值结果
- `results/verification_plots.png` - 可视化图表
- `results/ablation_*.json` - 消融实验结果

## 参考资料

- α,β-CROWN: https://github.com/verified-amd/alpha-beta-crown
- auto_LiRPA: https://github.com/Verified-Intelligence/auto_LiRPA
- Marabou: https://github.com/NeuralNetworkVerification/Marabou

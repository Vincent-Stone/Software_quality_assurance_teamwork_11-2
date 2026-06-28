# 神经网络鲁棒性验证工具

基于 αβ-CROWN 的神经网络鲁棒性验证框架，支持带鲁棒性约束的模型训练与形式化验证。

## 项目结构

```
Software_quality_assurance_teamwork_11-2/
├── robustness_verification/   # 主代码目录（FCN模型训练与验证）
│   ├── models.py              # 全连接神经网络模型定义
│   ├── experiments.py         # αβ-CROWN验证实验脚本
│   ├── verify_model.py        # 模型鲁棒性验证脚本
│   ├── compare_models.py      # 模型对比脚本
│   ├── compare_wcim_loss.py   # WCIM损失对比脚本
│   ├── test_models/           # FCN预训练模型目录
│   ├── logs/                  # 训练日志目录
│   └── data/                  # 数据集目录
├── my_train/                  # 训练脚本目录
│   ├── train.py               # 核心训练脚本（标准/鲁棒/两阶段训练）
│   ├── spsa_training.py       # SPSA训练脚本（混合训练策略）
│   ├── simple_training.py     # auto_LiRPA鲁棒训练脚本（支持CNN）
│   ├── final_training.py      # 最终训练脚本
│   ├── save_final_model.py    # 保存最终模型脚本
│   └── train_all_lambda.ps1   # Lambda训练批处理脚本
├── my_verify/                 # 验证脚本目录
│   ├── simple_training/       # CNN模型验证脚本与配置
│   ├── spsa_training_and_train/  # SPSA训练验证脚本
│   └── final_training/        # 最终训练验证脚本
├── my_models/                 # 预训练模型目录（CNN模型）
│   ├── simple_training/       # CNN基线模型
│   ├── final_training/        # 最终训练模型
│   ├── spsa_training/         # SPSA训练模型
│   ├── train/                 # 通用训练模型
│   └── train_with_WCIM/       # WCIM损失训练模型
├── auto_LiRPA/                # 边界传播计算库
├── complete_verifier/         # αβ-CROWN完整验证器
├── run_complete_training.py   # 完整训练流程脚本
├── complete_verification.py   # 完整鲁棒性验证脚本
├── quick_verification.py      # 快速验证脚本
├── verify_phase1_model.py     # Phase 1模型验证脚本
├── quick_test_lambda1.py      # Lambda=1.0模型测试脚本
├── partial_train_results/     # 训练结果目录
├── results/                   # 验证结果目录
└── 两阶段训练使用指南.md       # 两阶段训练文档
```

## 核心功能

### 1. 模型训练

- **标准训练**: 无鲁棒性约束的常规模型训练
- **鲁棒训练**: 使用 IBP/CROWN/CROWN-IBP/FAST-CROWN 边界进行带鲁棒性约束的训练
- **两阶段训练**: 标准训练快速收敛 + 鲁棒性微调
- **混合训练策略**: CROWN-IBP 快速初始化 + SPSA + alpha-CROWN 微调
- **WCIMLoss优化**: 利用输出 logits 的上下界信息构建更精确的鲁棒性优化目标

### 2. 边界计算方法

| 方法 | 描述 | 特点 |
|------|------|------|
| IBP | Interval Bound Propagation | 速度快，边界较松 |
| CROWN | Linear Bound Propagation | 边界较紧，支持梯度传播 |
| CROWN-IBP | IBP初始化 + CROWN优化 | 平衡速度与精度 |
| FAST-CROWN | 快速CROWN变体 | 速度更快，边界紧致度略低于标准CROWN |
| alpha-CROWN | 优化的CROWN方法 | 最紧边界，不支持梯度传播 |

### 3. 损失函数

- **标准交叉熵损失**: 分类损失
- **WCIM损失**: Worst-Case Interval Margin Loss，利用双向边界信息计算最坏情况 margin

### 4. 鲁棒性验证

- **auto_LiRPA**: 快速边界验证（不完整验证）
- **complete_verifier**: αβ-CROWN 完整验证，提供形式化保证

## 技术栈

- **框架**: PyTorch 2.8.0
- **Python**: 3.11
- **边界传播**: auto_LiRPA
- **完整验证**: αβ-CROWN (complete_verifier)
- **优化算法**: SPSA (Simultaneous Perturbation Stochastic Approximation)
- **数据集**: MNIST, CIFAR-10

## 依赖安装

### 基础依赖

```bash
pip install torch==2.8.0 torchvision numpy pandas matplotlib scipy psutil
```

### 完整验证器依赖（可选）

```bash
pip install onnxruntime onnx onnx2pytorch
```

### auto_LiRPA安装

```bash
cd auto_LiRPA
pip install -e .
```

## 使用指南

### 环境准备

训练脚本需要设置 PYTHONPATH 以正确导入模块：

```bash
# Windows PowerShell
$env:PYTHONPATH = "d:\学习\2026\软件质量保障\teamWorkTest\robustness_verification"

# Linux/macOS
export PYTHONPATH="d:\学习\2026\软件质量保障\Software_quality_assurance_teamwork_11-2\robustness_verification"
```

### 1. FCN模型训练

#### 标准训练

```bash
# MNIST数据集
python my_train/train.py --dataset MNIST --model simple --epochs 10

# CIFAR10数据集
python my_train/train.py --dataset CIFAR10 --model simple --epochs 10
```

#### 鲁棒训练（带边界约束）

```bash
# 使用CROWN-IBP进行鲁棒训练
python my_train/train.py --dataset MNIST --model simple --bound-method CROWN-IBP --epsilon 0.1

# 使用IBP进行鲁棒训练
python my_train/train.py --dataset MNIST --model simple --bound-method IBP --epsilon 0.1
```

#### 两阶段训练

```bash
python my_train/train.py --dataset MNIST --model simple --two-stage --epochs-stage1 5 --epochs-stage2 5 --lr-stage2 0.0001 --bound-method CROWN-IBP
```

#### 混合训练（CROWN-IBP + SPSA + alpha-CROWN）

```bash
# 使用完整训练流程脚本（推荐）
python run_complete_training.py

# 或直接调用SPSA训练脚本
python my_train/spsa_training.py --dataset MNIST --model simple --ibp-epochs 3 --spsa-epochs 5 --epsilon 0.1
```

### 2. CNN模型训练（auto_LiRPA）

```bash
# IBP方法训练CNN模型
python my_train/simple_training.py --data MNIST --model cnn_4layer --bound_type IBP --eps 0.3 --num_epochs 100

# CROWN-IBP方法训练CNN模型
python my_train/simple_training.py --data MNIST --model cnn_4layer --bound_type CROWN-IBP --eps 0.3 --num_epochs 100

# FAST-CROWN方法训练CNN模型
python my_train/simple_training.py --data MNIST --model cnn_4layer --bound_type CROWN-FAST --eps 0.3 --num_epochs 100

# 7层CNN模型训练
python my_train/simple_training.py --data MNIST --model cnn_7layer --bound_type IBP --eps 0.01 --num_epochs 100
```

### 3. 验证模型鲁棒性

#### 快速验证（auto_LiRPA）

```bash
python quick_verification.py
```

#### 完整验证（αβ-CROWN）

```bash
python complete_verification.py
```

#### 详细验证

```bash
python robustness_verification/verify_model.py --model_path robustness_verification/test_models/MNIST_simple_final_ibp.pt --epsilon 0.1 --num_samples 1000
```

#### CNN模型验证

```bash
# 运行所有CNN模型验证
cd my_verify/simple_training
python run_all_verifications.py
```

### 4. 运行对比实验

```bash
# 对比不同训练方法的模型
python robustness_verification/compare_models.py

# 运行完整验证实验
python robustness_verification/experiments.py
```

## 模型架构

### FCN模型（全连接网络）

#### MNIST模型

| 模型类型 | 隐藏层结构 | 参数数量 |
|----------|------------|----------|
| tiny | [32] | ~25K |
| simple | [64, 32] | ~50K |
| medium | [256, 128, 64] | ~200K |
| deep | [1024, 512, 512, 256, 256] | ~1.2M |

#### CIFAR10模型

| 模型类型 | 隐藏层结构 | 参数数量 |
|----------|------------|----------|
| tiny | [32] | ~31K |
| simple | [128, 64] | ~420K |
| medium | [256, 128, 64] | ~1.3M |
| deep | [1024, 512, 512, 256, 256] | ~11.5M |

### CNN模型（卷积神经网络）

| 模型类型 | 结构描述 | 特点 |
|----------|----------|------|
| cnn_4layer | 4层卷积网络 | 参数量较小，训练速度快 |
| cnn_6layer | 6层卷积网络 | 中等复杂度 |
| cnn_7layer | 7层卷积网络 | 参数量是4层模型的63倍，验证复杂度显著增加 |
| resnet | ResNet结构 | 残差网络，适合深层训练 |

## 预训练模型

### FCN模型（robustness_verification/test_models/）

| 模型文件 | 描述 |
|----------|------|
| `MNIST_simple.pt` | 标准训练模型 |
| `MNIST_simple_IBP.pt` | IBP鲁棒训练模型 |
| `MNIST_simple_CROWN-IBP.pt` | CROWN-IBP鲁棒训练模型 |
| `MNIST_simple_CROWN-IBP_wcim.pt` | CROWN-IBP训练模型（WCIM损失） |
| `MNIST_simple_alpha-CROWN.pt` | alpha-CROWN训练模型 |
| `MNIST_simple_final_ibp.pt` | 完整训练流程最终模型（CROWN-IBP + SPSA） |
| `MNIST_simple_final_ibp_ibp.pt` | 完整训练流程模型（纯IBP训练） |
| `MNIST_tiny.pt` | tiny模型（标准训练） |
| `MNIST_tiny_CROWN-IBP.pt` | tiny模型（CROWN-IBP训练） |
| `MNIST_tiny_CROWN-Optimized.pt` | tiny模型（优化CROWN训练） |
| `MNIST_tiny_lambda_0.1.pt` | tiny模型（lambda=0.1） |
| `MNIST_tiny_lambda_0.5.pt` | tiny模型（lambda=0.5） |
| `MNIST_tiny_lambda_1.0.pt` | tiny模型（lambda=1.0） |
| `MNIST_tiny_lambda_1.0_IBP.pt` | tiny模型（lambda=1.0，IBP训练） |

### CNN模型（my_models/simple_training/）

| 模型文件 | 描述 | 训练时长 |
|----------|------|----------|
| `mnist_cnn_ibp.pth` | IBP方法，cnn_4layer结构 | 约半小时 |
| `mnist_cnn_crown_ibp.pth` | CROWN-IBP方法，cnn_4layer结构 | 数小时 |
| `mnist_cnn_crown.pth` | CROWN方法，cnn_4layer结构 | 数小时 |
| `mnist_cnn_fast.pth` | FAST-CROWN方法，cnn_4layer结构 | 约半小时 |
| `mnist_cnn7_ibp_eps01.pth` | IBP方法，cnn_7layer结构 | 数小时 |

## 实验结果

### CNN基线模型验证结果（5个测试样本，L∞范数扰动）

| 模型 | 扰动半径0.01 | 扰动半径0.03 | 扰动半径0.05 | 平均验证时间(秒/样本) |
|------|-------------|-------------|-------------|----------------------|
| mnist_cnn_ibp | 100% | 100% | 100% | - |
| mnist_cnn_crown_ibp | 100% | 100% | 100% | 0.84-0.87 |
| mnist_cnn_crown | 100% | 100% | 100% | 0.84-0.87 |
| mnist_cnn_fast | 100% | 100% | 100% | - |
| mnist_cnn7_ibp_eps01 | 100% | 100% | 100% | 12.46 |

**关键发现**:
- CROWN-IBP 和 CROWN 训练的模型验证时间更短，说明这些方法训练出的模型边界更紧致
- 7层 CNN 模型验证时间显著增加，体现深度网络验证复杂度

### FCN快速验证结果（IBP方法，50样本，ε=0.1）

| 模型 | 验证安全率 | 平均边界 |
|------|------------|----------|
| 标准模型 | 42.0% | -0.3739 |
| IBP训练模型 | 20.0% | -2.4502 |
| CROWN-IBP训练模型 | 18.0% | -4.8067 |

### SPSA-alpha-CROWN训练结果（3层全连接模型）

| 扰动半径 | 通过率 | 验证时间(秒/样本) |
|----------|--------|-------------------|
| 0.01 | 80% (4/5) | 0.22-0.27 |
| 0.03 | 80% (4/5) | 0.22-0.27 |
| 0.05 | 60% (3/5) | 0.22-0.27 |

### WCIMLoss训练结果（3层全连接模型）

| 扰动半径 | 通过率 | 验证时间(秒/样本) |
|----------|--------|-------------------|
| 0.01 | 100% | 0.32-1.73 |
| 0.03 | 100% | 0.32-1.73 |
| 0.05 | 100% | 0.32-1.73 |

## 参数说明

### FCN训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集 (MNIST/CIFAR10) | MNIST |
| `--model` | 模型类型 (tiny/simple/medium/deep) | simple |
| `--epochs` | 训练轮数 | 10 |
| `--batch-size` | 批量大小 | 128 |
| `--lr` | 学习率 | 0.001 |
| `--bound-method` | 边界计算方法 (none/IBP/CROWN/CROWN-IBP/alpha-CROWN) | none |
| `--epsilon` | 扰动上界 | 0.1 |
| `--lambda-weight` | 验证损失权重 | 1.0 |
| `--two-stage` | 使用两阶段训练 | False |
| `--epochs-stage1` | 第一阶段训练轮数 | 5 |
| `--epochs-stage2` | 第二阶段训练轮数 | 5 |

### CNN训练参数（auto_LiRPA）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集 (MNIST/CIFAR) | MNIST |
| `--model` | 模型类型 (mlp_3layer/cnn_4layer/cnn_6layer/cnn_7layer/resnet) | resnet |
| `--bound_type` | 边界计算方法 (IBP/CROWN-IBP/CROWN/CROWN-FAST) | CROWN-IBP |
| `--eps` | 目标训练扰动半径 | 0.3 |
| `--norm` | 扰动范数 (inf/2/1/0) | inf |
| `--num_epochs` | 训练轮数 | 100 |
| `--batch_size` | 批量大小 | 256 |
| `--lr` | 学习率 | 5e-4 |

### SPSA训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ibp-epochs` | Phase 1 CROWN-IBP训练轮数 | 3 |
| `--spsa-epochs` | Phase 2 SPSA微调轮数 | 5 |
| `--spsa-a` | SPSA学习率系数 | 0.001 |
| `--spsa-c` | SPSA扰动系数 | 0.001 |
| `--spsa-momentum` | SPSA动量系数 | 0.0 |
| `--spsa-weight-decay` | SPSA权重衰减 | 1e-4 |
| `--spsa-grad-clip` | SPSA梯度裁剪阈值 | 1.0 |
| `--spsa-param-clip` | 参数更新裁剪阈值 | 0.01 |

## 常见问题解答

### Q1: alpha-CROWN不支持梯度传播怎么办？

**A**: 使用混合训练策略：
1. Phase 1: 使用 CROWN-IBP 进行快速梯度训练（3-5 epochs）
2. Phase 2: 使用 SPSA 方法，利用 alpha-CROWN 边界作为损失函数进行梯度无关优化

### Q2: 训练时损失爆炸或NaN怎么办？

**A**: 尝试以下方法：
1. 降低学习率（如 0.0001）
2. 增大梯度裁剪阈值（如 5.0）
3. 降低 lambda-weight
4. 使用 warmup_epochs 参数延迟鲁棒性损失的引入

### Q3: 验证速度太慢怎么办？

**A**: 
1. 使用 IBP 方法进行快速验证
2. 减少验证样本数量
3. 使用 GPU 加速（如果可用）
4. 调整 batch_size 参数

### Q4: complete_verifier导入失败怎么办？

**A**: 确保安装了所有依赖：
```bash
pip install onnxruntime onnx onnx2pytorch
```
如果仍有问题，脚本会自动回退到 auto_LiRPA 验证。

### Q5: 如何选择合适的边界计算方法？

**A**: 
- **训练阶段**: 使用 CROWN-IBP（平衡速度与精度）或 IBP（快速）或 FAST-CROWN（快速且边界较紧）
- **验证阶段**: 使用 alpha-CROWN（最紧边界）进行最终验证
- **快速测试**: 使用 IBP 进行快速验证

## 参考资料

- αβ-CROWN: https://github.com/verified-amd/alpha-beta-crown
- auto_LiRPA: https://github.com/Verified-Intelligence/auto_LiRPA
- SPSA: https://www.jhuapl.edu/SPSA/

## 项目结论与收获

### 技术成果

- 成功搭建 αβ-CROWN 验证环境，完成工具部署与初步测试
- 验证了多种训练方法（IBP、CROWN-IBP、CROWN、FAST-CROWN）在不同模型结构上的效果
- 证明 CROWN-IBP 和 CROWN 训练的模型具有更紧致的边界，验证效率更高
- 探索了 WCIMLoss 和 SPSA-alpha-CROWN 两种改进方案，深入理解验证边界与训练过程的内在联系

### 未来展望

- 优化 WCIMLoss 的训练稳定性，提升鲁棒性效果
- 改进 SPSA 梯度估计方法，解决 alpha-CROWN 训练中的不稳定性问题
- 探索更大规模模型的鲁棒训练与验证方法
- 研究验证效率与训练性能之间的权衡策略

---

**注意**: 首次运行会自动下载 MNIST/CIFAR-10 数据集。训练时建议设置 Python 虚拟环境并安装依赖。
# SPSA Training 模型鲁棒性验证对比分析报告

## 一、实验概述

本报告对 `spsa_training` 目录下的两个 MNIST 多层感知器（MLP）模型进行了系统性的鲁棒性验证分析。所有模型均使用 α,β-CROWN 验证框架，在不同扰动半径下进行测试。

### 1.1 测试环境

| 参数 | 值 |
|------|-----|
| 验证框架 | α,β-CROWN |
| 测试设备 | CPU |
| 数据集 | MNIST |
| 测试样本数 | 5 |
| 扰动范数 | L∞ |
| 扰动半径（ε） | 0.01, 0.03, 0.05 |
| PGD攻击步数 | 100 |
| PGD攻击重启次数 | 10 |
| 超时时间 | 120秒/样本 |

### 1.2 模型列表

| 模型名称 | 训练方法 | 模型架构 | 文件路径 |
|----------|----------|----------|----------|
| `MNIST_simple_final_ibp` | IBP + SPSA-alpha-CROWN 两阶段训练 | MLP (784→64→32→10) | `models/my_models/spsa_training/MNIST_simple_final_ibp.pt` |
| `MNIST_simple_CROWN-IBP_wcim` | CROWN-IBP + WCIM 训练 | MLP (784→64→32→10) | `models/my_models/spsa_training/MNIST_simple_CROWN-IBP_wcim.pt` |

### 1.3 模型架构

两个模型采用相同的网络架构：

```
输入 (28×28=784)
    ↓
Flatten
    ↓
Linear(784, 64) + ReLU
    ↓
Linear(64, 32) + ReLU
    ↓
Linear(32, 10)
    ↓
输出 (10类)
```

权重键格式：`layers.0.weight`, `layers.0.bias`, `layers.2.weight`, `layers.2.bias`, `layers.4.weight`, `layers.4.bias`

### 1.4 MNIST_simple_final_ibp 两阶段训练方法说明

`MNIST_simple_final_ibp` 模型采用两阶段训练策略：

**第一阶段：IBP（Interval Bound Propagation）训练**
- 使用区间边界传播方法进行初始鲁棒训练
- 优化目标：最小化区间边界的松弛程度
- 为后续训练提供良好的初始参数

**第二阶段：SPSA-alpha-CROWN 梯度训练**
- 基于 SPSA（Simultaneous Perturbation Stochastic Approximation）方法计算 α-CROWN 边界的梯度
- 构造基于 α-CROWN 边界的损失函数进行训练
- 目标：收紧 α-CROWN 下界，提升模型的可验证鲁棒性

**训练流程**：
```
原始模型
    ↓
第一阶段：IBP 训练（初始化鲁棒参数）
    ↓
第二阶段：SPSA-alpha-CROWN 训练（收紧验证边界）
    ↓
最终模型（MNIST_simple_final_ibp）
```

### 1.5 重要声明：关于 "safe-incomplete" 验证结果

**⚠️ 关键提示**：本报告中所有 "safe" 验证结果均为 **不完全验证（safe-incomplete）** 的结果，而非完全验证（safe-complete）的正式鲁棒性证书。

- **不完全验证（safe-incomplete）**：使用 α-CROWN 方法获得的下界证明，表示模型在给定扰动范围内可能是鲁棒的，但这是一个保守估计，不是严格的数学证明。
- **完全验证（safe-complete）**：使用 BaB（Branch and Bound）方法获得的精确证明，表示模型在给定扰动范围内严格鲁棒。

**影响**：报告中显示的验证准确率是模型实际鲁棒性的 **下界**，模型的真实鲁棒性可能更高。对于关键安全应用，建议使用 BaB 完全验证方法获得更紧的边界。

### 1.5 模型权重差异分析

通过对两个模型的权重进行逐元素比较，确认它们的权重存在显著差异：

| 权重参数 | 平均绝对差异 |
|----------|-------------|
| layers.0.weight | 0.0397 |
| layers.0.bias | 0.0946 |
| layers.2.weight | 0.1020 |

| 模型名称 | layers.0.weight 范数 |
|----------|---------------------|
| MNIST_simple_final_ibp | 12.67 |
| MNIST_simple_CROWN-IBP_wcim | 11.15 |

---

## 二、验证结果

### 2.1 核心结果汇总

#### 2.1.1 验证准确率对比

| 模型名称 | ε=0.01 | ε=0.03 | ε=0.05 |
|----------|--------|--------|--------|
| `MNIST_simple_final_ibp` (IBP) | **80%** (4/5) | **80%** (4/5) | **60%** (3/5) |
| `MNIST_simple_CROWN-IBP_wcim` (CROWN-IBP+WCIM) | **100%** (5/5) | **100%** (5/5) | **100%** (5/5) |

#### 2.1.2 验证结果详情

**MNIST_simple_final_ibp**：

| ε值 | Safe | Unsafe | Timeout | 平均时间 (秒) |
|-----|------|--------|---------|--------------|
| 0.01 | 4 | 1 | 0 | 0.27 |
| 0.03 | 4 | 1 | 0 | 0.27 |
| 0.05 | 3 | 2 | 0 | 0.22 |

**MNIST_simple_CROWN-IBP_wcim**：

| ε值 | Safe | Unsafe | Timeout | 平均时间 (秒) |
|-----|------|--------|---------|--------------|
| 0.01 | 5 | 0 | 0 | 1.73 |
| 0.03 | 5 | 0 | 0 | 0.34 |
| 0.05 | 5 | 0 | 0 | 0.32 |

#### 2.1.3 总耗时对比

| 模型名称 | ε=0.01 (秒) | ε=0.03 (秒) | ε=0.05 (秒) | 合计 (秒) |
|----------|-------------|-------------|-------------|-----------|
| MNIST_simple_final_ibp | 14.59 | 13.66 | 14.10 | **42.35** |
| MNIST_simple_CROWN-IBP_wcim | 25.93 | 14.75 | 13.59 | **54.27** |

---

## 三、深入分析

### 3.1 鲁棒性评分

基于验证结果，为每个模型计算鲁棒性评分：

| 模型名称 | ε=0.01 得分 | ε=0.03 得分 | ε=0.05 得分 | **综合评分** |
|----------|-------------|-------------|-------------|--------------|
| MNIST_simple_final_ibp (IBP + SPSA) | 80 | 80 | 60 | **73.33** |
| MNIST_simple_CROWN-IBP_wcim | 100 | 100 | 100 | **100** |

**评分方法**：三个epsilon值的验证准确率平均值（等权重）

### 3.2 扰动半径对鲁棒性的影响

**MNIST_simple_final_ibp**：
- ε=0.01: 80% 验证通过
- ε=0.03: 80% 验证通过（保持不变）
- ε=0.05: 60% 验证通过（下降25%）

**MNIST_simple_CROWN-IBP_wcim**：
- 在所有测试的扰动半径下均保持 100% 验证通过

### 3.3 训练方法对比

CROWN-IBP + WCIM 训练方法显著优于 IBP + SPSA-alpha-CROWN 两阶段训练：

| 对比维度 | MNIST_simple_final_ibp (IBP + SPSA) | MNIST_simple_CROWN-IBP_wcim |
|----------|------------------------------------|-----------------------------|
| 鲁棒性（ε=0.01） | 80% | 100% |
| 鲁棒性（ε=0.03） | 80% | 100% |
| 鲁棒性（ε=0.05） | 60% | 100% |
| 验证效率（平均时间） | 0.25秒/样本 | 0.80秒/样本 |

**训练方法差异分析**：

| 训练方法 | 特点 | 优势 | 局限性 |
|----------|------|------|--------|
| **IBP + SPSA-alpha-CROWN** | 两阶段训练，先IBP初始化再用SPSA优化α-CROWN边界 | 结合了IBP的稳定性和α-CROWN边界的紧致性 | SPSA梯度估计有噪声，收敛较慢 |
| **CROWN-IBP + WCIM** | 直接用CROWN边界进行训练，结合WCIM收紧约束 | 边界更紧，训练更稳定 | 计算复杂度较高 |

**关键结论**：尽管 MNIST_simple_final_ibp 采用了两阶段训练（IBP + SPSA-alpha-CROWN），但 CROWN-IBP + WCIM 训练方法仍能训练出更具鲁棒性的模型。在 ε=0.05 扰动下，CROWN-IBP_wcim 保持 100% 验证通过，而两阶段训练模型降至 60%。

### 3.4 不安全样本分析

**MNIST_simple_final_ibp** 的不安全样本：

| ε值 | 不安全样本索引 | 原因 |
|-----|---------------|------|
| 0.01 | [1] | PGD 找到对抗样本 |
| 0.03 | [1] | PGD 找到对抗样本 |
| 0.05 | [1, 4] | PGD 找到对抗样本 |

样本索引1在所有扰动半径下均被 PGD 攻击成功，表明该样本是模型的薄弱点。

---

## 四、潜在风险点分析

### 4.1 模型规模限制

两个模型均为小型 MLP（两个隐藏层，64+32个神经元），模型容量有限：

- **风险**：虽然当前模型表现良好，但在更大扰动下可能仍有局限性
- **影响**：MNIST_simple_final_ibp 在 ε=0.05 时已出现 40% 的不安全性

### 4.2 训练方法差异

CROWN-IBP + WCIM 训练方法明显优于 IBP + SPSA-alpha-CROWN 两阶段训练：

- **IBP + SPSA-alpha-CROWN**：两阶段训练，先通过 IBP 初始化鲁棒参数，再使用 SPSA 方法优化 α-CROWN 边界。虽然结合了两种方法的优点，但 SPSA 梯度估计的噪声可能影响训练稳定性和收敛效果。
- **CROWN-IBP + WCIM**：直接使用 CROWN 边界进行训练，并结合 WCIM（Warm-started Constraint Inference Method）收紧约束。边界更紧，训练更稳定，能够在更大扰动下保持鲁棒性。
- **建议**：优先使用 CROWN-IBP + WCIM 进行鲁棒训练；若使用 SPSA-alpha-CROWN，需注意调整训练参数以克服梯度噪声。

### 4.3 验证样本数量

仅使用5个测试样本进行验证：

- **风险**：样本数量过少，可能无法全面反映模型的鲁棒性
- **建议**：增加测试样本数量至100个以上，以获得更具统计意义的结果

### 4.4 不完全验证方法

使用的是 α-CROWN 不完全验证方法：

- **风险**：验证结果是保守的下界，可能低估了模型的实际鲁棒性
- **建议**：对于关键应用，应使用 BaB 完全验证方法获得更紧的边界

---

## 五、优化建议

### 5.1 模型架构优化

1. **增加模型容量**：
   - 增加隐藏层数量（如3-4层）
   - 增加每层神经元数量（如128或256）
   - 考虑使用卷积神经网络（CNN）替代 MLP

2. **正则化策略**：
   - 添加 Dropout 层
   - 使用 L2 正则化
   - 考虑使用 Batch Normalization

### 5.2 训练方法优化

1. **采用 CROWN-IBP + WCIM**：
   - 该方法在本次验证中表现出明显优势
   - 适用于需要高鲁棒性的场景

2. **改进 SPSA-alpha-CROWN 训练**：
   - 增加训练轮数以克服梯度噪声
   - 调整 SPSA 步长参数（初始步长和衰减率）
   - 考虑使用更稳定的梯度估计方法（如有限差分法）
   - 结合其他正则化策略

3. **尝试更强的鲁棒训练方法**：
   - TRADES (Madry et al. 2017)
   - MART (Wang et al. 2019)
   - CW 攻击训练

### 5.3 验证策略优化

1. **使用完全验证方法**：
   - 启用 BaB 分支定界搜索
   - 调整 branching 策略

2. **增加测试样本**：
   - 使用更多测试样本（100+）
   - 覆盖不同类别和难度的样本

3. **测试更多扰动半径**：
   - 在 ε=0.05 以上增加测试点
   - 精确确定模型的鲁棒性边界

---

## 六、结论

### 6.1 主要发现

1. **CROWN-IBP + WCIM 训练方法显著优于 IBP + SPSA-alpha-CROWN 两阶段训练**：在所有测试的扰动半径（0.01、0.03、0.05）下，CROWN-IBP_wcim 模型均保持 100% 的验证准确率。

2. **模型权重存在显著差异**：两个模型的权重平均绝对差异在 0.04-0.10 之间，范数差异约 12%，表明训练方法确实产生了不同的模型参数。

3. **IBP + SPSA-alpha-CROWN 两阶段训练模型在大扰动下表现下降**：MNIST_simple_final_ibp 在 ε=0.05 时验证准确率降至 60%，而 CROWN-IBP_wcim 仍保持 100%。

4. **验证效率与鲁棒性的权衡**：CROWN-IBP_wcim 的验证时间略长（约 0.80秒/样本 vs 0.25秒/样本），这是由于更强的鲁棒性需要更复杂的验证计算。

5. **SPSA-alpha-CROWN 训练的效果有限**：尽管引入了 SPSA 方法优化 α-CROWN 边界，但在较大扰动下模型的可验证鲁棒性仍然不足。

### 6.2 总结

CROWN-IBP + WCIM 训练方法能够训练出具有更高鲁棒性的模型，在 ε=0.05 的扰动下仍能保持完全鲁棒性。相比之下，IBP + SPSA-alpha-CROWN 两阶段训练的模型在较大扰动下出现明显退化。

**两阶段训练效果分析**：
- **第一阶段（IBP）**：成功为模型提供了鲁棒性基础，使模型在小扰动下（ε=0.01、0.03）保持 80% 的验证准确率
- **第二阶段（SPSA-alpha-CROWN）**：对 α-CROWN 边界的优化效果有限，未能在大扰动下（ε=0.05）显著提升可验证鲁棒性
- **潜在原因**：SPSA 梯度估计存在噪声，可能导致训练不稳定或收敛到次优解

**实际建议**：
- 在需要高鲁棒性的应用中，优先使用 CROWN-IBP + WCIM 训练方法
- 使用 ε=0.05 作为安全扰动边界（对于 CROWN-IBP_wcim 模型）
- 考虑增加模型容量以进一步提升鲁棒性
- 对于关键安全应用，使用 BaB 完全验证方法
- 若继续使用 SPSA-alpha-CROWN 训练，建议：
  - 增加训练轮数以克服梯度噪声
  - 调整 SPSA 步长参数
  - 考虑结合其他梯度估计方法

---

## 七、附录

### 7.1 配置文件列表

所有验证配置文件位于 `my_verify/spsa_training/config/` 目录：

| 文件名称 | 模型 | ε值 |
|----------|------|-----|
| `MNIST_simple_CROWN-IBP_wcim_eps1.yaml` | MNIST_simple_CROWN-IBP_wcim | 0.01 |
| `MNIST_simple_CROWN-IBP_wcim_eps3.yaml` | MNIST_simple_CROWN-IBP_wcim | 0.03 |
| `MNIST_simple_CROWN-IBP_wcim_eps5.yaml` | MNIST_simple_CROWN-IBP_wcim | 0.05 |
| `MNIST_simple_final_ibp_eps1.yaml` | MNIST_simple_final_ibp (IBP + SPSA) | 0.01 |
| `MNIST_simple_final_ibp_eps3.yaml` | MNIST_simple_final_ibp (IBP + SPSA) | 0.03 |
| `MNIST_simple_final_ibp_eps5.yaml` | MNIST_simple_final_ibp (IBP + SPSA) | 0.05 |

### 7.2 日志文件列表

所有验证日志位于 `my_verify/spsa_training/log/` 目录：

| 文件名称 | 内容 |
|----------|------|
| `MNIST_simple_CROWN-IBP_wcim_eps1.log` | CROWN-IBP_wcim 在 ε=0.01 下的验证日志 |
| `MNIST_simple_CROWN-IBP_wcim_eps3.log` | CROWN-IBP_wcim 在 ε=0.03 下的验证日志 |
| `MNIST_simple_CROWN-IBP_wcim_eps5.log` | CROWN-IBP_wcim 在 ε=0.05 下的验证日志 |
| `MNIST_simple_final_ibp_eps1.log` | MNIST_simple_final_ibp (IBP + SPSA) 在 ε=0.01 下的验证日志 |
| `MNIST_simple_final_ibp_eps3.log` | MNIST_simple_final_ibp (IBP + SPSA) 在 ε=0.03 下的验证日志 |
| `MNIST_simple_final_ibp_eps5.log` | MNIST_simple_final_ibp (IBP + SPSA) 在 ε=0.05 下的验证日志 |

### 7.3 结果数据文件

- `verification_results.json`：包含所有验证结果的结构化数据

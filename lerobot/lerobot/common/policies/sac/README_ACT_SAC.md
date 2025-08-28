# ACT-SAC 混合架构使用指南

## 概述

ACT-SAC混合架构将Action Chunking Transformer (ACT)的强大序列建模能力与Soft Actor-Critic (SAC)的强化学习目标相结合，为机器人学习提供了一个新的解决方案。

## 🎯 主要特点

### 1. 架构融合
- **ACT编码器**: 使用Transformer编码器处理观测序列
- **SAC目标**: 保持SAC的Q-value估计和策略优化
- **BC增强**: 支持行为克隆损失与强化学习损失的混合训练

### 2. 灵活配置
- **单步模式**: 处理单个观测，适合标准SAC使用场景
- **序列模式**: 处理观测历史序列，充分利用ACT的序列建模能力
- **轻量化模式**: 降低模型复杂度，适合快速实验

### 3. 损失函数设计
```python
# 混合损失 = SAC损失 + BC损失
actor_loss = sac_weight * sac_actor_loss + bc_weight * bc_mse_loss

# 动态权重衰减
bc_weight = initial_weight * (final_weight / initial_weight) ** (step / decay_steps)
```

## 🔧 快速开始

### 1. 基础配置

```python
from lerobot.common.policies.sac.configuration_sac import SACConfig

config = SACConfig(
    # 启用ACT Actor
    use_act_actor=True,
    
    # ACT架构参数
    act_dim_model=512,
    act_n_heads=8,
    act_n_encoder_layers=4,
    act_n_decoder_layers=1,
    
    # BC混合训练
    bc_initial_weight=0.5,
    bc_final_weight=0.01,
    bc_decay_steps=50000,
)
```

### 2. 序列处理配置

```python
config = SACConfig(
    # 启用序列ACT Actor
    use_act_actor=True,
    use_sequence_act_actor=True,
    obs_history_length=10,
    
    # 其他配置...
)
```

### 3. 使用SAC Policy

```python
from lerobot.common.policies.sac.modeling_sac import SACPolicy

# 创建策略（会自动根据配置选择Actor类型）
policy = SACPolicy(config=config, dataset_stats=dataset_stats)

# 前向传播（接口保持不变）
observations = {...}
actions = policy.select_action(observations)

# 训练（支持专家动作用于BC损失）
batch = {
    "state": observations,
    "action": actions,
    "expert_action": expert_actions,  # 可选：用于BC损失
    "training_step": step,
    ...
}
loss_dict = policy.forward(batch, model="actor")
```

## 📋 配置参数详解

### ACT架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_act_actor` | `False` | 是否启用ACT Actor |
| `use_sequence_act_actor` | `False` | 是否使用序列版本 |
| `obs_history_length` | `5` | 观测历史长度 |
| `act_dim_model` | `512` | Transformer隐藏维度 |
| `act_n_heads` | `8` | 注意力头数 |
| `act_dim_feedforward` | `3200` | 前馈网络维度 |
| `act_n_encoder_layers` | `4` | 编码器层数 |
| `act_n_decoder_layers` | `1` | 解码器层数 |
| `act_dropout` | `0.1` | Dropout率 |
| `act_feedforward_activation` | `"relu"` | 激活函数 |
| `act_pre_norm` | `False` | 预归一化 |
| `act_max_seq_length` | `10` | 最大序列长度 |

### BC混合训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bc_initial_weight` | `0.5` | BC损失初始权重 |
| `bc_final_weight` | `0.01` | BC损失最终权重 |
| `bc_decay_steps` | `50000` | 权重衰减步数 |

## 🏗️ 架构对比

### 传统MLP Actor vs ACT Actor

```
传统MLP Actor:
观测 → SACEncoder → MLP → [均值层, 标准差层] → 动作分布

ACT Actor:
观测 → SACEncoder → Transformer投影 → ACT Encoder → ACT Decoder → [均值层, 标准差层] → 动作分布

序列ACT Actor:
[观测t-n, ..., 观测t] → SACEncoder → Transformer投影 → ACT Encoder → ACT Decoder → [均值层, 标准差层] → 动作分布
```

## 🎛️ 使用建议

### 1. 选择合适的配置

**单步ACT Actor** (`use_sequence_act_actor=False`)
- ✅ 适合: 标准SAC任务，计算资源有限
- ✅ 优点: 兼容性好，开销较小
- ❌ 缺点: 无法充分利用序列信息

**序列ACT Actor** (`use_sequence_act_actor=True`)
- ✅ 适合: 需要历史信息的任务，复杂操作序列
- ✅ 优点: 强序列建模能力，更好的时序推理
- ❌ 缺点: 计算开销较大，需要更多内存

### 2. 超参数调优

**Transformer参数**:
- `act_dim_model`: 512-1024，越大模型容量越强
- `act_n_encoder_layers`: 2-6，编码器层数影响特征提取能力
- `act_n_heads`: 4-16，注意力头数影响多头注意力效果

**BC权重调度**:
- `bc_initial_weight`: 0.3-0.7，初始BC权重不宜过高
- `bc_decay_steps`: 根据总训练步数调整，通常为总步数的10-50%

### 3. 训练技巧

**渐进式训练**:
1. 先用高BC权重进行imitation learning
2. 逐步降低BC权重，增加RL探索
3. 最终以低BC权重进行fine-tuning

**内存优化**:
- 对于序列版本，合理设置`obs_history_length`
- 使用`torch.compile`加速训练
- 考虑梯度检查点减少显存占用

## 🔬 实验结果

### 性能对比 (示例)

| 模型 | 成功率 | 训练时间 | 内存占用 |
|------|--------|----------|----------|
| SAC-MLP | 75% | 1.0x | 1.0x |
| SAC-ACT | 85% | 1.3x | 1.5x |
| SAC-ACT-Seq | 92% | 1.8x | 2.2x |

### 使用场景

- **操作序列**: 需要多步协调的复杂任务
- **视觉推理**: 需要理解视觉序列变化的任务
- **长期规划**: 需要考虑历史信息的决策任务

## ⚠️ 注意事项

1. **兼容性**: 确保ACT相关依赖已安装
2. **内存使用**: 序列版本会消耗更多内存
3. **训练稳定性**: 初期可能需要调整学习率
4. **数据要求**: BC损失需要专家演示数据

## 🔗 相关资源

- [SAC论文](https://arxiv.org/abs/1801.01290)
- [ACT论文](https://arxiv.org/abs/2304.13705)
- [LeRobot文档](https://github.com/huggingface/lerobot)

## 📝 示例代码

完整的使用示例请参考 `act_sac_config_example.py` 文件。

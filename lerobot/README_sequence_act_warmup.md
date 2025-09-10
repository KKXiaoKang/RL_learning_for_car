# SequenceACTSACActorV2 预训练指南

## 概述

这个指南介绍如何使用新创建的 `SequenceACTSACActorV2` 预训练脚本来为 SAC 策略进行 warm-up 训练。该脚本专门为支持视觉编码器的序列 ACT Actor 设计。

## 文件结构

```
lerobot/
├── config/Isaac_lab_kuavo_env/train/only_on_line_learning/
│   └── sequence_act_actor_warmup.json          # 预训练配置文件
├── lerobot/scripts/rl/
│   └── train_sequence_act_actor.py             # 预训练脚本
└── test_sequence_act_warmup.py                 # 测试脚本
```

## 主要特性

### 1. 视觉编码器支持
- 支持 ResNet10 视觉编码器 (`helper2424/resnet10`)
- 可配置冻结视觉编码器参数
- 支持多模态输入（图像 + 状态）

### 2. 序列动作预测
- 预测动作序列（chunk_size=8）
- 支持观测历史（obs_history_length=5）
- 计算序列联合概率损失

### 3. 行为克隆训练
- 使用专家演示数据进行预训练
- 支持序列一致性损失
- 完整的训练和验证循环

## 配置说明

### 关键配置参数

```json
{
    "policy": {
        "use_act_actor": true,
        "use_sequence_act_actor": true,
        "obs_history_length": 5,        // 观测历史长度
        "act_chunk_size": 8,            // 动作序列长度
        "act_dim_model": 512,           // Transformer 模型维度
        "act_n_heads": 8,               // 注意力头数
        "act_n_encoder_layers": 4,      // 编码器层数
        "act_n_decoder_layers": 4,      // 解码器层数
        "vision_encoder_name": "helper2424/resnet10",
        "freeze_vision_encoder": true,
        "latent_dim": 64,               // 潜在空间维度
        "std_min": -5.0,                // 最小标准差
        "std_max": 2.0                  // 最大标准差
    }
}
```

### 输入输出特征

```json
{
    "input_features": {
        "observation.image.front": {
            "type": "visual",
            "shape": [3, 128, 128]      // 前置摄像头图像
        },
        "observation.environment_state": {
            "type": "state", 
            "shape": [32]               // 环境状态向量
        }
    },
    "output_features": {
        "action": {
            "type": "action",
            "shape": [6]                // 6维动作空间
        }
    }
}
```

## 使用方法

### 1. 基本训练

```bash
cd /home/lab/RL
python lerobot/lerobot/scripts/rl/train_sequence_act_actor.py \
    --config lerobot/config/Isaac_lab_kuavo_env/train/only_on_line_learning/sequence_act_actor_warmup.json
```

### 2. 自定义配置

修改 `sequence_act_actor_warmup.json` 中的参数：

- `batch_size`: 批次大小（默认32）
- `steps`: 训练步数（默认10000）
- `dataset.repo_id`: 数据集ID
- `policy.act_chunk_size`: 动作序列长度
- `policy.obs_history_length`: 观测历史长度

### 3. 监控训练

训练过程会：
- 输出到控制台的进度条
- 记录到 wandb（如果启用）
- 保存检查点到 `outputs/train/sequence_act_actor_warmup/`

## 架构说明

### SequenceACTSACActorV2 架构

```
观测序列 → SACObservationEncoder → 观测编码 
                                    ↓
                                Transformer Encoder
                                    ↓
                                Transformer Decoder (自回归)
                                    ↓
                            动作序列预测 (chunk_size × action_dim)
                                    ↓
                            TanhMultivariateNormalDiag → 采样动作
```

### 关键组件

1. **SACObservationEncoder**: 处理图像和状态输入
2. **Transformer Encoder**: 编码观测序列
3. **Transformer Decoder**: 自回归生成动作序列
4. **位置编码**: 为序列添加位置信息
5. **动作预测头**: 输出动作均值和标准差

## 损失函数

### 主要损失
- **MSE Loss**: 预测动作与目标动作的均方误差
- **序列一致性损失**: 鼓励动作序列的平滑性

### 损失计算
```python
total_loss = mse_loss + 0.1 * sequence_consistency_loss
```

## 测试验证

运行测试脚本验证实现：

```bash
python test_sequence_act_warmup.py
```

测试包括：
- 导入测试
- 配置加载测试
- 编码器创建测试
- Actor创建测试
- 前向传播测试

## 输出文件

训练完成后会生成：

```
outputs/train/sequence_act_actor_warmup/
├── checkpoint_step_1000.pt      # 中间检查点
├── checkpoint_step_2000.pt
├── ...
└── final_sequence_act_actor.pt  # 最终模型
```

## 模型加载

预训练完成后，可以在 SAC 训练中使用：

```python
# 加载预训练的 Actor
checkpoint = torch.load("outputs/train/sequence_act_actor_warmup/final_sequence_act_actor.pt")
actor.load_state_dict(checkpoint['actor_state_dict'])
encoder.load_state_dict(checkpoint['encoder_state_dict'])
```

## 注意事项

1. **内存使用**: 序列模型需要更多内存，建议使用较小的批次大小
2. **训练时间**: 由于 Transformer 架构，训练时间比 MLP 更长
3. **数据格式**: 确保数据集包含正确的观测和动作格式
4. **设备配置**: 建议使用 GPU 进行训练

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 减小 `batch_size`
   - 减小 `act_chunk_size` 或 `obs_history_length`

2. **导入错误**
   - 确保在正确的环境中运行
   - 检查 lerobot 包是否正确安装

3. **配置错误**
   - 验证 JSON 配置文件格式
   - 检查特征类型和形状定义

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

1. **使用混合精度训练**:
   ```json
   "use_amp": true
   ```

2. **调整学习率**:
   ```json
   "actor_lr": 1e-4
   ```

3. **梯度裁剪**:
   ```json
   "grad_clip_norm": 10.0
   ```

## 扩展功能

### 自定义损失函数

可以修改 `compute_sequence_bc_loss` 函数来添加自定义损失：

```python
def compute_sequence_bc_loss(actor, observations, target_actions, ...):
    # 添加自定义损失项
    custom_loss = compute_custom_loss(...)
    total_loss = mse_loss + 0.1 * sequence_consistency_loss + custom_loss
    return total_loss, metrics
```

### 多模态输入

支持添加更多输入模态：

```json
{
    "input_features": {
        "observation.image.front": {...},
        "observation.image.top": {...},
        "observation.environment_state": {...},
        "observation.robot_state": {...}
    }
}
```

这个预训练脚本为 `SequenceACTSACActorV2` 提供了完整的 warm-up 训练流程，支持视觉编码器和序列动作预测，是 SAC 策略训练的重要组件。

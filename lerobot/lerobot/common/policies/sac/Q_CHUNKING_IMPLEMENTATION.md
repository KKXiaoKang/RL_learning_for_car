# Q-chunking Implementation for SAC

## 📄 概述

基于论文 [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969)，我们在SAC算法中实现了Q-chunking方法。Q-chunking是一种简单而有效的技术，通过动作序列预测和联合概率优化来改进强化学习算法在长期任务和稀疏奖励任务上的表现。

## 🎯 核心思想

### 传统SAC vs Q-chunking SAC

| 特性 | 传统SAC | Q-chunking SAC |
|------|---------|----------------|
| **动作预测** | 单步动作 a_t | 动作序列 a_t:t+k |
| **策略优化** | π(a_t\|s_t) | π(a_t:t+k\|s_t:t+k) |
| **TD学习** | 1-step TD | n-step TD backup |
| **探索策略** | 随机单步探索 | 时间一致的序列探索 |

### Q-chunking的关键优势

1. **时间一致性探索**: 通过预测动作序列，确保探索行为在时间上的一致性
2. **更稳定的TD学习**: 使用n-step backup进行更稳定和高效的价值函数学习
3. **利用时间结构**: 在"chunked"动作空间中运行RL，更好地利用动作的时间依赖关系

## 🏗️ 架构实现

### 1. Q-chunking Critic损失

```python
def compute_loss_critic(self, observations, actions, rewards, next_observations, done):
    """Q-chunking Critic损失计算"""
    
    if use_q_chunking:
        # 1. 获取下一状态的动作序列预测
        next_action_sequence, next_log_probs_joint, _ = self.actor(
            next_observations, return_sequence=True
        )
        
        # 2. 使用序列的第一个动作计算Q值
        next_first_action = next_action_sequence[:, 0, :]
        
        # 3. 使用联合对数概率进行熵正则化
        min_q = min_q - (self.temperature * next_log_probs_joint)
        
        # 4. 计算n-step TD target
        if chunk_size > 1:
            n_step_returns = self.actor.compute_n_step_returns(...)
            td_target = n_step_returns + (1 - done) * (γ^n) * min_q
        else:
            td_target = rewards + (1 - done) * γ * min_q
```

### 2. Q-chunking Actor损失

Q-chunking Actor实现了三种策略：

#### 标准策略 (Standard)
```python
def _compute_standard_q_chunking_loss(self, action_sequence, log_probs_joint, ...):
    """使用第一个动作计算Q值，但使用序列联合概率"""
    first_action = action_sequence[:, 0, :]
    q_preds = self.critic_forward(observations, first_action)
    min_q = q_preds.min(dim=0)[0]
    
    # 核心：使用整个动作序列的联合概率
    sac_loss = ((self.temperature * log_probs_joint) - min_q).mean()
```

#### 保守策略 (Conservative)
```python
def _compute_conservative_q_chunking_loss(self, action_sequence, log_probs_joint, ...):
    """对序列中的多个动作计算Q值并取最小值"""
    q_values_list = []
    for t in range(horizon):
        action_t = action_sequence[:, t, :]
        q_t = self.critic_forward(observations, action_t).min(dim=0)[0]
        q_values_list.append(q_t)
    
    # 保守估计：取最小Q值
    conservative_q = torch.stack(q_values_list, dim=1).min(dim=1)[0]
    sac_loss = ((self.temperature * log_probs_joint) - conservative_q).mean()
```

#### 时间加权策略 (Temporal Weighted)
```python
def _compute_temporal_weighted_q_chunking_loss(self, action_sequence, log_probs_joint, ...):
    """对不同时间步的动作给予不同权重"""
    weights = torch.tensor([decay_factor ** t for t in range(horizon)])
    weights = weights / weights.sum()
    
    weighted_q_sum = 0.0
    for t in range(horizon):
        action_t = action_sequence[:, t, :]
        q_t = self.critic_forward(observations, action_t).min(dim=0)[0]
        weighted_q_sum += weights[t] * q_t
    
    sac_loss = ((self.temperature * log_probs_joint) - weighted_q_sum).mean()
```

### 3. Q-chunking温度损失

```python
def compute_loss_temperature(self, observations, ...):
    """Q-chunking温度损失，考虑序列复杂性"""
    
    if use_q_chunking:
        # 使用动作序列的联合概率
        _, log_probs_joint, _ = self.actor(observations, return_sequence=True)
        
        # 调整目标熵以适应序列长度
        chunk_size = self.actor.chunk_size
        adjusted_target_entropy = self._get_adjusted_target_entropy(chunk_size)
        
        temperature_loss = (-self.log_alpha.exp() * 
                           (log_probs_joint + adjusted_target_entropy)).mean()
```

## ⚙️ 配置参数

### Q-chunking配置项

```python
@dataclass
class SACConfig:
    # Q-chunking基础配置
    enable_q_chunking: bool = True                    # 是否启用Q-chunking
    q_chunking_strategy: str = "standard"             # Q-chunking策略
    q_chunking_horizon: int = 3                       # Q-chunking时间视野
    q_chunking_decay: float = 0.9                     # 时间衰减因子
    q_chunking_entropy_scaling: str = "linear"        # 熵缩放策略
```

### 策略选择指南

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **Standard** | 一般任务 | 计算效率高，论文方法 | 可能不够保守 |
| **Conservative** | 高风险任务 | 更稳定的Q值估计 | 计算开销大，学习慢 |
| **Temporal Weighted** | 需要平衡的任务 | 平衡近期和远期动作 | 参数调优复杂 |

## 🔧 使用方法

### 1. 配置文件设置

```json
{
    "use_act_actor": true,
    "use_sequence_act_actor": true,
    "enable_q_chunking": true,
    "q_chunking_strategy": "standard",
    "act_chunk_size": 8,
    "obs_history_length": 5,
    "q_chunking_horizon": 3,
    "q_chunking_decay": 0.9,
    "q_chunking_entropy_scaling": "linear"
}
```

### 2. 训练脚本

```python
# Q-chunking会自动启用，无需额外代码修改
python lerobot/scripts/rl/learner.py \
    --config-path="config/sac_q_chunking.json" \
    --env-name="YourEnv-v0"
```

### 3. 推理使用

```python
# 推理时自动使用Q-chunking策略
policy = SACPolicy.from_pretrained("path/to/model")
action = policy.select_action(observation)  # 自动返回第一个动作
```

## 📊 实验结果和期望改进

### 理论优势

1. **更好的探索**: 时间一致的动作序列探索
2. **更稳定的学习**: n-step TD backup减少方差
3. **更好的样本效率**: 利用动作序列的时间结构

### 预期性能提升

- **长期任务**: 20-40%样本效率提升
- **稀疏奖励**: 显著改善探索效果
- **操作任务**: 更平滑的动作执行

## 🔍 调试和监控

### 关键监控指标

```python
# 在训练日志中监控以下指标
- "q_chunking/joint_log_prob": 动作序列联合概率
- "q_chunking/chunk_size": 实际使用的序列长度
- "q_chunking/n_step_returns": n-step return值
- "q_chunking/adjusted_entropy": 调整后的目标熵
```

### 常见问题排查

1. **Q-chunking未启用**: 检查是否同时设置了`use_sequence_act_actor=True`
2. **损失爆炸**: 尝试降低`q_chunking_horizon`或调整熵缩放策略
3. **学习慢**: 考虑使用"conservative"策略或调整`q_chunking_decay`

## 🚀 最佳实践

1. **初始配置**: 从`q_chunking_strategy="standard"`开始
2. **序列长度**: `act_chunk_size=8`通常是个好起点
3. **时间视野**: `q_chunking_horizon=3`平衡计算和性能
4. **熵缩放**: `linear`缩放适用于大多数任务

## 📚 参考资料

- [Q-chunking论文](https://arxiv.org/abs/2507.07969)
- [SAC原始论文](https://arxiv.org/abs/1801.01290)
- [ACT论文](https://arxiv.org/abs/2304.13705)

---

*该实现基于Q-chunking论文的核心思想，结合了SAC算法的特点，为长期任务和稀疏奖励环境提供了更好的解决方案。*

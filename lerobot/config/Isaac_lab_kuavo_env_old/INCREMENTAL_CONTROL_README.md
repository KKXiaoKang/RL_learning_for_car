# Kuavo机器人增量控制系统说明

## 概述

本文档介绍了Kuavo机器人环境中新实现的增量控制系统，该系统将原有的绝对位置控制改为基于固定基准位置的增量控制，以提高模型学习精细操作动作的能力。

## 背景

在之前的训练中，发现robot可以学习到如何靠近箱子和手部靠近箱子，但一直学不会抬高箱子。action输出末端EEF pose时一直都是很随机的值，看起来几乎没学习到太多后面的动作。

## 新的Action Space设计

### 原有设计 (绝对位置控制)
```python
action[0] - linear cmd_vel x (机器人线性速度 x)
action[1] - linear cmd_vel y (机器人线性速度 y) 
action[2] - linear cmd_vel z (机器人线性速度 z)
action[3] - angular cmd_vel yaw (机器人角速度 yaw)
action[4] - 左手末端绝对位置 x
action[5] - 左手末端绝对位置 y
action[6] - 左手末端绝对位置 z
action[7] - 右手末端绝对位置 x
action[8] - 右手末端绝对位置 y
action[9] - 右手末端绝对位置 z
```

### 新设计 (增量控制)
```python
action[0] - linear cmd_vel x (机器人线性速度 x)
action[1] - linear cmd_vel y (机器人线性速度 y)
action[2] - linear cmd_vel z (机器人线性速度 z, 已禁用)
action[3] - angular cmd_vel yaw (机器人角速度 yaw)
action[4] - 左手末端相对于基准位置的增量 dx
action[5] - 左手末端相对于基准位置的增量 dy  
action[6] - 左手末端相对于基准位置的增量 dz
action[7] - 右手末端相对于基准位置的增量 dx
action[8] - 右手末端相对于基准位置的增量 dy
action[9] - 右手末端相对于基准位置的增量 dz
```

## 固定基准位置

系统使用以下固定基准位置：

```python
# 左手基准位置 (基于base_link坐标系)
FIXED_LEFT_POS = [0.3178026345146559, 0.4004180715613648, -0.019417275957965042]
FIXED_LEFT_QUAT = [0.0, -0.70711, 0.0, 0.70711]

# 右手基准位置 (基于base_link坐标系)  
FIXED_RIGHT_POS = [0.3178026345146559, -0.4004180715613648, -0.019417275957965042]
FIXED_RIGHT_QUAT = [0.0, -0.70711, 0.0, 0.70711]
```

实际末端位置 = 基准位置 + 累积增量

## 增量控制约束

### 基本约束
- `MAX_INCREMENT_RANGE = 0.2` : 最大累积增量范围 (±20cm)
- `MAX_INCREMENT_PER_STEP = 0.02` : 每步最大增量变化 (2cm)

### 位置约束 (转换为绝对位置后应用)
1. **X位置**: [0, 0.7] (前方运动限制)
2. **左手Y位置**: [0, 0.65] (正Y侧)
3. **右手Y位置**: [-0.65, 0] (负Y侧)  
4. **Z位置**: [-0.20, 0.65] (安全范围)
5. **机器人线性Z运动**: 禁用 (action[2] = 0)

### 平滑性约束
- **速度平滑性**: 单步增量限制确保手部运动不会突然跳跃
- **位置平滑性**: 指数移动平均平滑处理
- **累积范围限制**: 防止手部偏离基准位置过远

## 平滑处理机制

系统实现了多层平滑处理：

1. **输入平滑**: 指数移动平均 `arm_smoothing_factor = 0.4`
2. **增量限制**: 每步最大变化量约束
3. **累积约束**: 总增量范围限制
4. **任务约束**: 基于任务需求的位置约束

## VR干预支持

VR系统也支持增量控制：

- VR发送绝对位置数据
- 系统自动转换为相对于基准位置的增量
- 应用相同的约束和平滑处理
- 确保VR干预和策略学习的一致性

## 实现细节

### 核心方法
- `_process_incremental_action()`: 处理增量约束和平滑
- `_apply_task_specific_constraints()`: 应用任务特定约束
- `_publish_action_based_arm_poses()`: 转换增量为绝对位置并发布

### 状态跟踪
- `current_left_increment`: 当前左手累积增量
- `current_right_increment`: 当前右手累积增量  
- `last_left_increment`: 上一步左手增量
- `last_right_increment`: 上一步右手增量

### 重置机制
每个episode开始时，所有增量状态重置为零，确保从基准位置开始。

## 优势

1. **更易学习**: 增量控制让模型更容易学习精细的操作动作
2. **位置稳定**: 基于固定基准位置，避免累积误差
3. **平滑运动**: 多层约束确保手部运动的平滑性
4. **安全保障**: 约束机制防止不安全的运动
5. **VR兼容**: 支持VR干预的无缝集成

## 调试信息

启用debug模式后，系统会输出详细的增量控制信息：

```
[INCREMENTAL DEBUG] Left increment: [dx, dy, dz], Right increment: [dx, dy, dz]
[INCREMENTAL DEBUG] Left absolute pos: [x, y, z], Right absolute pos: [x, y, z]  
[INCREMENTAL DEBUG] Current cumulative - Left: [dx, dy, dz], Right: [dx, dy, dz]
```

## 使用建议

1. **训练初期**: 模型会从基准位置开始学习小幅增量动作
2. **干预策略**: VR干预时注意增量的合理性，避免过大跳跃
3. **参数调整**: 可根据任务需求调整 `MAX_INCREMENT_RANGE` 和 `MAX_INCREMENT_PER_STEP`
4. **监控训练**: 观察累积增量的变化趋势，确保学习进展

## 配置文件

使用现有的训练配置文件，无需额外修改。系统会自动识别并应用增量控制。

## 故障排除

1. **手部不动**: 检查增量是否被约束限制
2. **运动突兀**: 调整平滑参数 `arm_smoothing_factor`
3. **范围限制**: 检查累积增量是否超出 `MAX_INCREMENT_RANGE`
4. **VR干预异常**: 确认VR发送的位置数据格式正确 
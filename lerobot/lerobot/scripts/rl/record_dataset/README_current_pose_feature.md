# Robotic Bézier Tool - Current Robot Pose Integration Feature

## 概述

这个新功能允许 `robotic_bezier_action_record_tool.py` 自动获取机器人的当前末端执行器位置，并将其作为轨迹的初始点，而不是使用 `key_point.json` 文件中预定义的固定初始位置。

## 主要特性

1. **自动获取当前机器人位置**: 通过订阅 ROS 话题 `/fk/base_link_eef_left` 和 `/fk/base_link_eef_right` 来获取机器人的实时末端执行器位置。

2. **动态更新初始关键点**: 自动将 `key_point.json` 中 `frame_id: 0` 的关键点更新为机器人的当前位置。

3. **向后兼容**: 默认情况下禁用此功能，保持与现有工作流程的兼容性。

4. **调试支持**: 提供详细的调试信息，显示原始位置和更新后的位置。

## 使用方法

### 基本命令

使用当前机器人位置作为初始点：

```bash
# 生成并发布轨迹动作
python3 robotic_bezier_action_record_tool.py --mode actions --use-current-pose --debug

# 播放轨迹（使用当前位置作为起点）
python3 robotic_bezier_action_record_tool.py --mode play_actions --use-current-pose --rate 10.0 --debug

# 可视化轨迹（使用当前位置作为起点）
python3 robotic_bezier_action_record_tool.py --mode visualize --use-current-pose --debug
```

### 命令行参数

- `--use-current-pose`: 启用当前机器人位置作为初始关键点
- `--pose-timeout SECONDS`: 设置等待机器人位置数据的超时时间（默认5秒）
- `--debug`: 启用调试输出，显示位置更新详情

### 完整示例

```bash
# 循环播放轨迹，使用当前机器人位置，播放速率为5Hz，启用调试
python3 robotic_bezier_action_record_tool.py \
    --mode play_actions \
    --use-current-pose \
    --rate 5.0 \
    --loop \
    --debug \
    --pose-timeout 10.0
```

## 工作原理

### 1. ROS 话题订阅

工具会订阅以下 ROS 话题来获取机器人当前位置：
- `/fk/base_link_eef_left`: 左手末端执行器在 base_link 坐标系中的位置
- `/fk/base_link_eef_right`: 右手末端执行器在 base_link 坐标系中的位置

### 2. 初始位置更新

当启用 `--use-current-pose` 选项时：
1. 工具等待接收两个末端执行器的位置数据
2. 找到 `key_point.json` 中 `frame_id: 0` 的关键点
3. 将该关键点的位置和四元数更新为当前机器人位置
4. 使用更新后的关键点生成贝塞尔轨迹

### 3. 调试信息

启用 `--debug` 选项时，工具会显示：
- 接收到的机器人位置数据
- 原始关键点位置
- 更新后的关键点位置
- 轨迹生成状态

## 先决条件

### ROS 话题要求

确保以下 ROS 话题正在发布：
```bash
# 检查话题是否可用
rostopic list | grep "/fk/base_link_eef"

# 查看话题数据
rostopic echo /fk/base_link_eef_left
rostopic echo /fk/base_link_eef_right
```

### key_point.json 要求

确保 `key_point.json` 文件包含 `frame_id: 0` 的初始关键点：

```json
{
    "key_points": {
        "keyframes": [
            {
                "frame_id": 0,
                "left_hand": {
                    "position": [0.3178026345146559, 0.4004180715613648, -0.019417275957965042],
                    "quaternion": [0.0, -0.70711, 0.0, 0.70711]
                },
                "right_hand": {
                    "position": [0.3178026345146559, -0.4004180715613648, -0.019417275957965042],
                    "quaternion": [0.0, -0.70711, 0.0, 0.70711]
                }
            }
            // ... 其他关键点
        ]
    }
}
```

## 故障排除

### 常见问题

1. **超时错误**: "Failed to receive robot poses within Xs timeout"
   - 检查 ROS 话题是否正在发布
   - 增加超时时间：`--pose-timeout 15.0`
   - 确认话题名称正确

2. **没有找到 frame_id: 0**: "No keyframe with frame_id: 0 found"
   - 检查 `key_point.json` 文件格式
   - 确保至少有一个关键点的 `frame_id` 为 0

3. **ROS 连接失败**: "Failed to setup ROS"
   - 确认 ROS 环境已正确设置
   - 检查 `roscore` 是否运行
   - 验证必要的 ROS 包已安装

### 调试步骤

1. 启用调试模式：`--debug`
2. 检查 ROS 话题：
   ```bash
   rostopic hz /fk/base_link_eef_left
   rostopic hz /fk/base_link_eef_right
   ```
3. 验证话题数据格式：
   ```bash
   rostopic echo -n 1 /fk/base_link_eef_left
   ```

## 集成到现有工作流程

### 与 RL 环境集成

这个功能特别适用于强化学习环境，其中需要：
1. 从机器人的当前状态开始轨迹
2. 生成相对于当前位置的动作序列
3. 实现连续的轨迹执行

### 与录制工具集成

可以与其他录制工具配合使用：
```bash
# 先获取当前位置并生成轨迹
python3 robotic_bezier_action_record_tool.py --mode actions --use-current-pose

# 然后在其他录制流程中使用生成的动作
```

## 版本兼容性

- **向后兼容**: 默认行为保持不变
- **新功能**: 通过命令行参数选择性启用
- **ROS 依赖**: 需要 ROS 环境和相关消息包

## 开发说明

### 代码结构

新功能的主要组件：
- `_left_eef_pose_callback()`: 左手位置回调
- `_right_eef_pose_callback()`: 右手位置回调  
- `get_current_robot_poses()`: 获取当前位置的主方法
- `_load_key_points()`: 增强的关键点加载方法

### 线程安全

使用了线程锁来确保 ROS 回调和主线程之间的数据安全访问。

### 测试

运行测试脚本验证功能：
```bash
python3 test_current_pose_integration.py
```

# Robot System Module

这个模块提供了一个完整的机器人系统抽象框架，用于处理机器人数据集和可视化。

## 模块结构

```
robot_system/
├── __init__.py              # 模块初始化和导出
├── robotic.py              # 主要的Robotic类
├── camera.py               # 相机系统类
├── joint.py                # 关节系统类
├── models.py               # 模型模式和配置
├── dataset_wrapper.py      # 数据集包装器
└── README.md              # 本文档
```

## 核心组件

### 1. Robotic类 (robotic.py)

主要的机器人抽象类，整合了所有子系统：

- **相机系统**: 管理RGB和深度相机
- **关节系统**: 管理机器人关节和运动学
- **模型配置**: 设置固定基座或浮动基座模式
- **躯干配置**: 设置base_link在torso_position的位置
- **数据集集成**: 与数据加载系统集成

```python
from lerobot.common.robot_system import create_kuavo_robot

# 创建Kuavo机器人
robot = create_kuavo_robot()

# 设置关节位置
robot.set_dual_arm_positions([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7,  # 左臂
                             -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7])  # 右臂
```

### 2. 相机系统 (camera.py)

#### Camera (抽象基类)
- `RGBCamera`: RGB相机实现
- `DepthCamera`: 深度相机实现
- `CameraSystem`: 多相机管理器

```python
# 添加相机
robot.add_camera('front', camera_type='RGB')
robot.add_camera('depth_sensor', camera_type='DEPTH', depth_scale=1000.0)

# 配置相机内参
from lerobot.common.robot_system import CameraIntrinsics
intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
robot.get_camera('front').update_intrinsics(intrinsics)
```

#### 相机功能
- 图像提取和处理
- 3D点云生成（深度相机）
- 3D到2D投影
- 相机姿态管理

### 3. 关节系统 (joint.py)

#### Joint类
- 支持不同关节类型：`REVOLUTE`, `PRISMATIC`, `CONTINUOUS`, `FIXED`
- 关节限制和轴配置
- 前向运动学计算

#### JointCollection和ArmJointSystem
- 关节链和组管理
- 双臂机器人专用功能

```python
# 设置双臂关节
left_joints = [f'zarm_l{i}_joint' for i in range(1, 8)]
right_joints = [f'zarm_r{i}_joint' for i in range(1, 8)]
robot.setup_dual_arm_joints(left_joints, right_joints)

# 计算前向运动学
left_fk = robot.compute_forward_kinematics('left')
```

### 4. 模型配置 (models.py)

#### ModelMode
- `FIXED_BASE`: 固定基座模式
- `FLOATING_BASE`: 浮动基座模式

#### TorsoConfig
- 基座链接位置配置
- 躯干位置和姿态
- 世界坐标系变换

```python
from lerobot.common.robot_system import ModelMode, TorsoConfig

config = TorsoConfig(
    base_link_position=(0.0, 0.0, 0.0),
    torso_position=(0.0, 0.0, 0.4),
    model_mode=ModelMode.FLOATING_BASE
)
```

### 5. 数据集包装器 (dataset_wrapper.py)

#### DatasetWrapper (虚拟基类)
定义数据提取接口：
- `extract_robot_joint_positions()`: 提取机器人关节位置
- `extract_camera_images()`: 提取相机图片
- `extract_robot_world_position()`: 提取机器人世界系位置
- `extract_target_world_position()`: 提取目标物体世界系位置

#### LeRobotDatasetWrapper
LeRobotDataset的具体实现：

```python
# 创建数据集包装器
wrapper = robot.create_lerobot_dataset_wrapper(
    repo_id="your-dataset-repo-id",
    arm_joint_indices=(6, 20),  # 臂关节在状态向量中的索引
    robot_base_position_indices=(0, 3),  # 机器人位置索引
    target_position_key="target_position"
)

# 加载数据
dataset = wrapper.load_dataset()
episode_data = robot.get_episode_data(0)
```

## 使用示例

### 基本使用

```python
from lerobot.common.robot_system import create_kuavo_robot

# 1. 创建机器人
robot = create_kuavo_robot()

# 2. 配置数据集
robot.create_lerobot_dataset_wrapper(
    repo_id="your-dataset",
    arm_joint_indices=(6, 20)
)

# 3. 加载数据并处理
data_item = {...}  # 从数据集获取
robot.load_state_from_data(data_item)
images = robot.extract_camera_images(data_item)

# 4. 计算运动学
left_transforms = robot.compute_forward_kinematics('left')
```

### 自定义机器人配置

```python
from lerobot.common.robot_system import Robotic, ModelMode, TorsoConfig

# 创建自定义躯干配置
torso_config = TorsoConfig(
    base_link_position=(0.0, 0.0, 0.0),
    torso_position=(0.0, 0.0, 0.45),
    torso_orientation=(0.0, np.radians(5), 0.0),
    model_mode=ModelMode.FLOATING_BASE
)

# 创建机器人
robot = Robotic(
    name="CustomRobot",
    torso_config=torso_config,
    urdf_path="/path/to/robot.urdf"
)

# 添加关节和相机
robot.setup_dual_arm_joints(left_joints, right_joints)
robot.add_camera('high_res_front', camera_type='RGB')
```

## 工厂函数

模块提供了便捷的工厂函数：

- `create_dual_arm_robot()`: 创建标准双臂机器人
- `create_kuavo_robot()`: 创建Kuavo机器人配置

## 数据流

1. **数据加载**: DatasetWrapper从数据集提取数据
2. **状态更新**: Robotic类加载机器人状态
3. **运动学计算**: Joint系统计算前向运动学
4. **图像处理**: Camera系统处理图像数据
5. **可视化**: 与可视化系统集成

## 与现有可视化系统集成

这个模块可以与现有的`visualize_dataset_robotics.py`集成：

```python
# 在可视化脚本中使用
robot = create_kuavo_robot()
robot.create_lerobot_dataset_wrapper(repo_id)

for data_item in dataset:
    robot.load_state_from_data(data_item)
    joint_positions = robot.get_dual_arm_positions()
    # 传递给可视化函数
    log_robot_visualization(joint_positions, frame_index)
```

## 扩展性

模块设计为高度可扩展：

1. **新的相机类型**: 继承Camera基类
2. **新的关节类型**: 扩展JointType枚举
3. **新的数据集格式**: 实现DatasetWrapper接口
4. **新的机器人配置**: 创建新的工厂函数

## 完整示例

查看 `examples/robot_system_usage_example.py` 获取完整的使用示例，包括：

- 基本机器人创建
- Kuavo机器人配置
- 数据集集成
- 高级配置
- 数据处理工作流

## 依赖

- numpy
- torch
- lerobot (现有的LeRobotDataset)
- typing (Python标准库)
- pathlib (Python标准库)
- enum (Python标准库)
- dataclasses (Python标准库)

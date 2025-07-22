# ResNet10 特征可视化功能

这个功能允许你可视化 ResNet10 编码器在处理来自 RLKuavoGymEnv 环境的图像时提取的特征。

## 功能概述

当启用特征可视化时，系统会：
1. 捕获经过 ResNet10 编码器处理的图像特征
2. 将特征转换为可视化的热力图
3. 通过 ROS topic `/vision_features/resnet10_features` 发布可视化结果

## 使用方法

### 1. 启用特征可视化

在创建环境时，设置 `enable_feature_visualization=True`：

```python
from lerobot.gym_hil.envs.rl_kuavo_gym_env import RLKuavoGymEnv

env = RLKuavoGymEnv(
    debug=True,
    enable_feature_visualization=True,  # 启用特征可视化
    wbc_observation_enabled=True
)
```

### 2. 配置 SAC 策略

创建 SAC 配置时，确保启用特征可视化：

```python
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy

config = SACConfig(
    vision_encoder_name="helper2424/resnet10",  # 使用 ResNet10 编码器
    enable_feature_visualization=True,         # 启用特征可视化
    # ... 其他配置
)

policy = SACPolicy(config=config)
```

### 3. 运行测试脚本

使用提供的测试脚本：

```bash
cd lerobot/gym_hil
python test_feature_visualization.py
```

### 4. 查看可视化结果

在另一个终端中，使用以下任一方法查看特征可视化：

**方法 1: 使用 image_view**
```bash
rosrun image_view image_view image:=/vision_features/resnet10_features
```

**方法 2: 使用 rqt_image_view**
```bash
rqt_image_view
```
然后在界面中订阅 `/vision_features/resnet10_features` topic。

**方法 3: 使用 rviz**
1. 启动 rviz: `rviz`
2. 添加 Image display
3. 设置 Image Topic 为 `/vision_features/resnet10_features`

## 技术细节

### 特征处理流程

1. **图像输入**: 环境提供的图像 (480x640) 被 resize 到 224x224
2. **特征提取**: 图像通过 ResNet10 编码器，输出特征张量
3. **特征可视化**: 
   - 对于空间特征 (B, C, H, W): 计算通道平均值创建热力图
   - 对于平坦特征 (B, feature_dim): 重塑为网格形式显示
4. **归一化**: 特征值归一化到 0-255 范围
5. **发布**: 通过 ROS Image 消息发布

### ROS Topic 信息

- **Topic 名称**: `/vision_features/resnet10_features`
- **消息类型**: `sensor_msgs/Image`
- **编码格式**: `rgb8`
- **图像尺寸**: 224x224 pixels
- **更新频率**: 跟随环境步进频率

### 支持的特征形状

- **4D 张量 (B, C, H, W)**: 空间特征映射，显示为平均通道热力图
- **2D 张量 (B, feature_dim)**: 平坦特征向量，显示为网格形式

## 配置选项

### SACConfig 选项

```python
SACConfig(
    vision_encoder_name="helper2424/resnet10",  # 必须使用 ResNet10
    enable_feature_visualization=True,          # 启用特征可视化
    freeze_vision_encoder=True,                 # 推荐冻结编码器以获得一致的特征
    shared_encoder=True,                        # 在 actor 和 critic 间共享编码器
    # ... 其他配置
)
```

### RLKuavoGymEnv 选项

```python
RLKuavoGymEnv(
    enable_feature_visualization=True,  # 启用特征可视化
    debug=True,                         # 启用调试信息
    image_size=(224, 224),              # 确保图像尺寸匹配
    # ... 其他配置
)
```

## 故障排除

### 常见问题

1. **没有图像显示**
   - 检查 ROS 节点是否正确初始化
   - 确认 topic 名称是否正确: `/vision_features/resnet10_features`
   - 验证 `enable_feature_visualization=True` 是否在 config 和环境中都设置了

2. **特征图像全黑或全白**
   - 这可能表明特征值范围异常
   - 检查控制台输出的特征值范围信息
   - 确认输入图像格式正确 (RGB, [0,1] 范围)

3. **性能问题**
   - 特征可视化会增加计算开销
   - 可以通过调整发布频率来优化性能
   - 在生产环境中禁用特征可视化

### 调试信息

启用 debug 模式时，系统会输出：
- 特征张量的形状信息
- 特征值的数值范围
- 发布状态和频率信息

## 示例输出

成功运行时，你会看到类似的输出：

```
Feature visualization enabled for RLKuavoGymEnv
Features will be published to /vision_features/resnet10_features when using SAC policy with ResNet10 encoder
Published feature visualization - shape: torch.Size([1, 512, 7, 7]), feature range: [-2.341, 4.782], viz range: [0, 255]
```

可视化图像显示为热力图，其中：
- 亮区域表示高激活值的特征
- 暗区域表示低激活值的特征
- 颜色强度反映特征的重要程度

这个可视化有助于理解 ResNet10 编码器如何"看待"和处理输入图像，对于调试和理解模型行为非常有用。 
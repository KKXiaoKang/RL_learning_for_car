# Web Visualization for LeRobot Dataset

这个脚本提供了基于网页的数据集可视化功能，允许你通过浏览器查看 LeRobot 数据集的各个 episode。

## 主要特性

- **默认网页模式**: 脚本默认使用网页模式，无需额外参数
- **交互式界面**: 使用 Rerun 的网页查看器提供丰富的交互功能
- **多端口支持**: 可以自定义网页端口和 WebSocket 端口
- **多种模式**: 支持本地、远程和网页三种可视化模式

## 快速开始

### 基本用法（网页模式）

```bash
python lerobot/lerobot/scripts/visualize_dataset_web_server.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

然后在浏览器中打开: http://localhost:9095

### 自定义端口

```bash
python lerobot/lerobot/scripts/visualize_dataset_web_server.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --web-port 8080
```

然后在浏览器中打开: http://localhost:8080

## 可视化模式

### 1. Web 模式（默认）
- 启动网页服务器
- 通过浏览器访问可视化界面
- 支持实时数据流和交互

```bash
python lerobot/lerobot/scripts/visualize_dataset_web_server.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode web
```

### 2. Local 模式
- 启动本地查看器
- 适合本地开发和调试

```bash
python lerobot/lerobot/scripts/visualize_dataset_web_server.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode local
```

### 3. Distant 模式
- 适合远程服务器部署
- 支持 WebSocket 连接

```bash
python lerobot/lerobot/scripts/visualize_dataset_web_server.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087
```

## 参数说明

- `--repo-id`: 数据集仓库 ID（必需）
- `--episode-index`: 要可视化的 episode 索引（必需）
- `--mode`: 可视化模式 (`web`, `local`, `distant`)，默认为 `web`
- `--web-port`: 网页服务器端口，默认为 9095
- `--ws-port`: WebSocket 端口，默认为 9087
- `--batch-size`: 数据加载批次大小，默认为 32
- `--num-workers`: 数据加载进程数，默认为 4

## 测试

运行测试脚本来验证功能：

```bash
python test_web_visualization.py
```

## 注意事项

1. 确保你的浏览器支持 WebGL
2. 网页模式需要网络连接（即使是本地连接）
3. 如果端口被占用，可以指定其他端口
4. 使用 Ctrl+C 停止服务器

## 故障排除

### 端口被占用
```bash
# 使用其他端口
python lerobot/lerobot/scripts/visualize_dataset_web_server.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --web-port 9096
```

### 浏览器无法访问
- 检查防火墙设置
- 确认端口没有被其他程序占用
- 尝试使用 `localhost` 而不是 `127.0.0.1`

### 数据加载缓慢
- 减少 `--batch-size` 参数
- 减少 `--num-workers` 参数
- 检查网络连接（如果从远程加载数据）

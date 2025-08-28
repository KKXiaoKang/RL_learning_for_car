```bash
# av1 格式编码测试
python mp4_frame_publisher_simple.py --video episode_000003.mp4 --info

# MP4 文件使用了 AV1 编码，而你的系统无法解码这种格式。让我们尝试使用 ffmpeg 将其转换为更兼容的格式
ffmpeg -i episode_000003.mp4 -c:v libx264 -c:a copy episode_000003_h264.mp4

# 成功转换了视频格式。现在让我们测试新的 H.264 格式视频
python mp4_frame_publisher_simple.py --video episode_000003_h264.mp4 --info

# 测试提取一帧并发布到 ROS 话题：
python mp4_frame_publisher_simple.py --video episode_000003_h264.mp4 --frame 40

# 保存帧到本地文件
python mp4_frame_publisher_simple.py --video episode_000003_h264.mp4 --frame 40 --save --output test_frame_40.jpg

# 1. 显示视频信息
python mp4_frame_publisher_simple.py --video episode_000003_h264.mp4 --info

# 2. 提取并发布第40帧到ROS话题
python mp4_frame_publisher_simple.py --video episode_000003_h264.mp4 --frame 40
## 以20hz频率连续进行发布
python mp4_frame_publisher_simple.py --video episode_000003_h264.mp4 --frame 40 --continuous --frequency 20.0

# 3. 提取第40帧并保存为图片
python mp4_frame_publisher_simple.py --video episode_000003_h264.mp4 --frame 40 --save --output frame_40.jpg
```
# lerobot hil-serl快速启动教程
## 前置处理
### 运行mujoco env环境测试
```bash
python3 lerobot/scripts/rl/gym_manipulator.py --config_path config/Gym_mujoco_env/gym_collect/gym_hil_env_xbox_null.json
```

### 录制专家数据
```bash
python3 lerobot/scripts/rl/gym_manipulator.py --config_path config/Gym_mujoco_env/gym_collect/gym_hil_env_xbox_record.json
```

## 开始训练
### 开启actor网络
```bash
python3 lerobot/scripts/rl/actor.py --config_path config/Gym_mujoco_env/train/train_gym_hil_env_xbox.json
``` 

### 开启Critic网络，同时打开mujoco环境和加载wandb
```bash
python3 lerobot/scripts/rl/learner.py --config_path config/Gym_mujoco_env/train/train_gym_hil_env_xbox.json 
```

## 验证/推理网络
* 注意json当中的`pretrained_policy_name_or_path`和`mode`字段
```bash
python3 lerobot/scripts/rl/gym_manipulator.py --config_path config/Gym_mujoco_env/eval/gym_hil_env_xbox_eval.json
```
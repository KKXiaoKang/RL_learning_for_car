# SAC 初始化env提交
* ![环境演示](./IMG/image.png)

## 启动isaac lab环境
```bash
roslaunch rl_sac_env_isaac_lab lab_control_bt2pro.launch training_mode:=true
```

## 训练
```bash
python3 scripts/z_model_train.py
```

## 验证
```bash
python3 scripts/z_model_eval.py --model_path ./logs/sac_kuavo_navigation/run_20250619_115642/checkpoints/model_ep1500.pth 
```

## 转onnx
```bash
python3 scripts/pth2onnx.py --model_path logs/sac_kuavo_navigation/run_20250619_115642/checkpoints/model_ep1500.pth --output_path logs/sac_kuavo_navigation/run_20250619_115642/onnx/model_ep1500.onnx
```

# lerobo hil-serl
* ![lerobo-rlpd](./IMG/lerobo-rlpd.jpg)
# lerobot hil-serl快速启动教程
## 前置处理
### 运行mujoco env环境测试
```bash
python3 lerobot/scripts/rl/gym_manipulator.py --config_path config/gym_collect/gym_hil_env_xbox_null.json
```

### 录制专家数据
```bash
python3 lerobot/scripts/rl/gym_manipulator.py --config_path config/gym_collect/gym_hil_env_xbox_record.json
```

## 开始训练
### 开启actor网络
```bash
python3 lerobot/scripts/rl/actor.py --config_path config/train/train_gym_hil_env_xbox.json
``` 

### 开启Critic网络，同时打开mujoco环境和加载wandb
```bash
python3 lerobot/scripts/rl/learner.py --config_path config/train/train_gym_hil_env_xbox.json 
```

## 验证/推理网络
* 注意json当中的`pretrained_policy_name_or_path`和`mode`字段
```bash
python3 lerobot/scripts/rl/gym_manipulator.py --config_path config/eval/gym_hil_env_xbox_eval.json
```
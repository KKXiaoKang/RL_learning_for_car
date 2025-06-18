import json
import os
from algo.sac import SAC, ROSReplayBuffer

def load_sac_config():
    config_path = os.path.join(
        os.path.dirname(__file__), 
        'config/sac.json'  # 使用相对路径
    )
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # 类型转换确保数值类型正确
        config["learning_rate"] = float(config["learning_rate"])
        config["tau"] = float(config["tau"])
        config["gamma"] = float(config["gamma"])
        
        return config
        
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found at {config_path}")
    except json.JSONDecodeError:
        raise RuntimeError("Invalid JSON format in config file")

def main():
    # 加载配置
    config = load_sac_config()
    
    # 修改关键训练参数
    config.update({
        "learning_rate": 1e-4,  # 降低学习率
        "batch_size": 128,      # 减小batch size
        "learning_starts": 1000, # 增加预热步数
        "train_freq": 1,        # 每步都训练
        "gradient_steps": 1,    # 每次训练1步
        "buffer_size": 100000,  # 减小缓冲区大小
        "tau": 0.005,          # 目标网络更新率
        "gamma": 0.99,         # 折扣因子
    })
    
    # 创建SAC实例（启用ROS缓冲区）
    agent = SAC(
        policy="MlpPolicy",
        env="KuavoNavigation-v0",
        device="cuda",  # 强制使用GPU
        use_ros_buffer=True,  # 启用ROS数据收集
        tensorboard_log="./logs/sac_kuavo_navigation",  # 添加tensorboard日志
        **config
    )
    
    try:
        # 开始训练
        agent.learn(
            total_timesteps=1_000_000,
            log_interval=100
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        agent.save("sac_kuavo_navigation")

if __name__ == "__main__":
    main() 
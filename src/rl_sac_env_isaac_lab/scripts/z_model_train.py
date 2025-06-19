import json
import os
import datetime
from algo.sac import SAC

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
    
    # 创建基础日志目录
    base_log_dir = "./logs/sac_kuavo_navigation"
    os.makedirs(base_log_dir, exist_ok=True)
    
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
        use_ros_buffer=False,  # 禁用ROS数据收集
        tensorboard_log=base_log_dir,  # 使用基础日志目录
        **config
    )
    
    try:
        # 开始训练
        agent.learn(
            total_timesteps=1_000_000,
            log_interval=100,
            max_episode_steps=1000  # 设置每个episode的最大步长
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # 保存模型时也使用时间戳
        if agent.writer is not None:
            save_path = os.path.join(agent.writer.log_dir, "model.pth")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"sac_kuavo_navigation_{timestamp}.pth"
        
        agent.save(save_path)
        print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    main() 
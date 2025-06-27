import gymnasium as gym
import rospy
import numpy as np
from collections import deque
from tqdm import tqdm

# 需要先导入模块，以便注册生效
import gym_hil 

def run_test():
    """
    一个更完善的测试函数，用于评估RLCar-v0环境。
    """
    rospy.init_node('gym_test_node', anonymous=True)

    print("Attempting to create 'gym_hil/RLCar-v0' environment...")

    try:
        # 使用 gym.make 创建环境
        env = gym.make("gym_hil/RLCar-v0", debug=False) # 在循环中打印太多信息，所以这里设为False
        print("Environment created successfully!")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    # --- 测试参数 ---
    num_episodes = 10         # 要运行的总回合数
    max_steps_per_episode = 2000 # 每个回合的最大步数

    # --- 统计数据 ---
    episode_rewards = []
    episode_lengths = []
    
    # 使用tqdm来显示总体进度
    for i in range(num_episodes):
        print(f"\n{'='*20} Episode {i + 1}/{num_episodes} {'='*20}")
        
        try:
            # 重置环境
            obs, info = env.reset()
            current_episode_reward = 0.0
            
            # 使用tqdm显示当前回合的进度
            pbar = tqdm(range(max_steps_per_episode), desc=f"Episode {i+1}")
            for step in pbar:
                # 随机采样一个动作
                action = env.action_space.sample()

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)

                current_episode_reward += reward

                # 更新进度条的描述信息
                pbar.set_postfix({
                    "Reward": f"{reward:.2f}",
                    "Total Reward": f"{current_episode_reward:.2f}",
                    "Done": terminated,
                    "Info": info
                })

                # 如果回合结束，则跳出循环
                if terminated or truncated:
                    break
            
            # 记录本回合的数据
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(step + 1)
            
            print(f"Episode {i + 1} finished after {step + 1} steps with a total reward of {current_episode_reward:.2f}")
            if info.get('reached_goal'):
                print("Result: Reached the goal!")
            elif info.get('collided'):
                print("Result: Collided.")
            else:
                print("Result: Reached max steps.")

        except Exception as e:
            print(f"\nAn error occurred during episode {i + 1}: {e}")
            import traceback
            traceback.print_exc()
            break  # 出现错误时停止测试

    # --- 打印最终统计结果 ---
    print(f"\n{'='*20} Test Finished {'='*20}")
    if episode_rewards:
        print(f"Number of episodes completed: {len(episode_rewards)}")
        print(f"Average reward per episode: {np.mean(episode_rewards):.2f}")
        print(f"Average length per episode: {np.mean(episode_lengths):.2f}")
        print(f"Min reward: {np.min(episode_rewards):.2f}")
        print(f"Max reward: {np.max(episode_rewards):.2f}")
    else:
        print("No episodes were completed.")

    # 清理环境
    env.close()
    print("Environment closed.")

if __name__ == '__main__':
    run_test()
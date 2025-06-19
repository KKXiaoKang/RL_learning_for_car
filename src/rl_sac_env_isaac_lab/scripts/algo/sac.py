import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Any, Optional, Union
from .policies import SACPolicy
import random
import numpy as np
from collections import deque
import torch.nn.functional as F
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import message_filters
from kuavo_msgs.msg import jointCmd
import threading
import time
from std_msgs.msg import Empty
from kuavo_msgs.srv import resetIsaaclab, resetIsaaclabRequest
from gymnasium import spaces
from gymnasium.spaces import Box
import os

rospy.init_node('sac_model_agent_collector', anonymous=True)

class RunningMeanStd:
    """
    Calculates the running mean and standard deviation of a data stream.
    This is useful for normalizing observations in reinforcement learning.
    """
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def get_buffer(self):
        return self.buffer

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape, device="cuda"):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        
    def add(self, obs, action, reward, next_obs, done):
        # 将numpy数组转换为torch张量并移动到正确的设备
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.obs.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.actions.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.rewards.device)
        self.next_obs[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.next_obs.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=self.dones.device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        idx = random.sample(range(self.size), batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx]
        )

class GymEnvWrapper:
    def __init__(self):
        # rospy.init_node('gym_env_wrapper', anonymous=True)
        
        # 定义观测空间 (12维：机器人位姿6 + 目标位姿6)
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )
        
        # 定义动作空间 (2维关节力矩)
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # 动作发布器
        self.cmd_pub = rospy.Publisher('/joint_cmd', jointCmd, queue_size=1)
        
        # 同步订阅观测话题
        robot_pose_sub = message_filters.Subscriber('/robot_pose', PoseStamped)
        goal_pose_sub = message_filters.Subscriber('/goal_pose', PoseStamped)
        
        # 时间同步器
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [robot_pose_sub, goal_pose_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self._obs_callback)
        
        # 观测缓存
        self.latest_obs = None
        self.obs_lock = threading.Lock()
        
        # 重置服务客户端
        self.reset_client = rospy.ServiceProxy('/isaac_lab_reset_scene', resetIsaaclab)
        
        # 初始化last_distance
        self.last_distance = float('inf')

        # 活动边界定义
        self.x_range = (-24.62, 4.5)
        self.y_range = (-17.32, 26.35)
        self.safety_margin = 0.5
        self.x_safe_range = (self.x_range[0] + self.safety_margin, self.x_range[1] - self.safety_margin)
        self.y_safe_range = (self.y_range[0] + self.safety_margin, self.y_range[1] - self.safety_margin)
        
        # 新的危险区
        self.x_danger_range = (-10.46, 3.37)
        self.y_danger_range = (8.9, 24.9)
        
        # 到达机器人半径
        self.reach_agent_radius = 1.0

        # 添加调试标志
        self.debug = False

    def _obs_callback(self, robot_pose, goal_pose):
        """同步处理机器人位姿和目标位姿"""
        with self.obs_lock:
            # 提取机器人位姿
            robot_state = np.array([
                robot_pose.pose.position.x,
                robot_pose.pose.position.y,
                robot_pose.pose.position.z,
                robot_pose.pose.orientation.x,
                robot_pose.pose.orientation.y,
                robot_pose.pose.orientation.z,
                robot_pose.pose.orientation.w
            ])
            
            # 提取目标位姿
            goal_state = np.array([
                goal_pose.pose.position.x,
                goal_pose.pose.position.y,
                goal_pose.pose.position.z,
                goal_pose.pose.orientation.x,
                goal_pose.pose.orientation.y,
                goal_pose.pose.orientation.z,
                goal_pose.pose.orientation.w
            ])
            
            # 合并观测
            self.latest_obs = np.concatenate([robot_state, goal_state]).astype(np.float32)

    def step(self, action):
        """执行动作并返回(next_obs, reward, done, info)"""
        # 发布动作
        cmd = jointCmd()
        cmd.tau = action.tolist()
        self.cmd_pub.publish(cmd)
        
        # 等待新观测
        start_time = time.time()
        while (time.time() - start_time) < 1.0:
            with self.obs_lock:
                if self.latest_obs is not None:
                    obs = self.latest_obs.copy()
                    self.latest_obs = None
                    break
            time.sleep(0.01)
        else:
            raise TimeoutError("未收到新的观测数据")
        
        # 计算奖励和终止条件
        robot_pos = obs[0:2]
        goal_pos = obs[7:9]
        position_diff = robot_pos - goal_pos
        distance = np.linalg.norm(position_diff)
        
        # 改进的奖励函数
        reward = 0.0
        
        # 边界碰撞检测
        collided_boundary = not (self.x_safe_range[0] < robot_pos[0] < self.x_safe_range[1] and \
                        self.y_safe_range[0] < robot_pos[1] < self.y_safe_range[1])

        # 危险区检测
        in_danger_zone = (self.x_danger_range[0] < robot_pos[0] < self.x_danger_range[1] and \
                          self.y_danger_range[0] < robot_pos[1] < self.y_danger_range[1])

        collided = collided_boundary or in_danger_zone

        if collided:
            reward -= 10.0 # 碰撞时给予一个负奖励
            if self.debug:
                if collided_boundary:
                    print("Robot collided with boundary!")
                elif in_danger_zone:
                    print("Robot entered danger zone!")
        
        # 距离奖励：当距离减小时给予正奖励
        if self.last_distance != float('inf'):
            distance_change = self.last_distance - distance
            reward += 10.0 * distance_change
            if self.debug:
                print(f"Distance change: {distance_change:.3f}, Reward from distance: {10.0 * distance_change:.3f}")
        
        # 到达目标奖励
        if distance < self.reach_agent_radius: # 小于50cm 则达标已经到位
            reward += 300.0
            if self.debug:
                print("Reached goal! +100 reward")
        
        if self.debug:
            print(f"Robot pos: {robot_pos}, Goal pos: {goal_pos}")
            print(f"Distance: {distance:.3f}, Last distance: {self.last_distance:.3f}")
            # print(f"Action penalty: {action_penalty:.3f}")
            print(f"Total reward: {reward:.3f}")
        
        # 更新last_distance
        self.last_distance = distance
        
        # 判断是否终止
        done = distance < self.reach_agent_radius or collided
        
        return obs, reward, done, {"collided": collided, "distance": distance}

    def reset(self):
        """重置环境"""
        try:
            resp = self.reset_client(0)
            if not resp.success:
                raise RuntimeError(f"重置失败: {resp.message}")
            
            with self.obs_lock:
                self.latest_obs = None
                
            start_time = time.time()
            while (time.time() - start_time) < 5.0:
                with self.obs_lock:
                    if self.latest_obs is not None:
                        obs = self.latest_obs.copy()
                        # 初始化last_distance
                        position_diff = obs[0:2] - obs[7:9]
                        self.last_distance = np.linalg.norm(position_diff)
                        if self.debug:
                            print(f"Reset - Initial distance: {self.last_distance:.3f}")
                        return obs
                time.sleep(0.1)
            raise TimeoutError("重置后未收到新的观测数据")
            
        except rospy.ServiceException as e:
            raise RuntimeError(f"服务调用失败: {str(e)}")


class SAC:
    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnvWrapper, str],
        learning_rate: Union[float] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        use_ros_buffer: bool = False,
        **kwargs
    ):
        # 首先设置device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 环境初始化
        if isinstance(env, str):
            self.env = GymEnvWrapper()
        else:
            self.env = env
        
        # 使用env的space定义
        obs_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        
        # 观测归一化
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        
        # 经验缓冲区初始化
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            obs_shape,
            action_shape,
            device=self.device
        )
        
        # 其他参数设置
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        
        # Policy初始化
        if isinstance(policy, str):
            self.policy = SACPolicy(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                lr_schedule=lambda _: learning_rate
            ).to(self.device)
        elif issubclass(policy, SACPolicy):
            self.policy = policy(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                lr_schedule=lambda _: learning_rate
            ).to(self.device)
        else:
            raise ValueError("Unsupported policy type")
        
        # 初始化目标网络
        self.policy.update_target_network(tau=1.0)
        
        # 梯度更新计数器
        self._n_updates = 0
        
        # 设置tensorboard
        self.tensorboard_log = tensorboard_log
        if self.tensorboard_log:
            from torch.utils.tensorboard import SummaryWriter
            import datetime
            
            # 创建带时间戳的日志目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(self.tensorboard_log, f"run_{timestamp}")
            
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")
        else:
            self.writer = None
        
        print(f"Using device: {self.device}")
        print(f"Policy device: {next(self.policy.parameters()).device}")
        
    def train(self, gradient_steps=1, step=0):
        # 检查缓冲区是否有足够的数据
        if self.replay_buffer.size < self.batch_size:
            return  # 如果数据不够，直接返回
        
        for _ in range(gradient_steps):
            # 从缓冲区采样
            obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
            
            # 归一化观测
            mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
            var = torch.as_tensor(self.obs_rms.var, dtype=torch.float32, device=self.device)
            
            # 确保var非负
            var = torch.clamp(var, min=1e-8)

            normalized_obs = (obs - mean) / torch.sqrt(var)
            normalized_next_obs = (next_obs - mean) / torch.sqrt(var)
            
            # 裁剪观测值
            clipped_obs = torch.clamp(normalized_obs, -10.0, 10.0)
            clipped_next_obs = torch.clamp(normalized_next_obs, -10.0, 10.0)

            # 更新Critic
            q_loss = self.policy.calculate_loss_q(clipped_obs, actions, rewards, clipped_next_obs, dones, self.gamma)
            self.policy.critic_optimizer.zero_grad()
            q_loss.backward()
            self.policy.critic_optimizer.step()
            
            # 记录Q值损失
            if self.writer is not None:
                self.writer.add_scalar('train/q_loss', q_loss.item(), step) # 记录Q值损失
            
            # 更新Actor
            policy_loss, log_pi = self.policy.calculate_loss_pi(clipped_obs)
            self.policy.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.policy.actor_optimizer.step()
            
            # 记录策略损失和策略熵
            if self.writer is not None:
                self.writer.add_scalar('train/policy_loss', policy_loss.item(), step) # 记录策略损失
                self.writer.add_scalar('train/policy_entropy', -log_pi.mean().item(), step) # 记录策略熵
            
            # 更新alpha
            alpha_loss = self.policy.calculate_loss_alpha(log_pi)
            self.policy.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.policy.alpha_optimizer.step()
            
            # 记录alpha损失和alpha值
            if self.writer is not None:
                self.writer.add_scalar('train/alpha_loss', alpha_loss.item(), step) # 记录alpha损失
                self.writer.add_scalar('train/alpha', self.policy.get_alpha().item(), step) # 记录alpha值
            
            self._n_updates += 1
            # 更新目标网络
            if self._n_updates % self.target_update_interval == 0:
                self.policy.update_target_network(self.tau)
            
            # 记录Q值统计信息
            if self.writer is not None:
                with torch.no_grad():
                    q_values = self.policy.critic(clipped_obs, actions)
                    self.writer.add_scalar('train/q_value_mean', q_values.mean().item(), step) # 记录Q值平均值
                    self.writer.add_scalar('train/q_value_std', q_values.std().item(), step) # 记录Q值标准差

    def eval(self):
        pass

    def save(self, path: str):
        """
        Save all the attributes of the object and the model parameters in a dictionary.
        :param path: path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
            'obs_rms_count': self.obs_rms.count
        }, path)
    
    def load(self, path: str):
        """
        Load model parameters and attributes from a file.
        :param path: path to the saved model
        """
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data['policy_state_dict'])
        self.obs_rms.mean = data['obs_rms_mean']
        self.obs_rms.var = data['obs_rms_var']
        self.obs_rms.count = data['obs_rms_count']
        self.policy.update_target_network(tau=1.0) # Hard update

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 100,
        max_episode_steps: int = 1000,
        **kwargs
    ):
        # 初始化环境
        self.env = GymEnvWrapper()
        obs = self.env.reset()
        episode_rewards = []
        current_ep_reward = 0
        episode_lengths = []  # 添加episode长度记录
        current_ep_length = 0
        
        # 主训练循环
        for step in range(total_timesteps):
            # 更新观测统计量并归一化
            self.obs_rms.update(np.array([obs]))
            normalized_obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            clipped_obs = np.clip(normalized_obs, -10.0, 10.0)
            
            # 选择动作
            action = self.policy.predict(clipped_obs, deterministic=False)
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action)
            
            # 记录距离
            if self.writer is not None and "distance" in info:
                self.writer.add_scalar('rollout/distance', info['distance'], step)
            
            current_ep_length += 1
            # 检查是否因为超时而终止
            truncated = current_ep_length >= max_episode_steps
            episode_done = done or truncated # 完成 或者 超时

            # 存储经验 (使用未归一化的观测)
            self.replay_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=episode_done # 使用合并后的终止信号
            )
            
            # 更新统计
            current_ep_reward += reward
            obs = next_obs
            
            # 环境终止处理
            if episode_done:
                episode_count = len(episode_rewards) + 1  # 当前是第几个episode
                
                # 确定终止原因
                is_collided = info.get("collided", False)
                # 成功完成的条件是 done 为 True 且没有发生碰撞
                is_success = done and not is_collided
                # 超时的条件是 truncated 为 True 且 episode 不是因为 done 结束
                is_truncated = truncated and not done

                # 记录终止原因到 TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('episode/is_collided', float(is_collided), episode_count)
                    self.writer.add_scalar('episode/is_success', float(is_success), episode_count)
                    self.writer.add_scalar('episode/is_truncated', float(is_truncated), episode_count)
                
                if is_truncated:
                    print(f"Episode truncated at {current_ep_length} steps.")
                
                if is_collided:
                    print(f"Episode terminated due to collision after {current_ep_length} steps.")
                
                if is_success:
                    print(f"Episode finished successfully after {current_ep_length} steps.")

                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                
                # 记录episode统计信息
                if self.writer is not None:
                    self.writer.add_scalar('episode/reward', current_ep_reward, episode_count) # 记录每个episode的奖励
                    self.writer.add_scalar('episode/length', current_ep_length, episode_count) # 记录每个episode的步长
                    self.writer.add_scalar('episode/avg_reward', np.mean(episode_rewards[-10:]), episode_count) # 记录最近10个episode的平均奖励
                    self.writer.add_scalar('episode/avg_length', np.mean(episode_lengths[-10:]), episode_count) # 记录最近10个episode的平均步长
                
                print(f"Episode {episode_count} Reward: {current_ep_reward:.2f}, Length: {current_ep_length}")
                obs = self.env.reset()
                current_ep_reward = 0
                current_ep_length = 0
                
            # 训练步骤 - 只在有足够数据时进行
            if (step > self.learning_starts and 
                step % self.train_freq == 0 and 
                self.replay_buffer.size >= self.batch_size):
                self.train(gradient_steps=self.gradient_steps, step=step)
            
            # 日志记录
            if step % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                print(f"Step: {step}/{total_timesteps} | Avg Reward: {avg_reward:.2f} | Buffer Size: {self.replay_buffer.size}")
                
                # 记录训练统计信息
                if self.writer is not None:
                    self.writer.add_scalar('train/buffer_size', self.replay_buffer.size, step) # 记录缓冲区大小
                    self.writer.add_scalar('train/avg_reward', avg_reward, step) # 记录平均奖励
                    
                    # 记录动作统计信息
                    if len(episode_rewards) > 0:
                        actions = self.policy.predict(obs, deterministic=False)
                        self.writer.add_scalar('train/action_mean', np.mean(actions), step) # 记录动作平均值
                        self.writer.add_scalar('train/action_std', np.std(actions), step) # 记录动作标准差
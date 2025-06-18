# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional, Dict, Any
from gymnasium.spaces import Box
import numpy as np
from torch import nn as nn
import torch.nn.functional as F

class FlattenExtractor(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        # 计算展平后的特征维度
        self.features_dim = int(np.prod(observation_space.shape))
        
    def forward(self, observations):
        return torch.flatten(observations, start_dim=1)

class Actor(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch=[256, 256],
        activation_fn=nn.ELU,
        use_sde=False,
        log_std_init=-3,
        use_expln=False,
        clip_mean=2.0
    ):
        super().__init__()
        self.use_sde = use_sde
        self.action_dim = action_space.shape[0]
        
        # 构建特征提取器
        self.features_extractor = FlattenExtractor(observation_space)
        
        # 构建MLP
        self.latent_pi = nn.Sequential(
            nn.Linear(self.features_extractor.features_dim, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn()
        )
        
        # 输出层
        self.mu = nn.Linear(net_arch[-1], self.action_dim)
        self.log_std = nn.Linear(net_arch[-1], self.action_dim)
        
        # 初始化参数
        nn.init.constant_(self.log_std.bias, log_std_init)
        # 使用更保守的初始化
        for layer in self.latent_pi:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0.0)
        
        # 动作缩放参数
        self.tau_scale = 50.0  # 将[-1,1]缩放到[-50,50]
        
        # 添加梯度裁剪
        self.max_grad_norm = 1.0

    def forward(self, obs, deterministic=False):
        # 检查输入是否包含NaN
        if torch.isnan(obs).any():
            print("Warning: NaN detected in input")
            obs = torch.nan_to_num(obs, nan=0.0)
            
        latent_pi = self.latent_pi(obs)
        mean = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        
        # 限制log_std的范围
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        distribution = Normal(mean, std)
        # 动作在 tanh 之前
        actions_raw = distribution.rsample() if not deterministic else mean
        # 经过 tanh 激活，范围 [-1, 1]
        actions_tanh = torch.tanh(actions_raw)
        
        # 缩放动作到力矩范围
        actions_scaled = actions_tanh * self.tau_scale

        # 在 tanh 后的 [-1,1] 范围内计算动作惩罚
        action_penalty = -0.01 * torch.sum(torch.square(actions_tanh), dim=-1)

        if deterministic:
            return actions_scaled, None, None

        log_prob = distribution.log_prob(actions_raw).sum(axis=-1)
        # Gart-Martin等人提出的修正，用于tanh变换
        log_prob -= (2*(np.log(2) - actions_raw - F.softplus(-2*actions_raw))).sum(axis=1)

        return actions_scaled, log_prob, action_penalty

class ContinuousCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch=[256, 256],
        activation_fn=nn.ELU,
        n_critics=2
    ):
        super().__init__()
        self.n_critics = n_critics
        self.q_networks = nn.ModuleList()
        
        # 构建多个Q网络
        for _ in range(n_critics):
            q_net = nn.Sequential(
                nn.Linear(observation_space.shape[0] + action_space.shape[0], net_arch[0]),
                activation_fn(),
                nn.Linear(net_arch[0], net_arch[1]),
                activation_fn(),
                nn.Linear(net_arch[1], 1)
            )
            self.q_networks.append(q_net)

    def forward(self, obs, actions):
        input_tensor = torch.cat([obs, actions], dim=-1)
        return torch.cat([q_net(input_tensor) for q_net in self.q_networks], dim=1)

class SACPolicy(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=[256, 256],
        activation_fn=nn.ELU,
        **kwargs
    ):
        super().__init__()
        # 创建actor和critic
        self.actor = Actor(observation_space, action_space, net_arch, activation_fn)
        self.critic = ContinuousCritic(observation_space, action_space, net_arch, activation_fn)
        self.critic_target = ContinuousCritic(observation_space, action_space, net_arch, activation_fn)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 自动熵调整
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape)).item()
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        
        # 优化器配置
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_schedule(1))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_schedule(1))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_schedule(1))

    def forward(self, obs, deterministic=False):
        return self.actor(obs, deterministic=deterministic)

    def update_target_network(self, tau=0.005):
        # 软更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            return self.forward(obs, deterministic)

    def get_alpha(self):
        return self.log_alpha.exp()
    
    def calculate_loss_q(self, obs, actions, rewards, next_obs, dones, gamma):
        """
            计算Q值损失
            :param obs: 当前状态
            :param actions: 当前动作
            :param rewards: 奖励
            :param next_obs: 下一个状态
            :param dones: 是否结束
            :param gamma: 折扣因子
        """
        # 目标网络无梯度计算Q值
        with torch.no_grad():
            next_actions, log_pi_next, _ = self.actor(next_obs)
            target_q_values = self.critic_target(next_obs, next_actions)
            target_q_min = target_q_values.min(1)[0]
            target_q = rewards + (1 - dones) * gamma * (target_q_min - self.get_alpha().detach() * log_pi_next)
        # 当前网络计算Q值
        current_q = self.critic(obs, actions)  # [batch_size, n_critics]
        # 计算Q值损失
        q_loss = 0.5 * (current_q - target_q.unsqueeze(1)).pow(2).sum(dim=1).mean()
        return q_loss

    def calculate_loss_pi(self, obs):
        """
            计算策略损失
            :param obs: 当前状态
        """
        actions_pi, log_pi, action_penalty = self.actor(obs) # 当前网络计算动作
        q_values_pi = self.critic(obs, actions_pi)
        min_qf_pi = q_values_pi.min(1)[0]
        policy_loss = (self.get_alpha().detach() * log_pi - min_qf_pi + action_penalty).mean()
        return policy_loss, log_pi

    def calculate_loss_alpha(self, log_pi):
        alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """用于推理的预测方法"""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(next(self.parameters()).device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)  # 添加batch维度
            
            # Actor 现在返回 (动作, log_prob, 惩罚)
            action, _, _ = self.actor(obs_tensor, deterministic=deterministic)
            
            # 如果action是元组，取第一个元素（动作）
            if isinstance(action, tuple):
                action = action[0]
            return action.cpu().numpy().squeeze()

if __name__ == '__main__':
    # 创建模拟的动作空间对象（假设是Box类型）
    from gymnasium import spaces
    import numpy as np
    
    # 观察空间维度：92*15，动作空间维度：26
    obs_dim = 92 * 15
    action_dim = 26
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))  # 假设动作范围是[-1,1]
    
    model = SACPolicy(observation_space, action_space, lr_schedule=lambda _: 3e-4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

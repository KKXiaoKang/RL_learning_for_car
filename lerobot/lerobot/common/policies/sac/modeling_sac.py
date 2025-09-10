#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from dataclasses import asdict
from typing import Callable, Literal

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution

from lerobot.common.policies.normalize import NormalizeBuffer
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.sac.configuration_sac import SACConfig, is_image_feature
from lerobot.common.policies.utils import get_device_from_parameters

# Add ROS imports for visualization
try:
    import rospy
    from sensor_msgs.msg import Image, CompressedImage
    import cv2
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rospy = None
    Image = None
    CompressedImage = None
    cv2 = None


DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class SACPolicy(
    PreTrainedPolicy,
):
    config_class = SACConfig
    name = "sac"

    def __init__(
        self,
        config: SACConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        # 通过配置文件初始化父类
        super().__init__(config)
        # 验证输入输出特征
        config.validate_features() 
        self.config = config

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features["action"].shape[0] # 🔥 获取连续动作维度
        self._init_normalization(dataset_stats) # 🔥 初始化归一化, 通过 dataset_stats 中的 min 和 max 对输入数据进行归一化
        self._init_encoders() # 🔥 初始化编码器
        self._init_critics(continuous_action_dim) # 🔥 初始化critic
        self._init_actor(continuous_action_dim) # 🔥 初始化actor
        self._init_temperature() # 🔥 初始化温度
        self._init_warmup_parameters() # 🔥 初始化warm-up参数（如果启用）

        # Initialize member variable to store td_target for wandb logging
        self.last_td_target: Tensor | None = None
    
    def get_optim_params(self) -> dict:
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
            "temperature": self.log_alpha,
        }
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""

        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            # Cache and normalize image features
            observations_features = self.actor.encoder.get_cached_image_features(batch, normalize=True)

        # 🔥 Q-chunking: 处理序列ACT Actor的动作选择
        if hasattr(self.actor, 'chunk_size') and getattr(self.config, 'use_sequence_act_actor', False):
            # 序列ACT Actor：获取动作序列，但只返回第一个动作用于执行
            action_sequence, _, _ = self.actor(
                batch, 
                observations_features, 
                return_sequence=True
            )
            # Q-chunking核心：只使用序列的第一个动作进行实际执行
            actions = action_sequence[:, 0, :]  # (batch, action_dim)
        else:
            # 传统Actor：直接获取单步动作
            actions, _, _ = self.actor(batch, observations_features)

        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, observations_features)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            actions = torch.cat([actions, discrete_action], dim=-1)

        return actions

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def discrete_critic_forward(
        self, observations, use_target=False, observation_features=None
    ) -> torch.Tensor:
        """Forward pass through a discrete critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features to avoid recomputing encoder output

        Returns:
            Tensor of Q-values from the discrete critic network
        """
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic"] = "critic",
    ) -> dict[str, Tensor]:
        """Compute the loss for the given model

        Args:
            batch: Dictionary containing:
                - action: Action tensor
                - reward: Reward tensor
                - state: Observations tensor dict
                - next_state: Next observations tensor dict
                - done: Done mask tensor
                - observation_feature: Optional pre-computed observation features
                - next_observation_feature: Optional pre-computed next observation features
            model: Which model to compute the loss for ("actor", "critic", "discrete_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        # 注意：对于序列ACT Actor的情况，action可能不存在
        actions: Tensor = batch.get("action")  # 对于序列ACT可能为None
        observations: dict[str, Tensor] | list[dict[str, Tensor]] = batch["state"]  # 可能是序列
        observation_features: Tensor = batch.get("observation_feature")

        if model == "critic":
            # Critic需要actions，确保它存在
            if actions is None:
                raise ValueError("Critic loss computation requires 'action' in batch")
            
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_critic = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )

            return {"loss_critic": loss_critic}

        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Discrete critic也需要actions
            if actions is None:
                raise ValueError("Discrete critic loss computation requires 'action' in batch")
            
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_discrete_critic": loss_discrete_critic}
        if model == "actor":
            # 提取专家动作（如果可用）和训练步数
            expert_actions = batch.get("expert_action")
            expert_action_sequences = batch.get("expert_action_sequences")
            training_step = batch.get("training_step", 0)
            
            return {
                "loss_actor": self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
                    expert_actions=expert_actions,
                    expert_action_sequences=expert_action_sequences,
                    training_step=training_step,
                )
            }

        if model == "temperature":
            return {
                "loss_temperature": self.compute_loss_temperature(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    param.data * self.config.critic_target_update_weight
                    + target_param.data * (1.0 - self.config.critic_target_update_weight)
                )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
        use_q_chunking: bool = None,  # Q-chunking开关
    ) -> Tensor:
        """
        计算Critic损失，支持Q-chunking的n-step TD learning
        
        Q-chunking核心改进：
        1. 使用动作序列的第一个动作进行Q值计算
        2. 使用n-step TD targets进行更稳定的学习
        3. 考虑动作序列的时间一致性
        """
        # 检查是否启用Q-chunking
        if use_q_chunking is None:
            use_q_chunking = (hasattr(self.actor, 'chunk_size') and 
                             getattr(self.config, 'use_sequence_act_actor', False) and
                             getattr(self.config, 'enable_q_chunking', True))
        
        with torch.no_grad():
            if use_q_chunking:
                # Q-chunking模式：使用动作序列进行预测
                td_target = self._compute_q_chunking_td_target(
                    next_observations=next_observations,
                    rewards=rewards,
                    done=done,
                    next_observation_features=next_observation_features
                )
            else:
                # 传统SAC模式：单步TD target
                td_target = self._compute_traditional_td_target(
                    next_observations=next_observations,
                    rewards=rewards,
                    done=done,
                    next_observation_features=next_observation_features
                )

            # Store td_target for wandb logging (detach to avoid affecting gradients)
            self.last_td_target = td_target.detach().clone()

        # 计算当前Q值预测
        q_preds = self._compute_current_q_values(
            observations=observations,
            actions=actions,
            observation_features=observation_features,
            use_q_chunking=use_q_chunking
        )

        # 计算TD损失
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()
        
        return critics_loss
    
    def _compute_q_chunking_td_target(
        self,
        next_observations,
        rewards,
        done,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        """
        计算Q-chunking的TD target
        
        Q-chunking的核心创新：
        1. 使用动作序列的联合概率进行策略评估
        2. 使用n-step returns进行更稳定的值函数估计
        3. 在"chunked"动作空间中运行SAC
        """
        # 1. 获取下一状态的动作序列预测
        next_action_sequence, next_log_probs_joint, _ = self.actor(
            next_observations, 
            next_observation_features, 
            return_sequence=True
        )
        
        # 2. 使用序列的第一个动作计算Q值（Q-chunking的关键）
        next_first_action = next_action_sequence[:, 0, :]  # (batch, action_dim)
        
        # 3. 计算目标Q值
        q_targets = self.critic_forward(
            observations=next_observations,
            actions=next_first_action,
            use_target=True,
            observation_features=next_observation_features,
        )

        # 4. Critic ensemble子采样（防止高UTD时过拟合）
        if self.config.num_subsample_critics is not None:
            indices = torch.randperm(self.config.num_critics)
            indices = indices[: self.config.num_subsample_critics]
            q_targets = q_targets[indices]

        # 5. 计算最小Q值（保守估计）
        min_q, _ = q_targets.min(dim=0)
        
        # 6. Q-chunking核心：使用联合对数概率进行熵正则化
        if self.config.use_backup_entropy:
            # 注意：这里使用的是整个动作序列的联合概率，而不是单个动作的概率
            min_q = min_q - (self.temperature * next_log_probs_joint)

        # 7. 计算n-step TD target（Q-chunking的另一个关键改进）
        chunk_size = getattr(self.actor, 'chunk_size', 1)
        if chunk_size > 1 and hasattr(self.actor, 'compute_n_step_returns'):
            try:
                # 使用序列Actor的n-step return计算能力
                n_step_returns = self.actor.compute_n_step_returns(
                    rewards=rewards,
                    next_observations=next_observations,
                    done=done,
                    gamma=self.config.discount,
                    observation_features=next_observation_features
                )
                # n-step TD target: R_t + γ^n * Q_target(s_{t+n}, a_{t+n})
                td_target = n_step_returns + (1 - done) * (self.config.discount ** chunk_size) * min_q
                
                logging.debug(f"Q-chunking using {chunk_size}-step TD target")
            except Exception as e:
                logging.warning(f"Failed to compute n-step returns, falling back to 1-step: {e}")
                td_target = rewards + (1 - done) * self.config.discount * min_q
        else:
            # 降级到标准1-step TD target
            td_target = rewards + (1 - done) * self.config.discount * min_q

        return td_target
    
    def _compute_traditional_td_target(
        self,
        next_observations,
        rewards,
        done,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        """计算传统SAC的TD target"""
        # 传统Actor：直接获取单步动作
        next_action_preds, next_log_probs, _ = self.actor(next_observations, next_observation_features)

        # 计算目标Q值
        q_targets = self.critic_forward(
            observations=next_observations,
            actions=next_action_preds,
            use_target=True,
            observation_features=next_observation_features,
        )

        # Critic ensemble子采样
        if self.config.num_subsample_critics is not None:
            indices = torch.randperm(self.config.num_critics)
            indices = indices[: self.config.num_subsample_critics]
            q_targets = q_targets[indices]

        # 计算最小Q值
        min_q, _ = q_targets.min(dim=0)
        if self.config.use_backup_entropy:
            min_q = min_q - (self.temperature * next_log_probs)

        # 标准1-step TD target
        td_target = rewards + (1 - done) * self.config.discount * min_q
        return td_target
    
    def _compute_current_q_values(
        self,
        observations,
        actions,
        observation_features: Tensor | None = None,
        use_q_chunking: bool = False,
    ) -> Tensor:
        """计算当前状态-动作对的Q值"""
        if use_q_chunking:
            # Q-chunking: 处理可能的动作序列输入
            if len(actions.shape) == 3:
                # 动作序列输入：只使用第一个动作计算Q值
                current_actions = actions[:, 0, :]  # (batch, action_dim)
            else:
                # 单个动作输入
                current_actions = actions
        else:
            # 传统模式：直接使用动作
            current_actions = actions
            
        # 处理离散动作（如果存在）
        if self.config.num_discrete_actions is not None:
            current_actions = current_actions[:, :DISCRETE_DIMENSION_INDEX]
            
        # 计算Q值
        q_preds = self.critic_forward(
            observations=observations,
            actions=current_actions,
            use_target=False,
            observation_features=observation_features,
        )
        
        return q_preds

    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties: Tensor | None = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self.discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            # Get target Q-values from target network
            target_next_discrete_qs = self.discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)

            # Compute target Q-value with Bellman equation
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q

        # Get predicted Q-values for current observations
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)
        return discrete_critic_loss

    def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
        """
        计算温度参数损失，支持Q-chunking
        
        Q-chunking改进：使用动作序列的联合概率计算温度损失
        这确保温度参数适应动作序列的复杂性
        """
        # 检查是否启用Q-chunking
        use_q_chunking = (hasattr(self.actor, 'chunk_size') and 
                         getattr(self.config, 'use_sequence_act_actor', False) and
                         getattr(self.config, 'enable_q_chunking', True))
        
        # 基于Q-chunking论文 (https://arxiv.org/abs/2507.07969) 的温度损失计算
        if use_q_chunking:
            # Q-chunking模式：使用动作序列的联合对数概率
            with torch.no_grad():
                # 只对Actor前向传播禁用梯度，获取联合概率
                _, log_probs_joint, _ = self.actor(
                    observations, 
                    observation_features, 
                    return_sequence=True
                )
            
            # 根据chunk_size调整目标熵（论文核心思想）
            chunk_size = getattr(self.actor, 'chunk_size', 1)
            adjusted_target_entropy = self._get_adjusted_target_entropy(chunk_size)
            
            # Q-chunking温度损失：保持log_alpha梯度用于温度参数优化
            temperature_loss = (-self.log_alpha.exp() * (log_probs_joint + adjusted_target_entropy)).mean()
        else:
            # 传统SAC模式：使用单步对数概率
            with torch.no_grad():
                # 只对Actor前向传播禁用梯度
                _, log_probs, _ = self.actor(observations, observation_features)
            
            # 标准SAC温度损失：保持log_alpha梯度
            temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
                
        return temperature_loss
    
    def _get_adjusted_target_entropy(self, chunk_size: int) -> float:
        """
        计算Q-chunking的调整目标熵
        
        由于Q-chunking使用联合概率，目标熵需要相应调整：
        - 联合概率通常比单个动作概率更小（更负）
        - 目标熵应该根据序列长度进行缩放
        
        Args:
            chunk_size: 动作序列长度
            
        Returns:
            调整后的目标熵
        """
        base_target_entropy = self.target_entropy
        
        # Q-chunking目标熵调整策略
        entropy_scaling_strategy = getattr(self.config, 'q_chunking_entropy_scaling', 'linear')
        
        if entropy_scaling_strategy == 'linear':
            # 线性缩放：目标熵与序列长度成正比
            adjusted_entropy = base_target_entropy * chunk_size
        elif entropy_scaling_strategy == 'sqrt':
            # 平方根缩放：更温和的调整
            adjusted_entropy = base_target_entropy * math.sqrt(chunk_size)
        elif entropy_scaling_strategy == 'log':
            # 对数缩放：最温和的调整
            adjusted_entropy = base_target_entropy * math.log(chunk_size + 1)
        else:
            # 无缩放：保持原始目标熵
            adjusted_entropy = base_target_entropy
        
        logging.debug(f"Q-chunking entropy: base={base_target_entropy:.3f}, "
                     f"adjusted={adjusted_entropy:.3f}, chunk_size={chunk_size}")
        
        return adjusted_entropy

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
        expert_actions: Tensor | None = None,
        expert_action_sequences: Tensor | None = None,  # 专家动作序列用于序列ACT
        training_step: int = 0,
    ) -> Tensor:
        """
        计算Actor损失，包含SAC损失和可选的BC损失
        
        支持传统MLP Actor、基础ACT Actor和序列ACT Actor三种架构
        
        Args:
            observations: 观测数据
            observation_features: 预计算的观测特征
            expert_actions: 专家动作（来自离线数据集），用于计算单步BC损失
            expert_action_sequences: 专家动作序列，用于序列ACT的BC损失
            training_step: 当前训练步数，用于动态权重计算
            
        Returns:
            Actor总损失
        """
        # 检查Actor类型
        use_act_actor = getattr(self.config, 'use_act_actor', False)
        use_sequence_act_actor = getattr(self.config, 'use_sequence_act_actor', False)
        
        if use_act_actor and use_sequence_act_actor:
            # 序列ACT Actor - 支持动作序列预测
            return self._compute_sequence_actor_loss(
                observations=observations,
                observation_features=observation_features,
                expert_actions=expert_actions,
                expert_action_sequences=expert_action_sequences,
                training_step=training_step
            )
        elif use_act_actor:
            # 基础ACT Actor - 单步预测
            actions_pi, log_probs, means = self.actor(observations, observation_features)
        else:
            # 传统MLP Actor
            actions_pi, log_probs, _ = self.actor(observations, observation_features)

        # 传统单步损失计算
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        # 计算标准SAC Actor损失
        sac_actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        
        # 如果有专家动作，计算BC损失
        if expert_actions is not None:
            if use_act_actor:
                # 对于ACT Actor，使用确定性动作（means）计算BC损失
                bc_mse_loss = self._compute_bc_loss_act(
                    predicted_actions=means,  # 使用确定性输出
                    expert_actions=expert_actions
                )
            else:
                # 对于传统Actor，使用采样动作计算BC损失
                bc_mse_loss = self._compute_bc_loss(
                    predicted_actions=actions_pi,
                    expert_actions=expert_actions
                )
            
            # 计算动态权重（随训练步数衰减）
            bc_weight = self._compute_dynamic_bc_weight(training_step)
            sac_weight = 1.0 - bc_weight
            
            # 混合损失
            actor_loss = sac_weight * sac_actor_loss + bc_weight * bc_mse_loss
            
            # 存储损失组件用于日志记录
            self._last_bc_loss = bc_mse_loss.detach().clone()
            self._last_bc_weight = bc_weight
            self._last_sac_actor_loss = sac_actor_loss.detach().clone()
        else:
            # 纯SAC损失
            actor_loss = sac_actor_loss
            self._last_bc_loss = None
            self._last_bc_weight = 0.0
            self._last_sac_actor_loss = sac_actor_loss.detach().clone()
            
        return actor_loss
    
    def _compute_bc_loss(self, predicted_actions: Tensor, expert_actions: Tensor) -> Tensor:
        """
        计算行为克隆的MSE损失
        
        Args:
            predicted_actions: 策略网络预测的动作
            expert_actions: 专家演示的动作
            
        Returns:
            BC MSE损失
        """
        # 记录原始形状用于调试
        orig_pred_shape = predicted_actions.shape
        orig_expert_shape = expert_actions.shape
        
        # 首先处理动作维度不匹配的问题
        if predicted_actions.shape[1] != expert_actions.shape[1]:
            # 如果有离散动作维度，只使用连续部分
            if self.config.num_discrete_actions is not None:
                expert_actions = expert_actions[:, :predicted_actions.shape[1]]
                logging.debug(f"BC: Adjusted expert action dims for discrete actions: {orig_expert_shape} -> {expert_actions.shape}")
            else:
                raise ValueError(f"Action dimension mismatch: predicted {predicted_actions.shape[1]} dims, expert {expert_actions.shape[1]} dims")
        
        # 处理批次大小不匹配的问题
        if predicted_actions.shape[0] != expert_actions.shape[0]:
            pred_batch_size = predicted_actions.shape[0]
            expert_batch_size = expert_actions.shape[0]
            
            if expert_batch_size < pred_batch_size:
                # 重复专家动作以匹配预测批次大小
                repeat_times = (pred_batch_size + expert_batch_size - 1) // expert_batch_size
                expert_actions = expert_actions.repeat(repeat_times, 1)[:pred_batch_size]
                logging.debug(f"BC: Repeated expert actions: {orig_expert_shape} -> {expert_actions.shape}")
            elif expert_batch_size > pred_batch_size:
                # 截断专家动作以匹配预测批次大小
                expert_actions = expert_actions[:pred_batch_size]
                logging.debug(f"BC: Truncated expert actions: {orig_expert_shape} -> {expert_actions.shape}")
        
        # 最终检查形状是否匹配
        if predicted_actions.shape != expert_actions.shape:
            raise ValueError(f"Final shape mismatch after adjustments: predicted {predicted_actions.shape}, expert {expert_actions.shape}")
        
        # 计算MSE损失
        bc_loss = F.mse_loss(predicted_actions, expert_actions)
        return bc_loss
    
    def _compute_dynamic_bc_weight(self, training_step: int) -> float:
        """
        计算动态BC权重，随训练步数衰减
        
        策略：
        - 初始权重：0.5（BC和SAC各占一半）
        - 衰减方式：指数衰减
        - 最终权重：0.01（保持少量BC正则化）
        - 衰减步数：根据配置设定
        
        Args:
            training_step: 当前训练步数
            
        Returns:
            BC权重（0-1之间）
        """
        # 从配置中获取衰减参数，如果没有则使用默认值
        initial_bc_weight = getattr(self.config, 'bc_initial_weight', 0.5)
        final_bc_weight = getattr(self.config, 'bc_final_weight', 0.01)
        bc_decay_steps = getattr(self.config, 'bc_decay_steps', 50000)
        
        if training_step >= bc_decay_steps:
            return final_bc_weight
        
        # 指数衰减
        decay_ratio = training_step / bc_decay_steps
        weight = initial_bc_weight * (final_bc_weight / initial_bc_weight) ** decay_ratio
        
        return max(weight, final_bc_weight)
    
    def _compute_bc_loss_act(self, predicted_actions: Tensor, expert_actions: Tensor) -> Tensor:
        """
        计算ACT Actor的行为克隆MSE损失
        
        Args:
            predicted_actions: ACT Actor预测的确定性动作（means）
            expert_actions: 专家演示的动作
            
        Returns:
            BC MSE损失
        """
        # 记录原始形状用于调试
        orig_pred_shape = predicted_actions.shape
        orig_expert_shape = expert_actions.shape
        
        # 首先处理动作维度不匹配的问题
        if predicted_actions.shape[1] != expert_actions.shape[1]:
            # 如果有离散动作维度，只使用连续部分
            if self.config.num_discrete_actions is not None:
                expert_actions = expert_actions[:, :predicted_actions.shape[1]]
                logging.debug(f"ACT BC: Adjusted expert action dims for discrete actions: {orig_expert_shape} -> {expert_actions.shape}")
            else:
                raise ValueError(f"Action dimension mismatch: predicted {predicted_actions.shape[1]} dims, expert {expert_actions.shape[1]} dims")
        
        # 处理批次大小不匹配的问题
        if predicted_actions.shape[0] != expert_actions.shape[0]:
            pred_batch_size = predicted_actions.shape[0]
            expert_batch_size = expert_actions.shape[0]
            
            if expert_batch_size < pred_batch_size:
                # 重复专家动作以匹配预测批次大小
                repeat_times = (pred_batch_size + expert_batch_size - 1) // expert_batch_size
                expert_actions = expert_actions.repeat(repeat_times, 1)[:pred_batch_size]
                logging.debug(f"ACT BC: Repeated expert actions: {orig_expert_shape} -> {expert_actions.shape}")
            elif expert_batch_size > pred_batch_size:
                # 截断专家动作以匹配预测批次大小
                expert_actions = expert_actions[:pred_batch_size]
                logging.debug(f"ACT BC: Truncated expert actions: {orig_expert_shape} -> {expert_actions.shape}")
        
        # 最终检查形状是否匹配
        if predicted_actions.shape != expert_actions.shape:
            raise ValueError(f"Final shape mismatch after adjustments: predicted {predicted_actions.shape}, expert {expert_actions.shape}")
        
        # 计算MSE损失
        bc_loss = F.mse_loss(predicted_actions, expert_actions)
        return bc_loss
    
    def _compute_sequence_actor_loss(
        self,
        observations,
        observation_features: Tensor | None = None,
        expert_actions: Tensor | None = None,
        expert_action_sequences: Tensor | None = None,
        training_step: int = 0,
    ) -> Tensor:
        """
        计算Q-chunking序列ACT Actor的损失
        
        Q-chunking的核心改进：
        1. 使用动作序列的联合概率进行SAC优化
        2. 支持多种Q-chunking策略（标准/保守/时间加权）
        3. 考虑动作序列的时间一致性
        
        Args:
            observations: 观测数据（可能是序列）
            observation_features: 预计算的观测特征
            expert_actions: 单步专家动作
            expert_action_sequences: 专家动作序列 (batch, chunk_size, action_dim)
            training_step: 当前训练步数
            
        Returns:
            Q-chunking Actor总损失
        """
        # 1. 获取动作序列预测（联合概率）
        action_sequence, log_probs_joint, means_sequence = self.actor(
            observations, 
            observation_features, 
            return_sequence=True
        )
        
        # 2. Q-chunking SAC损失计算
        sac_actor_loss = self._compute_q_chunking_sac_loss(
            action_sequence=action_sequence,
            log_probs_joint=log_probs_joint,
            observations=observations,
            observation_features=observation_features
        )
        
        # 3. 计算序列BC损失
        bc_sequence_loss = None
        if expert_action_sequences is not None:
            # 使用完整的动作序列计算BC损失
            bc_sequence_loss = self._compute_sequence_bc_loss(
                predicted_sequence=means_sequence,  # 使用确定性预测
                expert_sequence=expert_action_sequences
            )
        elif expert_actions is not None:
            # 如果只有单步专家动作，使用第一个预测动作计算BC损失
            bc_sequence_loss = self._compute_bc_loss_act(
                predicted_actions=means_sequence[:, 0, :],  # 只使用第一个动作
                expert_actions=expert_actions
            )
        
        # 5. 混合损失计算
        if bc_sequence_loss is not None:
            bc_weight = self._compute_dynamic_bc_weight(training_step)
            sac_weight = 1.0 - bc_weight
            
            # 混合损失
            actor_loss = sac_weight * sac_actor_loss + bc_weight * bc_sequence_loss
            
            # 存储损失组件
            self._last_bc_loss = bc_sequence_loss.detach().clone()
            self._last_bc_weight = bc_weight
            self._last_sac_actor_loss = sac_actor_loss.detach().clone()
            
            # 序列特有的损失信息
            self._last_sequence_length = action_sequence.shape[1]
            self._last_joint_log_prob = log_probs_joint.mean().detach().clone()
            
            logging.debug(f"Sequence Actor Loss - SAC: {sac_actor_loss.item():.4f}, "
                         f"BC: {bc_sequence_loss.item():.4f}, "
                         f"Weight: {bc_weight:.3f}, "
                         f"Sequence Length: {self._last_sequence_length}")
        else:
            # 纯SAC损失
            actor_loss = sac_actor_loss
            self._last_bc_loss = None
            self._last_bc_weight = 0.0
            self._last_sac_actor_loss = sac_actor_loss.detach().clone()
            self._last_sequence_length = action_sequence.shape[1]
            self._last_joint_log_prob = log_probs_joint.mean().detach().clone()
        
        return actor_loss
    
    def _compute_sequence_bc_loss(self, predicted_sequence: Tensor, expert_sequence: Tensor) -> Tensor:
        """
        计算动作序列的BC损失
        
        Args:
            predicted_sequence: 预测的动作序列 (batch, chunk_size, action_dim)
            expert_sequence: 专家动作序列 (batch, chunk_size, action_dim)
            
        Returns:
            序列BC损失
        """
        # 确保形状匹配
        if predicted_sequence.shape != expert_sequence.shape:
            # 如果序列长度不匹配，截取或填充
            pred_len = predicted_sequence.shape[1]
            expert_len = expert_sequence.shape[1]
            
            if expert_len < pred_len:
                # 重复最后一个专家动作
                last_action = expert_sequence[:, -1:, :].expand(-1, pred_len - expert_len, -1)
                expert_sequence = torch.cat([expert_sequence, last_action], dim=1)
                logging.debug(f"Sequence BC: Padded expert sequence from {expert_len} to {pred_len}")
            elif expert_len > pred_len:
                # 截取专家序列
                expert_sequence = expert_sequence[:, :pred_len, :]
                logging.debug(f"Sequence BC: Truncated expert sequence from {expert_len} to {pred_len}")
        
        # 计算序列MSE损失
        # 选项1：平均所有时间步的损失
        sequence_mse = F.mse_loss(predicted_sequence, expert_sequence)
        
        # 选项2：加权损失（可选）- 给近期动作更高权重
        # weights = torch.exp(-0.1 * torch.arange(predicted_sequence.shape[1], device=predicted_sequence.device))
        # weights = weights / weights.sum()
        # weighted_mse = (weights.view(1, -1, 1) * (predicted_sequence - expert_sequence) ** 2).mean()
        
        return sequence_mse
    
    def _compute_q_chunking_sac_loss(
        self,
        action_sequence: Tensor,
        log_probs_joint: Tensor,
        observations,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """
        计算Q-chunking的SAC损失
        
        Q-chunking的核心创新：SAC损失使用整个动作序列的联合概率
        这确保策略优化时考虑动作序列的时间一致性和长期规划
        
        相比传统SAC只考虑单步动作，Q-chunking考虑的是：
        loss = E[α * log π(a₁:ₜ|s₁:ₜ) - Q(s₁, a₁)]
        其中 a₁:ₜ 是动作序列，π(a₁:ₜ|s₁:ₜ) 是联合动作概率
        
        Args:
            action_sequence: 预测的动作序列 (batch, chunk_size, action_dim)
            log_probs_joint: 动作序列的联合对数概率 (batch,)
            observations: 观测数据
            observation_features: 预计算的观测特征
            
        Returns:
            Q-chunking SAC Actor损失
        """
        # 1. 获取当前观测（用于Q值计算）
        if isinstance(observations, list):
            current_obs = observations[-1]  # 取序列的最后一个观测作为当前观测
        else:
            current_obs = observations
        
        # 2. Q-chunking策略选择
        q_chunking_strategy = getattr(self.config, 'q_chunking_strategy', 'standard')
        
        if q_chunking_strategy == 'conservative':
            # 保守策略：对序列中的多个动作计算Q值并取最小值
            sac_loss = self._compute_conservative_q_chunking_loss(
                action_sequence, log_probs_joint, current_obs, observation_features
            )
        elif q_chunking_strategy == 'temporal_weighted':
            # 时间加权策略：对不同时间步的动作给予不同权重
            sac_loss = self._compute_temporal_weighted_q_chunking_loss(
                action_sequence, log_probs_joint, current_obs, observation_features
            )
        else:
            # 标准策略：只使用第一个动作计算Q值，但使用联合概率
            sac_loss = self._compute_standard_q_chunking_loss(
                action_sequence, log_probs_joint, current_obs, observation_features
            )
        
        return sac_loss
    
    def _compute_standard_q_chunking_loss(
        self,
        action_sequence: Tensor,
        log_probs_joint: Tensor,
        current_obs,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """
        标准Q-chunking策略：使用第一个动作计算Q值，但使用序列联合概率
        
        这是Q-chunking论文中的主要方法
        """
        # 使用序列的第一个动作计算Q值
        first_action = action_sequence[:, 0, :]  # (batch, action_dim)
        
        q_preds = self.critic_forward(
            observations=current_obs,
            actions=first_action,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]
        
        # 🔥 Q-chunking核心：使用整个动作序列的联合概率
        # 这确保策略学习到时间一致的动作序列
        sac_actor_loss = ((self.temperature * log_probs_joint) - min_q_preds).mean()
        
        return sac_actor_loss
    
    def _compute_conservative_q_chunking_loss(
        self,
        action_sequence: Tensor,
        log_probs_joint: Tensor,
        current_obs,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """
        保守Q-chunking策略：对序列中的多个动作计算Q值并取最小值
        
        这提供了更保守的价值估计，可能更稳定但学习较慢
        """
        batch_size, chunk_size, action_dim = action_sequence.shape
        
        # 计算序列中每个动作的Q值
        q_values_list = []
        for t in range(min(chunk_size, getattr(self.config, 'q_chunking_horizon', 3))):
            action_t = action_sequence[:, t, :]  # (batch, action_dim)
            
            q_preds_t = self.critic_forward(
                observations=current_obs,
                actions=action_t,
                use_target=False,
                observation_features=observation_features,
            )
            min_q_t = q_preds_t.min(dim=0)[0]
            q_values_list.append(min_q_t)
        
        # 对多个时间步的Q值取最小值（保守估计）
        stacked_q_values = torch.stack(q_values_list, dim=1)  # (batch, horizon)
        conservative_q = stacked_q_values.min(dim=1)[0]  # (batch,)
        
        # 使用联合概率和保守Q值
        sac_actor_loss = ((self.temperature * log_probs_joint) - conservative_q).mean()
        
        return sac_actor_loss
    
    def _compute_temporal_weighted_q_chunking_loss(
        self,
        action_sequence: Tensor,
        log_probs_joint: Tensor,
        current_obs,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """
        时间加权Q-chunking策略：对不同时间步的动作给予不同权重
        
        近期动作权重更高，远期动作权重较低
        """
        batch_size, chunk_size, action_dim = action_sequence.shape
        horizon = min(chunk_size, getattr(self.config, 'q_chunking_horizon', 3))
        
        # 创建时间衰减权重
        decay_factor = getattr(self.config, 'q_chunking_decay', 0.9)
        weights = torch.tensor([decay_factor ** t for t in range(horizon)], 
                              device=action_sequence.device)
        weights = weights / weights.sum()  # 归一化权重
        
        # 计算序列中每个动作的加权Q值
        weighted_q_sum = 0.0
        for t in range(horizon):
            action_t = action_sequence[:, t, :]  # (batch, action_dim)
            
            q_preds_t = self.critic_forward(
                observations=current_obs,
                actions=action_t,
                use_target=False,
                observation_features=observation_features,
            )
            min_q_t = q_preds_t.min(dim=0)[0]
            weighted_q_sum += weights[t] * min_q_t
        
        # 使用联合概率和加权Q值
        sac_actor_loss = ((self.temperature * log_probs_joint) - weighted_q_sum).mean()
        
        return sac_actor_loss

    def _init_normalization(self, dataset_stats):
        """Initialize input/output normalization modules."""
        self.normalize_inputs = nn.Identity()  # 网络层占位初始化
        self.normalize_targets = nn.Identity() # 网络层占位初始化
        """
            pre-train 阶段使用config当中的dataset_stats对输入数据进行归一化
            eval 阶段使用 ./pretrained_model/config.json 当中的dataset_stats对输出数据进行归一化
        """
        if self.config.dataset_stats is not None: # 如果config当中定义了dataset_stats
            params = _convert_normalization_params_to_tensor(self.config.dataset_stats) # 将归一化参数转换为张量
            self.normalize_inputs = NormalizeBuffer(
                self.config.input_features, self.config.normalization_mapping, params
            )
            """
                优先使用dataset_stats中的min和max对输出数据进行归一化, 
                如果dataset_stats为None
                则使用params对输出数据进行归一化
            """
            stats = dataset_stats or params # 如果dataset_stats 为None，则使用params
            self.normalize_targets = NormalizeBuffer(
                self.config.output_features, self.config.normalization_mapping, stats
            )

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config, self.normalize_inputs)
        self.encoder_actor = (
            self.encoder_critic
            if self.shared_encoder
            else SACObservationEncoder(self.config, self.normalize_inputs)
        )

    def _init_critics(self, continuous_action_dim):
        """
                观测输入 → SACObservationEncoder → 观测编码 (256维)
                动作输入 → 动作归一化 → 归一化动作 (2维)
                                        ↓
                                    拼接 (258维)
                                        ↓
                                CriticHead 1: MLP(258→256→256→1) → Q1
                                        ↓
                                CriticHead 2: MLP(258→256→256→1) → Q2
                                        ↓
                                    输出: [2, batch_size] 的Q值张量    
        """
        """Build critic ensemble, targets, and optional discrete critic."""
        # 步骤1: 初始化 当前Q网络(据数量初始化多个当前Q网络)
        """
        举例:
            可以混合不同类型的评论家头
            mixed_heads = [
                CriticHead(input_dim, [256, 256]),  # 标准评论家
                CustomCriticHead(input_dim, [512, 256]),  # 自定义评论家
                LightweightCriticHead(input_dim, [128, 128])  # 轻量级评论家
            ]
            ensemble = CriticEnsemble(encoder, mixed_heads, normalization)
        """
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=heads, output_normalization=self.normalize_targets
        )
        # 步骤2: 初始化 目标Q网络(根据数量初始化多个目标Q网络)
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=target_heads, output_normalization=self.normalize_targets
        )
        # 步骤3: 将当前Q网络的参数加载到目标Q网络
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        # 步骤4: 使用torch.compile 编译当前Q网络和目标Q网络 - 编译优化
        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        # 步骤5: 初始化 离散Q网络(根据数量初始化多个离散Q网络) - 离散Q的网络用于末端执行器的抓/放
        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_discrete_critics(self):
        """Build discrete discrete critic ensemble and target networks."""
        """
            按需构建离散网络Q的网络集合, 用于末端执行器的抓/放
            离散网络Q的当前网络:
                输入: 观测编码
                输出: Q值向量
            离散网络Q的目标网络:
                输入: 观测编码
                输出: Q值向量
        """
        self.discrete_critic = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs), # 将config当中的离散Q网络的网络参数转换为字典
        )
        self.discrete_critic_target = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs), # 将config当中的离散Q网络的网络参数转换为字典
        )

        # TODO: (maractingi, azouitine) Compile the discrete critic
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_actor(self, continuous_action_dim):
        """初始化策略Actor网络和默认目标熵值。
        
        Actor网络架构说明：
        支持两种架构：
        1. 传统MLP Actor (use_act_actor=False)
        2. ACT Transformer Actor (use_act_actor=True)
        
        ACT Actor架构：
        观测输入 → SACObservationEncoder → 观测编码 
                                        ↓
                                    ACT Transformer Encoder-Decoder
                                        ↓
                                    均值层 (dim_model→action_dim) → 动作均值
                                    标准差层 (dim_model→action_dim) → 动作标准差
                                        ↓
                                    TanhMultivariateNormalDiag → 采样动作
        """
        # 检查是否使用ACT Actor
        use_act_actor = getattr(self.config, 'use_act_actor', False)
        use_sequence_actor = getattr(self.config, 'use_sequence_act_actor', False)
        
        # 🔥 调试输出：确保这些信息总是被打印
        print(f"[DEBUG ACTOR INIT] use_act_actor: {use_act_actor}")
        print(f"[DEBUG ACTOR INIT] use_sequence_act_actor: {use_sequence_actor}")
        logging.info(f"[DEBUG ACTOR INIT] use_act_actor: {use_act_actor}, use_sequence_act_actor: {use_sequence_actor}")
        
        if use_act_actor:
            # 检查是否使用序列版本            
            if use_sequence_actor:
                # 使用真正的序列ACT Actor (延迟导入以避免循环依赖)
                from lerobot.common.policies.sac.modeling_sac_sequence_act_actor import SequenceACTSACActorV2
                
                print("🚀 [PRINT] Initializing Sequence ACT-SAC Actor V2")
                logging.info("🚀 Initializing Sequence ACT-SAC Actor V2")
                
                self.actor = SequenceACTSACActorV2(
                    encoder=self.encoder_actor,
                    action_dim=continuous_action_dim,
                    # 序列参数
                    chunk_size=getattr(self.config, 'act_chunk_size', 8),
                    obs_history_length=getattr(self.config, 'obs_history_length', 5),
                    # ACT Transformer参数
                    dim_model=getattr(self.config, 'act_dim_model', 512),
                    n_heads=getattr(self.config, 'act_n_heads', 8),
                    dim_feedforward=getattr(self.config, 'act_dim_feedforward', 3200),
                    n_encoder_layers=getattr(self.config, 'act_n_encoder_layers', 4),
                    n_decoder_layers=getattr(self.config, 'act_n_decoder_layers', 4),
                    dropout=getattr(self.config, 'act_dropout', 0.1),
                    feedforward_activation=getattr(self.config, 'act_feedforward_activation', 'relu'),
                    pre_norm=getattr(self.config, 'act_pre_norm', False),
                    # SAC策略参数
                    encoder_is_shared=self.shared_encoder,
                    use_tanh_squash=getattr(self.config.policy_kwargs, 'use_tanh_squash', True),
                )
                print("✅ [PRINT] Using Sequence ACT-SAC Actor with history length: {}".format(
                    getattr(self.config, 'obs_history_length', 5)))
                logging.info("✅ Using Sequence ACT-SAC Actor with history length: {}".format(
                    getattr(self.config, 'obs_history_length', 5)))
            else:
                # 使用基础ACT Actor (延迟导入以避免循环依赖)
                from lerobot.common.policies.sac.modeling_sac_act_actor import ACTSACActor
                
                logging.info("🤖 Initializing ACT-SAC Hybrid Actor")
                
                self.actor = ACTSACActor(
                    encoder=self.encoder_actor,
                    action_dim=continuous_action_dim,
                    # ACT Transformer参数
                    dim_model=getattr(self.config, 'act_dim_model', 512),
                    n_heads=getattr(self.config, 'act_n_heads', 8),
                    dim_feedforward=getattr(self.config, 'act_dim_feedforward', 3200),
                    n_encoder_layers=getattr(self.config, 'act_n_encoder_layers', 4),
                    n_decoder_layers=getattr(self.config, 'act_n_decoder_layers', 1),
                    dropout=getattr(self.config, 'act_dropout', 0.1),
                    feedforward_activation=getattr(self.config, 'act_feedforward_activation', 'relu'),
                    pre_norm=getattr(self.config, 'act_pre_norm', False),
                    # SAC策略参数
                    encoder_is_shared=self.shared_encoder,
                    **asdict(self.config.policy_kwargs),
                    # 序列参数
                    max_seq_length=getattr(self.config, 'act_max_seq_length', 10),
                )
                logging.info("✅ Using ACT-SAC Actor")
        else:
            # 使用传统MLP Actor
            logging.info("🔧 Using traditional MLP Actor")
            self.actor = Policy(
                encoder=self.encoder_actor,  # 观测编码器，将原始观测转换为特征向量
                network=MLP(  # 主干网络：多层感知机
                    input_dim=self.encoder_actor.output_dim,  # 输入维度：观测编码的维度
                    **asdict(self.config.actor_network_kwargs),  # 网络配置参数（隐藏层维度、激活函数等）
                ),
                action_dim=continuous_action_dim,  # 动作维度：连续动作的维度
                encoder_is_shared=self.shared_encoder,  # 编码器是否在Actor和Critic之间共享
                **asdict(self.config.policy_kwargs),  # 策略配置参数（标准差范围、是否使用tanh等）
            )

        # 设置目标熵值，用于温度参数的自动调节
        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            # 如果没有指定目标熵值，则根据动作维度自动计算
            # 目标熵值 = -动作维度 / 2，这是一个经验公式
            dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -np.prod(dim) / 2

    def _init_temperature(self):
        """Set up temperature parameter and initial log_alpha."""
        # 初始化熵值
        temp_init = self.config.temperature_init
        # 初始化 log_alpha - 优化器会主要将梯度附加到log_alpha上
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)])) # 输入x>0的变量，输出log_alpha可以为正数或者负数，
        # 从 log_alpha 当中通过exp还原为temperature
        self.temperature = self.log_alpha.exp().item() # temperature 必须为一个 x > 0 的数字

    def _init_warmup_parameters(self):
        """Initialize warm-up parameters if enabled."""
        if not self.config.enable_warmup or not self.config.warmup_model_path:
            return
        
        import logging
        from .warmup_utils import WarmupParameterLoader, validate_warmup_model_path
        
        # Validate warmup model path
        validated_path = validate_warmup_model_path(self.config.warmup_model_path)
        if not validated_path:
            logging.warning(f"Invalid warm-up model path: {self.config.warmup_model_path}")
            return
        
        # Create loader and load parameters
        loader = WarmupParameterLoader()
        if not loader.load_warmup_parameters(validated_path):
            logging.error("Failed to load warm-up parameters")
            return
        
        # Apply parameters to actor
        success = loader.apply_warmup_parameters(
            target_model=self.actor,
            strict=self.config.warmup_strict_loading,
            freeze_loaded_params=self.config.warmup_freeze_loaded_params
        )
        
        if success:
            logging.info("✅ Successfully applied warm-up parameters to SAC Actor")
            
            # Log warm-up information
            warmup_info = loader.get_warmup_info()
            logging.info(f"Warm-up info: {warmup_info}")
            
            # Store warmup info in config for later reference
            if not hasattr(self.config, '_warmup_info'):
                self.config._warmup_info = warmup_info
        else:
            logging.error("❌ Failed to apply warm-up parameters to SAC Actor")


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: SACConfig, input_normalizer: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.input_normalization = input_normalizer
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()
        
        # Initialize feature visualization components
        self._init_feature_visualization()

    def _init_feature_visualization(self):
        """Initialize ROS publisher and utilities for feature visualization."""
        # Check if feature visualization is enabled in config
        if not getattr(self.config, 'enable_feature_visualization', False):
            self.enable_feature_viz = False
            self.feature_viz_enabled = False
            rospy.loginfo("-- Feature visualization disabled in config -- enable_feature_visualization: False")
            return
            
        # Determine which processes should enable feature visualization
        import os
        import sys
        
        # Determine if this should enable feature visualization
        should_enable_viz = False
        
        # Method 1: Check command line arguments for inference scripts
        cmd_line = ' '.join(sys.argv)
        if 'gym_manipulator.py' in cmd_line or 'actor.py' in cmd_line:
            should_enable_viz = True
        
        # Method 2: Check if environment variable is set (can be set by actor.py or gym_manipulator.py)
        process_type = os.environ.get('LEROBOT_PROCESS_TYPE')
        if process_type == 'actor':
            should_enable_viz = True
            
        # Method 3: Check process name/title
        try:
            import setproctitle
            proc_title = setproctitle.getproctitle().lower()
            if 'actor' in proc_title or 'gym_manipulator' in proc_title:
                should_enable_viz = True
        except ImportError:
            pass
        
        # Method 4: Check if this is an inference/evaluation context
        # If we're loading a pretrained model and not in a learner context, enable visualization
        if not should_enable_viz:
            # Check if we're in a learner process (learners typically don't need visualization)
            is_learner = ('learner.py' in cmd_line or 
                         process_type == 'learner' or
                         os.environ.get('LEROBOT_IS_LEARNER') == 'true')
            
            if not is_learner:
                # This is likely an inference/evaluation context
                should_enable_viz = True
        
        self.enable_feature_viz = should_enable_viz

        if self.enable_feature_viz and ROS_AVAILABLE and rospy is not None:
            try:
                # Initialize ROS publisher for feature visualization (using CompressedImage for better performance)
                self.feature_viz_pub = rospy.Publisher('/vision_features/resnet10_features/compressed', CompressedImage, queue_size=1, tcp_nodelay=True)
                self.feature_viz_enabled = True
                rospy.loginfo("Feature visualization enabled - publishing to /vision_features/resnet10_features/compressed")
            except Exception as e:
                rospy.logwarn(f"Failed to initialize feature visualization: {e}")
                self.feature_viz_enabled = False
        else:
            self.feature_viz_enabled = False
            rospy.loginfo("-- Feature visualization disabled -- [if self.enable_feature_viz and ROS_AVAILABLE and rospy is not None] ")

    def _visualize_features(self, features: Tensor, image_key: str = "observation.image.front"):
        """
        Convert CNN features to visualization image and publish to ROS topic.
        
        Args:
            features: Feature tensor from ResNet10, shape (B, C, H, W) or (B, feature_dim)
            image_key: Key identifying which image these features come from
        """
        if not self.feature_viz_enabled:
            return
            
        try:
            with torch.no_grad():
                # Handle different feature tensor shapes
                if len(features.shape) == 4:  # (B, C, H, W) - spatial features
                    # Take first batch item and convert to numpy
                    feat = features[0].cpu().numpy()  # (C, H, W)
                    
                    # Create feature map visualization
                    # Method 1: Average across channels to create a single heatmap
                    feat_avg = np.mean(feat, axis=0)  # (H, W)
                    
                    # Normalize to 0-255
                    feat_norm = ((feat_avg - feat_avg.min()) / 
                               (feat_avg.max() - feat_avg.min() + 1e-8) * 255).astype(np.uint8)
                    
                    # Convert grayscale to RGB for visualization
                    feat_rgb = np.stack([feat_norm, feat_norm, feat_norm], axis=-1)  # (H, W, 3)
                    
                    # Resize to reasonable size for visualization (224x224)
                    import cv2
                    feat_rgb = cv2.resize(feat_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
                    
                elif len(features.shape) == 2:  # (B, feature_dim) - flattened features
                    # For flattened features, create a simple visualization
                    feat = features[0].cpu().numpy()  # (feature_dim,)
                    
                    # Reshape to a square-ish grid for visualization
                    grid_size = int(np.sqrt(len(feat)))
                    if grid_size * grid_size < len(feat):
                        grid_size += 1
                    
                    # Pad and reshape
                    padded_feat = np.pad(feat, (0, grid_size * grid_size - len(feat)), 'constant')
                    feat_grid = padded_feat.reshape(grid_size, grid_size)
                    
                    # Normalize to 0-255
                    feat_norm = ((feat_grid - feat_grid.min()) / 
                               (feat_grid.max() - feat_grid.min() + 1e-8) * 255).astype(np.uint8)
                    
                    # Convert to RGB and resize
                    feat_rgb = np.stack([feat_norm, feat_norm, feat_norm], axis=-1)
                    import cv2
                    feat_rgb = cv2.resize(feat_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
                
                else:
                    rospy.logwarn(f"Unsupported feature shape for visualization: {features.shape}")
                    return
                
                # Publish as ROS CompressedImage message
                ros_image = CompressedImage()
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = f"features_{image_key.replace('.', '_')}"
                ros_image.format = "jpeg"
                
                # Encode image to JPEG format
                _, encoded_img = cv2.imencode('.jpg', feat_rgb)
                ros_image.data = encoded_img.tobytes()
                
                self.feature_viz_pub.publish(ros_image)
                
                # Log occasionally for debugging
                if hasattr(self, '_viz_counter'):
                    self._viz_counter += 1
                else:
                    self._viz_counter = 1
                    
                if self._viz_counter % 30 == 1:  # Log every 30 frames
                    rospy.loginfo(f"Published feature visualization - shape: {features.shape}, "
                                f"feature range: [{feat.min():.3f}, {feat.max():.3f}], "
                                f"viz range: [0, 255]")
                
        except Exception as e:
            rospy.logwarn(f"Error in feature visualization: {e}")

    def _init_image_layers(self) -> None:
        # If the config explicitly disables vision features, skip image processing entirely
        if getattr(self.config, 'disable_vision_features', False):
            self.has_images = False
            self.image_keys = []
            self.image_encoder = None
            self.spatial_embeddings = nn.ModuleDict()
            self.post_encoders = nn.ModuleDict()
            return

        # If the config clearly indicates no vision, just exit.
        # This is a stronger check that relies on explicit config values.
        if self.config.vision_encoder_name is None and self.config.image_encoder_hidden_dim == 0:
            self.has_images = False
            self.image_keys = []
            self.image_encoder = None
            self.spatial_embeddings = nn.ModuleDict()
            self.post_encoders = nn.ModuleDict()
            return

        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            # Set image-related attributes to None or empty to avoid errors
            self.image_encoder = None
            self.spatial_embeddings = nn.ModuleDict()
            self.post_encoders = nn.ModuleDict()
            return

        if self.config.vision_encoder_name is not None:
            self.image_encoder = PretrainedImageEncoder(self.config)
        else:
            self.image_encoder = DefaultImageEncoder(self.config)

        if self.config.freeze_vision_encoder:
            freeze_image_encoder(self.image_encoder)

        dummy = torch.zeros(1, *self.config.input_features[self.image_keys[0]].shape)
        with torch.no_grad():
            _, channels, height, width = self.image_encoder(dummy).shape

        self.spatial_embeddings = nn.ModuleDict()
        self.post_encoders = nn.ModuleDict()

        for key in self.image_keys:
            name = key.replace(".", "_")
            self.spatial_embeddings[name] = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.config.image_embedding_pooling_dim,
            )
            self.post_encoders[name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    in_features=channels * self.config.image_embedding_pooling_dim,
                    out_features=self.config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=self.config.latent_dim),
                nn.Tanh(),
            )

    def _init_state_layers(self) -> None:
        self.has_env = "observation.environment_state" in self.config.input_features
        self.has_state = "observation.state" in self.config.input_features
        if self.has_env:
            dim = self.config.input_features["observation.environment_state"].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        if self.has_state:
            dim = self.config.input_features["observation.state"].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images:
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env:
            out += self.config.latent_dim
        if self.has_state:
            out += self.config.latent_dim
        self._out_dim = out

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        obs = self.input_normalization(obs)
        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs, normalize=False) # 输入obs，返回缓存ResNet10的特征
            parts.append(self._encode_images(cache, detach)) # 编码图像
        if self.has_env: # 如果config当中定义了observation.environment_state
            parts.append(self.env_encoder(obs["observation.environment_state"])) # 编码环境状态
        if self.has_state: # 如果config当中定义了observation.state
            parts.append(self.state_encoder(obs["observation.state"])) # 编码状态
        if parts:
            return torch.cat(parts, dim=-1) # 将所有部分拼接在一起

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state"
        )

    def get_cached_image_features(self, obs: dict[str, Tensor], normalize: bool = False) -> dict[str, Tensor]:
        """Extract and optionally cache image features from observations.

        This function processes image observations through the vision encoder once and returns
        the resulting features.
        When the image encoder is shared between actor and critics AND frozen, these features can be safely cached and
        reused across policy components (actor, critic, discrete_critic), avoiding redundant forward passes.

        Performance impact:
        - The vision encoder forward pass is typically the main computational bottleneck during training and inference
        - Caching these features can provide 2-4x speedup in training and inference

        Normalization behavior:
        - When called from inside forward(): set normalize=False since inputs are already normalized
        - When called from outside forward(): set normalize=True to ensure proper input normalization

        Usage patterns:
        - Called in select_action() with normalize=True
        - Called in learner.py's get_observation_features() to pre-compute features for all policy components
        - Called internally by forward() with normalize=False

        Args:
            obs: Dictionary of observation tensors containing image keys
            normalize: Whether to normalize observations before encoding
                  Set to True when calling directly from outside the encoder's forward method
                  Set to False when calling from within forward() where inputs are already normalized

        Returns:
            Dictionary mapping image keys to their corresponding encoded features
        """
        # Early return if vision features are disabled
        if not self.has_images or len(self.image_keys) == 0 or self.image_encoder is None:
            return {}
        
        if normalize:
            obs = self.input_normalization(obs) # 归一化图像
        batched = torch.cat([obs[k] for k in self.image_keys], dim=0) # 🔥 关键步骤：只提取图像键对应的数据, 同时拼接在一起
        out = self.image_encoder(batched)
        
        # Add feature visualization here
        if self.feature_viz_enabled and len(self.image_keys) > 0:
            # Visualize features for the first image key
            first_key = self.image_keys[0]
            first_image_features = out[:1]  # Take features for first image only
            self._visualize_features(first_image_features, first_key)
        
        chunks = torch.chunk(out, len(self.image_keys), dim=0) # 将输出分割为多个小块
        return dict(zip(self.image_keys, chunks, strict=False)) # 返回字典，键为图像键，值为小块

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """Encode image features from cached observations.

        This function takes pre-encoded image features from the cache and applies spatial embeddings and post-encoders.
        It also supports detaching the encoded features if specified.

        Args:
            cache (dict[str, Tensor]): The cached image features.
            detach (bool): Usually when the encoder is shared between actor and critics,
            we want to detach the encoded features on the policy side to avoid backprop through the encoder.
            More detail here `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`

        Returns:
            Tensor: The encoded image features.
        
        cache:缓存ResNet10的特征 
        更多细节见 `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`

        detach: 当编码器在actor和critic之间共享时, 我们希望在policy actor侧分离编码器以避免通过编码器进行反向传播,但是可以编码器只通过 critic 的梯度更新
        更多细节见 `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`
        """
        feats = [] # 存储编码后的特征
        for k, feat in cache.items(): # 遍历缓存中的特征
            safe_key = k.replace(".", "_") # 将特征键中的点替换为下划线
            x = self.spatial_embeddings[safe_key](feat) # 将特征通过空间学习嵌入
            x = self.post_encoders[safe_key](x) # 保持梯度传播
            if detach:
                x = x.detach() # 如果detach为True，则将特征分离
            feats.append(x) # 将编码后的特征添加到列表中
        return torch.cat(feats, dim=-1) # 将所有特征拼接在一起

    @property
    def output_dim(self) -> int:
        """
            输出观测编码的维度
        """
        return self._out_dim


class MLP(nn.Module):
    """Multi-layer perceptron builder.

    Dynamically constructs a sequence of layers based on `hidden_dims`:
      1) Linear (in_dim -> out_dim)
      2) Optional Dropout if `dropout_rate` > 0 and (not final layer or `activate_final`)
      3) LayerNorm on the output features
      4) Activation (standard for intermediate layers, `final_activation` for last layer if `activate_final`)

    Arguments:
        input_dim (int): Size of input feature dimension.
        hidden_dims (list[int]): Sizes for each hidden layer.
        activations (Callable or str): Activation to apply between layers.
        activate_final (bool): Whether to apply activation at the final layer.
        dropout_rate (Optional[float]): Dropout probability applied before normalization and activation.
        final_activation (Optional[Callable or str]): Activation for the final layer when `activate_final` is True.

    For each layer, `in_dim` is updated to the previous `out_dim`. All constructed modules are
    stored in `self.net` as an `nn.Sequential` container.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = idx == total - 1
            # 2-4) optionally add dropout, normalization, and activation
            if not is_last or activate_final:
                if dropout_rate and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(out_dim))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticHead(nn.Module):
    """
    职责: 实现单个Q值网络的逻辑
    输入: 观测编码 + 归一化动作
    输出: Q值
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class CriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        output_normalization (nn.Module): normalization layer for actions.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.

    管理多个CriticHead模块的集合
    职责: 管理多个CriticHead模块的集合
    输入: 观测编码 + 归一化动作
    输出: Q值向量
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[CriticHead],
        output_normalization: nn.Module,
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.output_normalization = output_normalization
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}
        # NOTE: We normalize actions it helps for sample efficiency
        actions: dict[str, torch.tensor] = {"action": actions}
        # NOTE: Normalization layer took dict in input and outputs a dict that why
        actions = self.output_normalization(actions)["action"]
        actions = actions.to(device)

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


class DiscreteCritic(nn.Module):
    """
        离散Q网络的评论家头
        职责: 实现单个离散Q值网络的逻辑
        输入: 观测编码
        输出: Q值
    """
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.encoder = encoder # 观测编码器
        self.output_dim = output_dim # 输出层的大小，针对目标的动作维度（比如Franka 抓/放/保持 三个动作）

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )

        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim) # 网络的最后一层，将前面MLP网络提取的特征映射为最终的输出
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, observations: torch.Tensor, observation_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        obs_enc = self.encoder(observations, cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    """
    SAC策略网络（Actor网络）
    
    网络架构：
    1. 观测编码器 (encoder): 将原始观测转换为特征向量
    2. 主干网络 (network): 多层感知机，处理编码后的观测特征
    3. 均值层 (mean_layer): 输出动作的均值
    4. 标准差层 (std_layer): 输出动作的标准差（用于探索）
    
    前向传播流程：
    观测输入 → 编码器 → 观测特征 → 主干网络 → 特征向量
                                        ↓
                                    均值层 → 动作均值
                                    标准差层 → 动作标准差
                                        ↓
                                    TanhMultivariateNormalDiag → 采样动作
    
    输出：
    - actions: 采样的动作
    - log_probs: 动作的对数概率
    - means: 动作的均值（用于确定性动作选择）
    """
    def __init__(
        self,
        encoder: SACObservationEncoder,  # 观测编码器
        network: nn.Module,  # 主干网络（通常是MLP）
        action_dim: int,  # 动作维度
        std_min: float = -5,  # 标准差的最小值（log空间）
        std_max: float = 2,  # 标准差的最大值（log空间）
        fixed_std: torch.Tensor | None = None,  # 固定的标准差（如果为None则学习）
        init_final: float | None = None,  # 最终层的初始化参数
        use_tanh_squash: bool = False,  # 是否使用tanh压缩动作
        encoder_is_shared: bool = False,  # 编码器是否与Critic共享
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder  # 观测编码器
        self.network = network  # 主干网络
        self.action_dim = action_dim  # 动作维度
        self.std_min = std_min  # 标准差最小值
        self.std_max = std_max  # 标准差最大值
        self.fixed_std = fixed_std  # 固定标准差
        self.use_tanh_squash = use_tanh_squash  # 是否使用tanh压缩
        self.encoder_is_shared = encoder_is_shared  # 编码器是否共享

        # 找到主干网络最后一个线性层的输出维度
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        
        # 均值层：将主干网络的输出映射为动作均值
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            # 使用均匀分布初始化
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            # 使用正交初始化
            orthogonal_init()(self.mean_layer.weight)

        # 标准差层：将主干网络的输出映射为动作标准差
        if fixed_std is None:
            # 如果使用学习型标准差
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                # 使用均匀分布初始化
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                # 使用正交初始化
                orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,  # 观测输入
        observation_features: torch.Tensor | None = None,  # 预计算的观测特征
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            observations: 观测字典
            observation_features: 预计算的观测特征（用于缓存）
            
        Returns:
            actions: 采样的动作
            log_probs: 动作的对数概率
            means: 动作的均值
        """
        # 如果编码器是共享的，则分离梯度以避免通过编码器进行反向传播
        # 这很重要，可以避免编码器通过策略网络更新
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # 获取主干网络的输出
        outputs = self.network(obs_enc)
        
        # 计算动作均值
        means = self.mean_layer(outputs)

        # 计算动作标准差
        if self.fixed_std is None:
            # 使用学习型标准差
            log_std = self.std_layer(outputs)  # 输出log标准差
            std = torch.exp(log_std)  # 转换为标准差
            std = torch.clamp(std, self.std_min, self.std_max)  # 裁剪到指定范围
        else:
            # 使用固定标准差
            std = self.fixed_std.expand_as(means)

        # 🔥 NaN检测和修复：检查输入的均值和标准差是否包含NaN
        if torch.isnan(means).any() or torch.isnan(std).any():
            import logging
            logging.error("[ACTOR] NaN detected in action distribution parameters!")
            logging.error(f"means contains NaN: {torch.isnan(means).any().item()}")
            logging.error(f"std contains NaN: {torch.isnan(std).any().item()}")
            
            # 使用零均值和小的固定标准差作为fallback
            means = torch.zeros_like(means)  # 零均值
            std = torch.ones_like(std) * 0.01  # 小的固定标准差
            logging.warning("[ACTOR] Using fallback values: zero means and 0.01 std")

        # 确保标准差在合理范围内（数值稳定性）
        std = torch.clamp(std, min=1e-6, max=10.0)

        # 构建变换分布：使用tanh变换的多元正态分布
        try:
            dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
            # 采样动作（使用重参数化技巧）
            actions = dist.rsample()
            # 计算动作的对数概率
            log_probs = dist.log_prob(actions)
            
            # 检查采样结果是否包含NaN
            if torch.isnan(actions).any() or torch.isnan(log_probs).any():
                import logging
                logging.error("[ACTOR] NaN detected in sampled actions or log_probs")
                # 使用零动作和很小的log_prob作为fallback
                actions = torch.zeros_like(actions)
                log_probs = torch.full_like(log_probs, -1000.0)  # 很小的概率
                
        except Exception as e:
            import logging
            logging.error(f"[ACTOR] Error creating distribution: {e}")
            # 创建fallback动作和概率
            batch_size = means.shape[0]
            action_dim = means.shape[1]
            actions = torch.zeros((batch_size, action_dim), device=means.device, dtype=means.dtype)
            log_probs = torch.full((batch_size,), -1000.0, device=means.device, dtype=means.dtype)

        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """获取观测的编码特征"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


class DefaultImageEncoder(nn.Module):
    def __init__(self, config: SACConfig):
        super().__init__()
        image_key = next(key for key in config.input_features if is_image_feature(key))
        self.image_enc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_features[image_key].shape[0],
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=7,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.image_enc_layers(x)
        return x


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: SACConfig):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: SACConfig):
        """Set up CNN encoder"""
        from transformers import AutoModel

        self.image_enc_layers = AutoModel.from_pretrained(config.vision_encoder_name, trust_remote_code=True)

        if hasattr(self.image_enc_layers.config, "hidden_sizes"):
            self.image_enc_out_shape = self.image_enc_layers.config.hidden_sizes[-1]  # Last channel dimension
        elif hasattr(self.image_enc_layers, "fc"):
            self.image_enc_out_shape = self.image_enc_layers.fc.in_features
        else:
            raise ValueError("Unsupported vision encoder architecture, make sure you are using a CNN")
        return self.image_enc_layers, self.image_enc_out_shape

    def forward(self, x):
        enc_feat = self.image_enc_layers(x).last_hidden_state # 🔥 这里调用 ResNet10
        return enc_feat


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, C, H, W] where B is batch size,
                     C is number of channels, H is height, and W is width
        Returns:
            Output tensor of shape [B, C*F] where F is the number of features
        """

        features_expanded = features.unsqueeze(-1)  # [B, C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum over H,W dimensions

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        return output


class RescaleFromTanh(Transform):
    """
    从Tanh范围重新缩放到指定范围的变换
    
    这个变换用于将动作从Tanh的[-1, 1]范围重新缩放到动作空间的实际范围[low, high]。
    
    变换公式：
    - 前向变换：y = 0.5 * (x + 1.0) * (high - low) + low
    - 反向变换：x = 2.0 * (y - low) / (high - low) - 1.0
    
    作用：
    - 将标准化的动作范围映射到实际的动作空间
    - 保持变换的可逆性
    - 提供正确的雅可比行列式用于概率计算
    """
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()

        self.low = low  # 动作空间的下界
        self.high = high  # 动作空间的上界

    def _call(self, x):
        """前向变换：从[-1, 1]重新缩放到[low, high]"""
        # 重新缩放公式：y = 0.5 * (x + 1.0) * (high - low) + low
        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        """反向变换：从[low, high]重新缩放到[-1, 1]"""
        # 反向缩放公式：x = 2.0 * (y - low) / (high - low) - 1.0
        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        """计算雅可比行列式的对数绝对值"""
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))
        scale = 0.5 * (self.high - self.low)

        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    """
    Tanh变换的多元正态分布
    
    这个分布用于SAC中的动作采样，它结合了：
    1. 基础分布：多元正态分布（对角协方差矩阵）
    2. 变换：Tanh变换，将动作压缩到[-1, 1]范围
    3. 可选变换：从[-1, 1]重新缩放到动作空间的实际范围
    
    作用：
    - 确保动作在有效范围内
    - 提供平滑的动作分布
    - 支持重参数化采样（用于梯度计算）
    
    使用场景：
    - SAC算法中的动作采样
    - 连续动作空间的策略网络
    """
    def __init__(self, loc, scale_diag, low=None, high=None):
        # 创建基础分布：多元正态分布（对角协方差矩阵）
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))

        # 定义变换序列
        transforms = [TanhTransform(cache_size=1)]  # Tanh变换

        # 如果指定了动作范围，添加重新缩放变换
        if low is not None and high is not None:
            low = torch.as_tensor(low)
            high = torch.as_tensor(high)
            # 在Tanh变换之前插入重新缩放变换
            transforms.insert(0, RescaleFromTanh(low, high))

        # 创建变换分布
        super().__init__(base_dist, transforms)

    def mode(self):
        """获取分布的众数（最可能的动作）"""
        # 众数是基础分布的均值经过变换后的结果
        x = self.base_dist.mean

        # 依次应用所有变换
        for transform in self.transforms:
            x = transform(x)

        return x

    def stddev(self):
        """获取变换后分布的标准差"""
        std = self.base_dist.stddev

        x = std

        # 依次应用所有变换
        for transform in self.transforms:
            x = transform(x)

        return x


def _convert_normalization_params_to_tensor(normalization_params: dict) -> dict:
    converted_params = {}
    for outer_key, inner_dict in normalization_params.items():
        converted_params[outer_key] = {}
        for key, value in inner_dict.items():
            converted_params[outer_key][key] = torch.tensor(value)
            if "image" in outer_key:
                # 图像数据需要特殊处理：将形状调整为 (3, 1, 1) 以便广播
                converted_params[outer_key][key] = converted_params[outer_key][key].view(3, 1, 1)

    return converted_params

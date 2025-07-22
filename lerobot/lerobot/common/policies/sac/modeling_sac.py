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
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rospy = None
    Image = None
    CvBridge = None
    CvBridgeError = None


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
        actions: Tensor = batch["action"]
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor = batch.get("observation_feature")

        if model == "critic":
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
            return {
                "loss_actor": self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
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
    ) -> Tensor:
        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.actor(next_observations, next_observation_features)

            # 2- compute q targets
            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critics subsample size
            min_q, _ = q_targets.min(dim=0)  # Get values from min operation
            if self.config.use_backup_entropy:
                min_q = min_q - (self.temperature * next_log_probs)

            td_target = rewards + (1 - done) * self.config.discount * min_q

        # 3- compute predicted qs
        if self.config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()
        return critics_loss

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
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.actor(observations, observation_features)
        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        actions_pi, log_probs, _ = self.actor(observations, observation_features)

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        return actor_loss

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
        1. 观测编码器 (SACObservationEncoder): 将原始观测转换为特征向量
        2. 主干网络 (MLP): 多层感知机，处理编码后的观测特征
        3. 均值层 (mean_layer): 输出动作的均值
        4. 标准差层 (std_layer): 输出动作的标准差（用于探索）
        
        网络流程：
        观测输入 → SACObservationEncoder → 观测编码 (256维)
                                        ↓
                                    MLP主干网络 (256→256→256)
                                        ↓
                                    均值层 (256→action_dim) → 动作均值
                                    标准差层 (256→action_dim) → 动作标准差
                                        ↓
                                    TanhMultivariateNormalDiag → 采样动作
        """
        # 注意：Actor只选择连续动作部分，离散动作由离散Critic处理
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
        self.enable_feature_viz = True

        if self.enable_feature_viz and ROS_AVAILABLE and rospy is not None:
            try:
                # Initialize ROS publisher for feature visualization
                self.feature_viz_pub = rospy.Publisher('/vision_features/resnet10_features', Image, queue_size=1, tcp_nodelay=True)
                self.cv_bridge = CvBridge()
                self.feature_viz_enabled = True
                rospy.loginfo("Feature visualization enabled - publishing to /vision_features/resnet10_features")
            except Exception as e:
                rospy.logwarn(f"Failed to initialize feature visualization: {e}")
                self.feature_viz_enabled = False
        else:
            self.feature_viz_enabled = False

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
                
                # Publish as ROS Image message
                ros_image = self.cv_bridge.cv2_to_imgmsg(feat_rgb, "rgb8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = f"features_{image_key.replace('.', '_')}"
                
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

        # 构建变换分布：使用tanh变换的多元正态分布
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        # 采样动作（使用重参数化技巧）
        actions = dist.rsample()

        # 计算动作的对数概率
        log_probs = dist.log_prob(actions)

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

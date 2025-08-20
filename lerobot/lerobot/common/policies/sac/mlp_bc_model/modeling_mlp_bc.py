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
from lerobot.common.policies.sac.mlp_bc_model.configuration_mlp_bc import MLPBCConfig, is_image_feature
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.policies.sac.modeling_sac import SACObservationEncoder, TanhMultivariateNormalDiag, orthogonal_init, _convert_normalization_params_to_tensor

class MLPBCPolicy(PreTrainedPolicy):
    config_class = MLPBCConfig
    name = "mlp_bc"

    def __init__(
        self,
        config: MLPBCConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        # 通过配置文件初始化父类
        super().__init__(config)
        # 验证输入输出特征
        config.validate_features() 
        self.config = config

        # Determine action dimension and initialize all components
        action_feature = config.output_features["action"]
        if hasattr(action_feature, 'shape'):
            continuous_action_dim = action_feature.shape[0]  # 🔥 获取连续动作维度
        else:
            # Handle case where shape is a list/dict from JSON config
            continuous_action_dim = action_feature["shape"][0]  # 🔥 获取连续动作维度
        self._init_normalization(dataset_stats) # 🔥 初始化归一化, 通过 dataset_stats 中的 min 和 max 对输入数据进行归一化
        self._init_encoders() # 🔥 初始化编码器
        self._init_actor(continuous_action_dim) # 🔥 初始化actor

        # Initialize member variable to store td_target for wandb logging
        self.last_td_target: Tensor | None = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for behavior cloning training.
        
        Args:
            observations: Dictionary containing observation tensors
            
        Returns:
            predicted_actions: Predicted actions from the policy
        """
        # Get predicted actions using the actor network
        predicted_actions, _, _ = self.actor(observations)
        return predicted_actions

    def compute_loss(
        self, 
        observations: dict[str, torch.Tensor], 
        target_actions: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute behavior cloning loss and metrics.
        
        Args:
            observations: Dictionary containing observation tensors
            target_actions: Target actions from expert demonstrations
            
        Returns:
            loss: Computed behavior cloning loss
            metrics: Dictionary containing training metrics
        """
        # Forward pass to get predicted actions
        predicted_actions = self.forward(observations)
        
        # Compute loss using config
        loss = self.config.compute_bc_loss(predicted_actions, target_actions)
        
        # Compute additional metrics
        metrics = self.config.compute_metrics(predicted_actions, target_actions)
        
        return loss, metrics

    def predict(self, observations: dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """
        Predict actions for given observations (inference mode).
        
        Args:
            observations: Dictionary containing observation tensors
            deterministic: If True, return mean actions; if False, sample from distribution
            
        Returns:
            predicted_actions: Predicted actions
        """
        self.eval()
        with torch.no_grad():
            if deterministic:
                # Return mean actions for deterministic behavior
                predicted_actions = self.forward(observations)
            else:
                # Sample from the policy distribution
                actions, _, _ = self.actor(observations)
                predicted_actions = actions
        return predicted_actions

    def select_action(self, observations: dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """
        Select action for given observations (for environment interaction).
        This is the main method used during policy execution.
        
        Args:
            observations: Dictionary containing observation tensors
            deterministic: If True, return mean actions; if False, sample from distribution
            
        Returns:
            selected_actions: Selected actions for environment interaction
        """
        return self.predict(observations, deterministic=deterministic)

    def reset(self) -> None:
        """
        Reset the policy state. For MLP BC, there's no internal state to reset,
        but this method is required by the PreTrainedPolicy interface.
        """
        pass

    def get_optim_params(self) -> dict:
        """
        Get optimization parameters for the policy.
        Returns a dictionary mapping parameter groups to their parameters.
        """
        return {
            "policy": list(self.parameters())
        }

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
        """Initialize encoder for MLP BC policy."""
        # For MLP BC, we don't create any encoders at this level
        # All encoding is handled inside the Policy (actor)
        self.shared_encoder = self.config.shared_encoder
        # Don't create any encoder attributes to avoid duplication
        # self.encoder_actor and self.encoder_critic will be None

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
        # Get network kwargs as dict (handle both dataclass and dict)
        if hasattr(self.config.actor_network_kwargs, '__dict__'):
            actor_network_kwargs = asdict(self.config.actor_network_kwargs)
        else:
            actor_network_kwargs = self.config.actor_network_kwargs if isinstance(self.config.actor_network_kwargs, dict) else {}
            
        # Get policy kwargs as dict (handle both dataclass and dict)
        if hasattr(self.config.policy_kwargs, '__dict__'):
            policy_kwargs = asdict(self.config.policy_kwargs)
        else:
            policy_kwargs = self.config.policy_kwargs if isinstance(self.config.policy_kwargs, dict) else {}
            
        # Let Policy create and manage its own encoder completely
        # Don't store any encoder reference at MLPBCPolicy level
        
        # Calculate input_dim first
        temp_encoder = SACObservationEncoder(self.config, self.normalize_inputs)
        input_dim = temp_encoder.output_dim
        del temp_encoder  # Clean up temporary encoder
        
        self.actor = Policy(
            encoder=SACObservationEncoder(self.config, self.normalize_inputs),  # Policy owns the encoder
            network=MLP(  # 主干网络：多层感知机
                input_dim=input_dim,  # 输入维度：观测编码的维度
                **actor_network_kwargs,  # 网络配置参数（隐藏层维度、激活函数等）
            ),
            action_dim=continuous_action_dim,  # 动作维度：连续动作的维度
            encoder_is_shared=False,  # For BC, we don't share encoders (no critic)
            **policy_kwargs,  # 策略配置参数（标准差范围、是否使用tanh等）
        )
        
        # DO NOT store encoder reference to avoid duplication
        # self.encoder_actor = None  # Explicitly set to None

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
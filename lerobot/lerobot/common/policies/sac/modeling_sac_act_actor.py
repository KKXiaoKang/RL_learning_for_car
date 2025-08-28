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

"""
ACT-SAC混合Actor网络
将ACT的Transformer架构与SAC的强化学习目标结合
"""

import logging
import math
from typing import Callable

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, TransformedDistribution

from lerobot.common.policies.act.modeling_act import (
    ACTEncoder, ACTDecoder, ACTEncoderLayer, ACTDecoderLayer,
    create_sinusoidal_pos_embedding, ACTSinusoidalPositionEmbedding2d,
    get_activation_fn
)
from lerobot.common.policies.sac.modeling_sac import (
    SACObservationEncoder, TanhMultivariateNormalDiag, 
    orthogonal_init
)
from lerobot.common.policies.utils import get_device_from_parameters


class ACTSACActor(nn.Module):
    """
    结合ACT和SAC的混合Actor网络
    
    架构设计：
    1. 使用ACT的Transformer编码器处理观测序列
    2. 输出单个动作（而非动作序列）以兼容SAC
    3. 支持SAC的概率分布输出和重参数化技巧
    4. 保持ACT的强序列建模能力
    
    核心特点：
    - 使用ACT的encoder-decoder架构
    - 输出动作均值和标准差用于SAC的随机策略
    - 支持图像和状态观测的联合编码
    - 兼容SAC的Q-value估计和BC损失
    """
    
    def __init__(
        self,
        encoder: SACObservationEncoder,  # SAC观测编码器
        action_dim: int,  # 动作维度
        # ACT Transformer参数
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        dropout: float = 0.1,
        feedforward_activation: str = "relu",
        pre_norm: bool = False,
        # SAC策略参数
        std_min: float = 1e-5,
        std_max: float = 10.0,
        init_final: float = 0.05,
        use_tanh_squash: bool = True,
        encoder_is_shared: bool = False,
        # 序列参数
        max_seq_length: int = 10,  # 最大序列长度（用于位置编码）
    ):
        super().__init__()
        
        self.encoder = encoder
        self.action_dim = action_dim
        self.dim_model = dim_model
        self.std_min = std_min
        self.std_max = std_max
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared
        self.max_seq_length = max_seq_length
        
        # 创建ACT配置对象（用于初始化Transformer组件）
        self.act_config = self._create_act_config(
            dim_model, n_heads, dim_feedforward, n_encoder_layers, 
            n_decoder_layers, dropout, feedforward_activation, pre_norm
        )
        
        # 观测特征到Transformer输入的投影层
        self.obs_to_transformer_proj = nn.Linear(encoder.output_dim, dim_model)
        
        # ACT Transformer Encoder
        self.transformer_encoder = ACTEncoder(self.act_config, is_vae_encoder=False)
        
        # ACT Transformer Decoder
        self.transformer_decoder = ACTDecoder(self.act_config)
        
        # 位置编码
        self.register_buffer(
            "encoder_pos_embed",
            create_sinusoidal_pos_embedding(max_seq_length, dim_model).unsqueeze(1)  # (seq, 1, dim)
        )
        self.register_buffer(
            "decoder_pos_embed", 
            create_sinusoidal_pos_embedding(1, dim_model).unsqueeze(1)  # 只需要1个输出位置
        )
        
        # 动作输出层（SAC风格）
        self.mean_layer = nn.Linear(dim_model, action_dim)
        self.std_layer = nn.Linear(dim_model, action_dim)
        
        # 初始化输出层
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
            nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)
            orthogonal_init()(self.std_layer.weight)
    
    def _create_act_config(self, dim_model, n_heads, dim_feedforward, 
                          n_encoder_layers, n_decoder_layers, dropout, 
                          feedforward_activation, pre_norm):
        """创建ACT配置对象"""
        class SimpleACTConfig:
            def __init__(self):
                self.dim_model = dim_model
                self.n_heads = n_heads
                self.dim_feedforward = dim_feedforward
                self.n_encoder_layers = n_encoder_layers
                self.n_decoder_layers = n_decoder_layers
                self.dropout = dropout
                self.feedforward_activation = feedforward_activation
                self.pre_norm = pre_norm
        
        return SimpleACTConfig()
    
    def forward(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        前向传播
        
        Args:
            observations: 观测字典
            observation_features: 预计算的观测特征
            
        Returns:
            actions: 采样的动作 (batch_size, action_dim)
            log_probs: 动作的对数概率 (batch_size,)
            means: 动作的均值 (batch_size, action_dim)
        """
        device = get_device_from_parameters(self)
        batch_size = next(iter(observations.values())).shape[0]
        
        # 1. 编码观测
        if self.encoder_is_shared:
            obs_features = self.encoder(observations, cache=observation_features, detach=True)
        else:
            obs_features = self.encoder(observations, cache=observation_features, detach=False)
        
        # 2. 投影到Transformer空间
        transformer_input = self.obs_to_transformer_proj(obs_features)  # (batch, dim_model)
        
        # 3. 为Transformer准备序列格式
        # 这里我们将单个观测扩展为序列（可以扩展为处理观测历史）
        transformer_input = transformer_input.unsqueeze(0)  # (1, batch, dim_model) - seq_len=1
        
        # 4. 添加位置编码
        pos_embed = self.encoder_pos_embed[:1]  # 取第一个位置编码 (1, 1, dim_model)
        pos_embed = pos_embed.expand(-1, batch_size, -1)  # (1, batch, dim_model)
        
        # 5. Transformer Encoder
        encoder_out = self.transformer_encoder(
            transformer_input, 
            pos_embed=pos_embed
        )  # (1, batch, dim_model)
        
        # 6. Transformer Decoder
        # 为decoder准备输入（零初始化）
        decoder_input = torch.zeros(
            (1, batch_size, self.dim_model),
            device=device,
            dtype=transformer_input.dtype
        )  # (1, batch, dim_model)
        
        decoder_pos_embed = self.decoder_pos_embed[:1].expand(-1, batch_size, -1)  # (1, batch, dim_model)
        
        decoder_out = self.transformer_decoder(
            decoder_input,
            encoder_out,
            decoder_pos_embed=decoder_pos_embed,
            encoder_pos_embed=pos_embed
        )  # (1, batch, dim_model)
        
        # 7. 提取输出特征
        output_features = decoder_out.squeeze(0)  # (batch, dim_model)
        
        # 8. 计算动作分布参数
        means = self.mean_layer(output_features)  # (batch, action_dim)
        log_stds = self.std_layer(output_features)  # (batch, action_dim)
        
        # 限制标准差范围
        log_stds = torch.clamp(log_stds, self.std_min, self.std_max)
        stds = torch.exp(log_stds)
        
        # 9. 构建动作分布并采样
        if self.use_tanh_squash:
            # 使用Tanh变换的多元正态分布（SAC标准做法）
            dist = TanhMultivariateNormalDiag(loc=means, scale_diag=stds)
        else:
            # 普通多元正态分布
            dist = MultivariateNormal(means, torch.diag_embed(stds))
        
        # 重参数化采样
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs, means
    
    def get_action_distribution(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> TanhMultivariateNormalDiag:
        """
        获取动作分布（用于分析或特殊采样）
        
        Returns:
            动作分布对象
        """
        with torch.no_grad():
            _, _, means = self.forward(observations, observation_features)
            
            device = get_device_from_parameters(self)
            batch_size = next(iter(observations.values())).shape[0]
            
            # 重新计算分布参数（避免重复前向传播）
            obs_features = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
            transformer_input = self.obs_to_transformer_proj(obs_features)
            transformer_input = transformer_input.unsqueeze(0)
            
            pos_embed = self.encoder_pos_embed[:1].expand(-1, batch_size, -1)
            encoder_out = self.transformer_encoder(transformer_input, pos_embed=pos_embed)
            
            decoder_input = torch.zeros((1, batch_size, self.dim_model), device=device, dtype=transformer_input.dtype)
            decoder_pos_embed = self.decoder_pos_embed[:1].expand(-1, batch_size, -1)
            decoder_out = self.transformer_decoder(decoder_input, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=pos_embed)
            
            output_features = decoder_out.squeeze(0)
            means = self.mean_layer(output_features)
            log_stds = self.std_layer(output_features)
            log_stds = torch.clamp(log_stds, self.std_min, self.std_max)
            stds = torch.exp(log_stds)
            
            return TanhMultivariateNormalDiag(loc=means, scale_diag=stds)
    
    def deterministic_action(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """
        获取确定性动作（用于评估）
        
        Returns:
            确定性动作 (batch_size, action_dim)
        """
        with torch.no_grad():
            _, _, means = self.forward(observations, observation_features)
            
            if self.use_tanh_squash:
                # 如果使用tanh压缩，需要应用tanh变换
                return torch.tanh(means)
            else:
                return means


# class SequenceACTSACActor(ACTSACActor):
#     """
#     支持观测序列的ACT-SAC Actor
    
#     这个版本可以处理观测历史序列，更好地利用ACT的序列建模能力
#     """
    
#     def __init__(self, *args, obs_history_length: int = 5, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.obs_history_length = obs_history_length
        
#         # 更新位置编码以支持更长的序列
#         self.register_buffer(
#             "encoder_pos_embed",
#             create_sinusoidal_pos_embedding(obs_history_length, self.dim_model).unsqueeze(1)
#         )
    
#     def forward(
#         self,
#         observations: dict[str, Tensor] | list[dict[str, Tensor]],
#         observation_features: Tensor | None = None,
#     ) -> tuple[Tensor, Tensor, Tensor]:
#         """
#         支持观测序列的前向传播
        
#         Args:
#             observations: 观测字典或观测序列 [obs_t-n, ..., obs_t-1, obs_t]
#             observation_features: 预计算的观测特征
#         """
#         device = get_device_from_parameters(self)
        
#         # 处理单个观测或观测序列
#         if isinstance(observations, dict):
#             # 单个观测，复制为序列
#             obs_list = [observations] * self.obs_history_length
#         else:
#             # 观测序列
#             obs_list = observations[-self.obs_history_length:]  # 取最后N个观测
            
#             # 如果序列不够长，用最早的观测填充
#             while len(obs_list) < self.obs_history_length:
#                 obs_list.insert(0, obs_list[0])
        
#         batch_size = next(iter(obs_list[0].values())).shape[0]
        
#         # 编码每个观测
#         obs_features_list = []
#         for obs in obs_list:
#             if self.encoder_is_shared:
#                 obs_feat = self.encoder(obs, cache=None, detach=True)
#             else:
#                 obs_feat = self.encoder(obs, cache=None, detach=False)
#             obs_features_list.append(obs_feat)
        
#         # 堆叠为序列
#         obs_features_seq = torch.stack(obs_features_list, dim=0)  # (seq_len, batch, feature_dim)
        
#         # 投影到Transformer空间
#         transformer_input = self.obs_to_transformer_proj(obs_features_seq)  # (seq_len, batch, dim_model)
        
#         # 添加位置编码
#         pos_embed = self.encoder_pos_embed.expand(-1, batch_size, -1)  # (seq_len, batch, dim_model)
        
#         # Transformer Encoder
#         encoder_out = self.transformer_encoder(transformer_input, pos_embed=pos_embed)
        
#         # Transformer Decoder（只解码一个动作）
#         decoder_input = torch.zeros((1, batch_size, self.dim_model), device=device, dtype=transformer_input.dtype)
#         decoder_pos_embed = self.decoder_pos_embed[:1].expand(-1, batch_size, -1)
        
#         decoder_out = self.transformer_decoder(
#             decoder_input,
#             encoder_out,
#             decoder_pos_embed=decoder_pos_embed,
#             encoder_pos_embed=pos_embed
#         )
        
#         # 计算动作分布和采样
#         output_features = decoder_out.squeeze(0)
#         means = self.mean_layer(output_features)
#         log_stds = self.std_layer(output_features)
#         log_stds = torch.clamp(log_stds, self.std_min, self.std_max)
#         stds = torch.exp(log_stds)
        
#         if self.use_tanh_squash:
#             dist = TanhMultivariateNormalDiag(loc=means, scale_diag=stds)
#         else:
#             dist = MultivariateNormal(means, torch.diag_embed(stds))
        
#         actions = dist.rsample()
#         log_probs = dist.log_prob(actions)
        
#         return actions, log_probs, means

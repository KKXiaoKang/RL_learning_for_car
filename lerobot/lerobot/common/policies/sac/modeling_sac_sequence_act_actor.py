#!/usr/bin/env python3
"""
真正的序列ACT-SAC Actor实现

这个实现充分利用ACT的核心优势：
1. 预测动作序列（chunk）而不是单个动作
2. 计算整个动作序列的联合概率损失
3. 支持自回归动作生成
4. 与SAC框架的序列损失集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal
import logging
from typing import Optional, Tuple, Dict, List

from lerobot.common.policies.sac.modeling_sac import TanhMultivariateNormalDiag
from lerobot.common.policies.utils import get_device_from_parameters


def create_sinusoidal_pos_embedding(seq_len: int, dim: int) -> Tensor:
    """创建正弦位置编码"""
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
    
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


class SequenceACTSACActorV2(nn.Module):
    """
    真正的序列ACT-SAC Actor
    
    核心特性：
    1. 预测动作序列（action chunks）
    2. 支持自回归生成
    3. 计算序列联合概率损失
    4. 与SAC损失函数集成
    
    架构：
    观测序列 → Encoder → ACT Transformer → 动作序列预测
    """
    
    def __init__(
        self,
        encoder,  # SACObservationEncoder
        action_dim: int,
        chunk_size: int = 8,  # 预测的动作序列长度
        obs_history_length: int = 5,  # 观测历史长度
        
        # Transformer 参数
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,  # 增加decoder层数以处理序列
        dropout: float = 0.1,
        feedforward_activation: str = "relu",
        pre_norm: bool = False,
        
        # SAC 参数
        std_min: float = -5,
        std_max: float = 2,
        use_tanh_squash: bool = True,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.obs_history_length = obs_history_length
        self.dim_model = dim_model
        self.std_min = std_min
        self.std_max = std_max
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared
        
        # 特征投影层
        self.obs_to_transformer_proj = nn.Linear(encoder.output_dim, dim_model)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=feedforward_activation,
            norm_first=pre_norm,
            batch_first=False  # (seq, batch, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=feedforward_activation,
            norm_first=pre_norm,
            batch_first=False  # (seq, batch, dim)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # 动作预测头
        self.action_mean_head = nn.Linear(dim_model, action_dim)
        self.action_std_head = nn.Linear(dim_model, action_dim)
        
        # 位置编码
        self.register_buffer(
            "obs_pos_embed",
            create_sinusoidal_pos_embedding(obs_history_length, dim_model).unsqueeze(1)
        )
        self.register_buffer(
            "action_pos_embed", 
            create_sinusoidal_pos_embedding(chunk_size, dim_model).unsqueeze(1)
        )
        
        # 可学习的序列开始token
        self.action_start_token = nn.Parameter(torch.randn(1, 1, dim_model))
        
        logging.info(f"✅ Initialized SequenceACTSACActorV2 with chunk_size={chunk_size}, obs_history_length={obs_history_length}")
    
    def encode_observations(self, observations: List[Dict[str, Tensor]]) -> Tensor:
        """
        编码观测序列
        
        Args:
            observations: 观测序列 [obs_t-n, ..., obs_t-1, obs_t]
            
        Returns:
            编码后的观测序列 (obs_history_length, batch_size, dim_model)
        """
        device = get_device_from_parameters(self)
        
        # 确保有足够的观测历史
        if len(observations) < self.obs_history_length:
            # 用最早的观测填充
            while len(observations) < self.obs_history_length:
                observations.insert(0, observations[0])
        else:
            # 取最后N个观测
            observations = observations[-self.obs_history_length:]
        
        # 编码每个观测
        obs_features_list = []
        for obs in observations:
            # 移动到正确的设备
            obs = {k: v.to(device) for k, v in obs.items()}
            
            if self.encoder_is_shared:
                obs_feat = self.encoder(obs, cache=None, detach=True)
            else:
                obs_feat = self.encoder(obs, cache=None, detach=False)
            obs_features_list.append(obs_feat)
        
        # 堆叠为序列
        obs_features_seq = torch.stack(obs_features_list, dim=0)  # (seq_len, batch, feature_dim)
        
        # 投影到Transformer空间
        transformer_input = self.obs_to_transformer_proj(obs_features_seq)
        
        return transformer_input
    
    def forward(
        self,
        observations: List[Dict[str, Tensor]] | Dict[str, Tensor],
        observation_features: Optional[Tensor] = None,
        action_history: Optional[Tensor] = None,  # 动作历史用于自回归
        return_sequence: bool = True,  # 是否返回完整序列
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        前向传播：预测动作序列
        
        Args:
            observations: 观测序列或单个观测
            observation_features: 预计算的观测特征（暂不使用）
            action_history: 历史动作序列（用于自回归生成）
            return_sequence: 是否返回完整动作序列
            
        Returns:
            如果return_sequence=True:
                actions: 动作序列 (batch_size, chunk_size, action_dim)
                log_probs: 序列对数概率 (batch_size,) - 联合概率
                means: 动作均值序列 (batch_size, chunk_size, action_dim)
            如果return_sequence=False:
                actions: 单个动作 (batch_size, action_dim) - 序列的第一个动作
                log_probs: 单个动作的对数概率 (batch_size,)
                means: 单个动作的均值 (batch_size, action_dim)
        """
        device = get_device_from_parameters(self)
        
        # 处理单个观测的情况
        if isinstance(observations, dict):
            observations = [observations]
        
        batch_size = next(iter(observations[0].values())).shape[0]
        
        # 1. 编码观测序列
        obs_encoded = self.encode_observations(observations)  # (obs_seq_len, batch, dim_model)
        
        # 2. 添加观测位置编码
        obs_pos_embed = self.obs_pos_embed.expand(-1, batch_size, -1)
        obs_encoded = obs_encoded + obs_pos_embed
        
        # 3. Transformer Encoder
        obs_memory = self.transformer_encoder(obs_encoded)  # (obs_seq_len, batch, dim_model)
        
        # 4. 构建解码器输入序列
        if action_history is not None:
            # 使用历史动作作为输入
            decoder_input = self._prepare_decoder_input_with_history(action_history, batch_size, device)
        else:
            # 使用可学习的start token
            decoder_input = self.action_start_token.expand(self.chunk_size, batch_size, -1)
        
        # 5. 添加动作位置编码
        action_pos_embed = self.action_pos_embed.expand(-1, batch_size, -1)
        decoder_input = decoder_input + action_pos_embed
        
        # 6. Transformer Decoder（自回归）
        if self.training:
            # 训练时使用teacher forcing
            decoder_output = self._teacher_forcing_decode(decoder_input, obs_memory)
        else:
            # 推理时使用自回归生成
            decoder_output = self._autoregressive_decode(obs_memory, batch_size, device)
        
        # 7. 预测动作分布参数
        action_means = self.action_mean_head(decoder_output)  # (chunk_size, batch, action_dim)
        action_log_stds = self.action_std_head(decoder_output)  # (chunk_size, batch, action_dim)
        
        # 转换维度为 (batch, chunk_size, action_dim)
        action_means = action_means.transpose(0, 1)
        action_log_stds = action_log_stds.transpose(0, 1)
        
        # 限制标准差范围
        action_log_stds = torch.clamp(action_log_stds, self.std_min, self.std_max)
        action_stds = torch.exp(action_log_stds)
        
        # 8. 采样动作序列并计算联合概率
        actions_sequence, log_probs_joint = self._sample_action_sequence(action_means, action_stds)
        
        # 9. 根据需要返回序列或单个动作
        if return_sequence:
            return actions_sequence, log_probs_joint, action_means
        else:
            # 返回序列的第一个动作（用于SAC的即时执行）
            first_action = actions_sequence[:, 0, :]  # (batch, action_dim)
            first_mean = action_means[:, 0, :]  # (batch, action_dim)
            
            # 计算第一个动作的对数概率
            if self.use_tanh_squash:
                first_dist = TanhMultivariateNormalDiag(
                    loc=first_mean, 
                    scale_diag=action_stds[:, 0, :]
                )
            else:
                first_dist = MultivariateNormal(
                    first_mean, 
                    torch.diag_embed(action_stds[:, 0, :])
                )
            
            first_log_prob = first_dist.log_prob(first_action)
            
            return first_action, first_log_prob, first_mean
    
    def _teacher_forcing_decode(self, decoder_input: Tensor, obs_memory: Tensor) -> Tensor:
        """训练时的teacher forcing解码"""
        # 创建因果掩码以防止看到未来的动作
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.chunk_size).to(decoder_input.device)
        
        decoder_output = self.transformer_decoder(
            decoder_input,
            obs_memory,
            tgt_mask=tgt_mask
        )
        
        return decoder_output
    
    def _autoregressive_decode(self, obs_memory: Tensor, batch_size: int, device: torch.device) -> Tensor:
        """推理时的自回归解码"""
        outputs = []
        current_input = self.action_start_token.expand(1, batch_size, -1)
        
        for i in range(self.chunk_size):
            # 添加位置编码
            pos_embed = self.action_pos_embed[i:i+1].expand(-1, batch_size, -1)
            current_input_with_pos = current_input + pos_embed
            
            # 解码当前时间步
            output = self.transformer_decoder(
                current_input_with_pos,
                obs_memory
            )
            
            outputs.append(output)
            
            # 准备下一个时间步的输入
            if i < self.chunk_size - 1:
                current_input = output
        
        return torch.cat(outputs, dim=0)  # (chunk_size, batch, dim_model)
    
    def _sample_action_sequence(self, means: Tensor, stds: Tensor) -> Tuple[Tensor, Tensor]:
        """
        采样动作序列并计算联合对数概率
        
        这个方法实现了Q-chunking的核心：将动作序列作为一个整体进行采样和概率计算
        
        Args:
            means: 动作均值序列 (batch, chunk_size, action_dim)
            stds: 动作标准差序列 (batch, chunk_size, action_dim)
            
        Returns:
            actions: 采样的动作序列 (batch, chunk_size, action_dim)
            log_probs: 联合对数概率 (batch,) - Q-chunking的关键输出
        """
        batch_size = means.shape[0]
        actions_list = []
        log_probs_list = []
        
        for t in range(self.chunk_size):
            # 为每个时间步创建分布
            if self.use_tanh_squash:
                dist = TanhMultivariateNormalDiag(
                    loc=means[:, t, :], 
                    scale_diag=stds[:, t, :]
                )
            else:
                dist = MultivariateNormal(
                    means[:, t, :], 
                    torch.diag_embed(stds[:, t, :])
                )
            
            # 采样动作（使用重参数化技巧确保梯度可传播）
            action = dist.rsample()
            log_prob = dist.log_prob(action)
            
            actions_list.append(action)
            log_probs_list.append(log_prob)
        
        # 组合结果
        actions_sequence = torch.stack(actions_list, dim=1)  # (batch, chunk_size, action_dim)
        log_probs_individual = torch.stack(log_probs_list, dim=1)  # (batch, chunk_size)
        
        # 🔥 Q-chunking核心：计算动作序列的联合对数概率
        # 这假设了动作在给定观测序列下是条件独立的，但仍然捕获了序列的时间一致性
        log_probs_joint = log_probs_individual.sum(dim=1)  # (batch,)
        
        return actions_sequence, log_probs_joint
    
    def _prepare_decoder_input_with_history(
        self, 
        action_history: Tensor, 
        batch_size: int, 
        device: torch.device
    ) -> Tensor:
        """准备带有历史动作的解码器输入"""
        # action_history: (batch, history_len, action_dim)
        history_len = action_history.shape[1]
        
        # 如果历史长度不够，用start token填充
        if history_len < self.chunk_size:
            padding_len = self.chunk_size - history_len
            start_tokens = self.action_start_token.expand(padding_len, batch_size, -1)
            
            # 将历史动作投影到transformer空间
            history_proj = self.action_mean_head.weight.T @ action_history.transpose(1, 2)  # 简单的反向投影
            history_proj = history_proj.transpose(1, 2).transpose(0, 1)
            
            decoder_input = torch.cat([start_tokens, history_proj], dim=0)
        else:
            # 截取最后chunk_size个动作
            recent_history = action_history[:, -self.chunk_size:, :]
            history_proj = self.action_mean_head.weight.T @ recent_history.transpose(1, 2)
            decoder_input = history_proj.transpose(1, 2).transpose(0, 1)
        
        return decoder_input
    
    def get_action_sequence_distribution(
        self,
        observations: List[Dict[str, Tensor]] | Dict[str, Tensor],
        observation_features: Optional[Tensor] = None,
    ) -> List[TanhMultivariateNormalDiag]:
        """
        获取完整动作序列的分布
        
        Returns:
            动作分布列表，每个时间步一个分布
        """
        with torch.no_grad():
            _, _, means_sequence = self.forward(
                observations, 
                observation_features, 
                return_sequence=True
            )
            
            # 重新计算标准差（避免重复前向传播的简化版本）
            batch_size = means_sequence.shape[0]
            device = means_sequence.device
            
            # 简化：使用固定标准差
            fixed_std = torch.ones_like(means_sequence) * 0.1
            
            distributions = []
            for t in range(self.chunk_size):
                if self.use_tanh_squash:
                    dist = TanhMultivariateNormalDiag(
                        loc=means_sequence[:, t, :],
                        scale_diag=fixed_std[:, t, :]
                    )
                else:
                    dist = MultivariateNormal(
                        means_sequence[:, t, :],
                        torch.diag_embed(fixed_std[:, t, :])
                    )
                distributions.append(dist)
            
            return distributions
    
    def compute_n_step_returns(
        self,
        rewards: Tensor,
        next_observations: List[Dict[str, Tensor]],
        done: Tensor,
        gamma: float = 0.99,
        observation_features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算n-step returns以支持Q-chunking的n-step TD learning
        
        这是Q-chunking论文中的关键组件：使用动作序列进行n-step backup
        
        Args:
            rewards: 单步奖励 (batch_size,)
            next_observations: 下一个观测序列
            done: 终止标志 (batch_size,)
            gamma: 折扣因子
            observation_features: 预计算的观测特征
            
        Returns:
            n_step_returns: n步回报 (batch_size,)
        """
        with torch.no_grad():
            # 如果chunk_size=1，退化为标准1-step return
            if self.chunk_size == 1:
                return rewards
            
            # 对于multi-step，我们需要计算整个chunk的累计奖励
            # 这里简化实现，实际应该通过环境交互获得chunk_size步的奖励
            
            # 预测下一个状态的动作序列
            next_action_sequence, next_log_probs_joint, _ = self.forward(
                next_observations,
                observation_features,
                return_sequence=True
            )
            
            # 简化的n-step return计算
            # 实际实现中，这应该通过真实的环境rollout来计算
            n_step_return = rewards  # 起始值为即时奖励
            
            # 对于演示，我们使用几何级数近似n-step return
            discount_factor = gamma ** self.chunk_size
            n_step_return = rewards * (1 - discount_factor) / (1 - gamma) if gamma != 1 else rewards * self.chunk_size
            
            return n_step_return
    
    def get_chunked_action_for_execution(
        self,
        observations: List[Dict[str, Tensor]] | Dict[str, Tensor],
        observation_features: Optional[Tensor] = None,
        action_index: int = 0,
    ) -> Tensor:
        """
        获取动作序列中的特定动作用于执行
        
        这支持Q-chunking的执行策略：一次预测多步，但逐步执行
        
        Args:
            observations: 观测序列
            observation_features: 预计算的观测特征
            action_index: 要执行的动作在序列中的索引
            
        Returns:
            单个动作用于执行
        """
        with torch.no_grad():
            action_sequence, _, _ = self.forward(
                observations,
                observation_features,
                return_sequence=True
            )
            
            # 返回指定索引的动作
            if action_index >= self.chunk_size:
                logging.warning(f"Action index {action_index} exceeds chunk size {self.chunk_size}, using last action")
                action_index = self.chunk_size - 1
            
            return action_sequence[:, action_index, :]  # (batch_size, action_dim)

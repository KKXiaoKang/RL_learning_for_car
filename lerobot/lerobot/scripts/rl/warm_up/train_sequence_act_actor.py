#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from lerobot.common.policies.sac.modeling_sac_sequence_act_actor import SequenceACTSACActorV2
from lerobot.common.policies.sac.modeling_sac import SACObservationEncoder
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.normalize import NormalizeBuffer
from lerobot.common.policies.sac.modeling_sac import _convert_normalization_params_to_tensor
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config_from_json(config_path: str) -> TrainPipelineConfig:
    """Load training configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create a TrainPipelineConfig-like object with nested structure
    class ConfigObj:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, ConfigObj(v))
                else:
                    setattr(self, k, v)
    
    return ConfigObj(config_dict)

class SequenceDataset(torch.utils.data.Dataset):
    """
    支持时序观测序列的数据集类
    
    这个类确保每个样本都包含真正的时序观测序列，而不是重复的观测
    """
    
    def __init__(self, base_dataset: LeRobotDataset, obs_history_length: int = 5, chunk_size: int = 8):
        self.base_dataset = base_dataset
        self.obs_history_length = obs_history_length
        self.chunk_size = chunk_size
        
        # 计算有效的序列索引
        self.valid_indices = self._compute_valid_indices()
        logging.info(f"Created SequenceDataset with {len(self.valid_indices)} valid sequence samples")
    
    def _compute_valid_indices(self):
        """计算可以形成完整序列的有效索引"""
        valid_indices = []
        
        # 检查数据集的实际结构
        if len(self.base_dataset) == 0:
            logging.warning("Base dataset is empty!")
            return valid_indices
        
        # 检查第一个样本的结构
        first_sample = self.base_dataset[0]
        logging.info(f"First sample keys: {list(first_sample.keys())}")
        
        # 尝试不同的episode标识方法
        episode_data = {}
        
        # 方法1: 检查是否有episode_index字段
        if 'episode_index' in first_sample:
            logging.info("Using episode_index field for episode grouping")
            for idx in range(len(self.base_dataset)):
                sample = self.base_dataset[idx]
                episode_id = sample['episode_index']
                
                if episode_id not in episode_data:
                    episode_data[episode_id] = []
                episode_data[episode_id].append(idx)
        
        # 方法2: 如果没有episode_index，尝试使用episode字段
        elif 'episode' in first_sample:
            logging.info("Using episode field for episode grouping")
            for idx in range(len(self.base_dataset)):
                sample = self.base_dataset[idx]
                episode_id = sample['episode']
                
                if episode_id not in episode_data:
                    episode_data[episode_id] = []
                episode_data[episode_id].append(idx)
        
        # 方法3: 如果都没有，假设所有数据属于同一个episode
        else:
            logging.warning("No episode field found, treating all data as single episode")
            episode_data[0] = list(range(len(self.base_dataset)))
        
        logging.info(f"Found {len(episode_data)} episodes")
        for episode_id, indices in episode_data.items():
            logging.info(f"Episode {episode_id}: {len(indices)} samples")
        
        # 为每个episode生成有效的序列索引
        short_episodes_count = 0
        for episode_id, indices in episode_data.items():
            # 确保有足够的样本形成序列
            if len(indices) >= self.obs_history_length:
                # 为每个可能的起始位置创建序列
                for start_idx in range(len(indices) - self.obs_history_length + 1):
                    sequence_indices = indices[start_idx:start_idx + self.obs_history_length]
                    valid_indices.append(sequence_indices)
            else:
                # 如果episode太短，仍然创建序列但重复最后一个样本
                short_episodes_count += 1
                if len(indices) > 0:
                    # 重复最后一个样本来达到所需长度
                    sequence_indices = indices[:]
                    while len(sequence_indices) < self.obs_history_length:
                        sequence_indices.append(indices[-1])  # 重复最后一个样本
                    valid_indices.append(sequence_indices)
        
        # 只在有短episode时输出一次警告
        if short_episodes_count > 0:
            logging.warning(f"Found {short_episodes_count} episodes with less than {self.obs_history_length} samples. Creating sequences by repeating samples.")
        
        logging.info(f"Generated {len(valid_indices)} valid sequence indices")
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """获取一个时序序列样本"""
        sequence_indices = self.valid_indices[idx]
        
        # 收集序列中的所有观测和动作
        observations_sequence = []
        actions_sequence = []
        
        for seq_idx in sequence_indices:
            sample = self.base_dataset[seq_idx]
            
            # 提取观测
            obs_dict = {}
            for key, value in sample.items():
                if key.startswith('observation.'):
                    obs_dict[key] = value
            
            # 提取动作
            action = sample['action']
            
            observations_sequence.append(obs_dict)
            actions_sequence.append(action)
        
        # 返回当前时刻的观测序列和目标动作
        # 目标动作是序列中最后一个动作（当前时刻的动作）
        target_action = actions_sequence[-1]
        
        return {
            'observations_sequence': observations_sequence,
            'target_action': target_action,
            'sequence_length': len(observations_sequence)
        }


def create_sequence_behavior_cloning_dataset(
    dataset: LeRobotDataset, 
    batch_size: int = 32, 
    num_workers: int = 4, 
    split_ratio: float = 0.9,
    obs_history_length: int = 5,
    chunk_size: int = 8
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for sequence behavior cloning training"""
    
    def collate_fn(batch):
        """Custom collate function to handle sequence data"""
        # 收集所有序列数据
        all_obs_sequences = []
        all_target_actions = []
        
        for item in batch:
            all_obs_sequences.append(item['observations_sequence'])
            all_target_actions.append(item['target_action'])
        
        # 堆叠目标动作
        stacked_actions = torch.stack(all_target_actions, dim=0)
        
        # 处理观测序列 - 保持序列结构
        # 每个样本的观测序列: List[Dict[str, Tensor]]
        # 我们需要将其转换为: Dict[str, Tensor] where Tensor.shape = (batch_size, seq_len, ...)
        
        # 获取第一个样本的观测键
        first_obs_keys = all_obs_sequences[0][0].keys()
        
        stacked_observations = {}
        for key in first_obs_keys:
            # 收集所有样本中该键的序列
            key_sequences = []
            for obs_sequence in all_obs_sequences:
                # obs_sequence是一个观测序列: List[Dict]
                # 我们需要提取该键的所有时间步
                key_timesteps = [obs[key] for obs in obs_sequence]
                key_sequences.append(torch.stack(key_timesteps, dim=0))  # (seq_len, ...)
            
            # 堆叠所有样本: (batch_size, seq_len, ...)
            stacked_observations[key] = torch.stack(key_sequences, dim=0)
        
        return {
            'observations_sequence': stacked_observations,
            'target_actions': stacked_actions
        }
    
    # 创建序列数据集
    sequence_dataset = SequenceDataset(dataset, obs_history_length, chunk_size)
    
    # 检查是否成功创建了序列数据
    if len(sequence_dataset) == 0:
        logging.error("Failed to create sequence dataset! Falling back to simple dataset.")
        # 回退到简单的数据集
        return create_simple_behavior_cloning_dataset(
            dataset, batch_size, num_workers, split_ratio, obs_history_length, chunk_size
        )
    
    # Split dataset into train and validation
    total_size = len(sequence_dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        sequence_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducible splits
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_simple_behavior_cloning_dataset(
    dataset: LeRobotDataset, 
    batch_size: int = 32, 
    num_workers: int = 4, 
    split_ratio: float = 0.9,
    obs_history_length: int = 5,
    chunk_size: int = 8
) -> tuple[DataLoader, DataLoader]:
    """Create simple behavior cloning dataset as fallback"""
    logging.warning("Using simple behavior cloning dataset (no temporal sequences)")
    
    def simple_collate_fn(batch):
        """Simple collate function without temporal sequences"""
        observations = []
        actions = []
        
        for item in batch:
            # Collect observations
            obs_dict = {}
            for key, value in item.items():
                if key.startswith('observation.'):
                    obs_dict[key] = value
            
            # Collect action
            action = item['action']
            
            observations.append(obs_dict)
            actions.append(action)
        
        # Stack observations and actions
        stacked_observations = {}
        for key in observations[0].keys():
            stacked_observations[key] = torch.stack([obs[key] for obs in observations], dim=0)
        
        stacked_actions = torch.stack(actions, dim=0)
        
        return {
            'observations_sequence': stacked_observations,
            'target_actions': stacked_actions
        }
    
    # Split dataset into train and validation
    total_size = len(dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducible splits
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=simple_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=simple_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_observation_encoder(config) -> SACObservationEncoder:
    """Create SACObservationEncoder with vision support"""
    from lerobot.common.policies.sac.configuration_sac import SACConfig
    from lerobot.common.policies.normalize import NormalizeBuffer
    
    # Create input features dict
    input_features = {}
    for key in config.policy.input_features.__dict__:
        feature_config = getattr(config.policy.input_features, key)
        if hasattr(feature_config, '__dict__'):  # ConfigObj
            feature_type = FeatureType(feature_config.type)
            shape = tuple(feature_config.shape)
            input_features[key] = PolicyFeature(type=feature_type, shape=shape)
        else:
            input_features[key] = feature_config
    
    # Create a minimal SACConfig for the encoder with proper defaults
    sac_config = SACConfig(
        # Set input features
        input_features=input_features,
        
        # Set vision encoder parameters
        vision_encoder_name=getattr(config.policy, 'vision_encoder_name', None),
        freeze_vision_encoder=getattr(config.policy, 'freeze_vision_encoder', False),
        enable_feature_visualization=getattr(config.policy, 'enable_feature_visualization', False),
        image_encoder_hidden_dim=getattr(config.policy, 'image_encoder_hidden_dim', 0),
        shared_encoder=getattr(config.policy, 'shared_encoder', False),
        latent_dim=getattr(config.policy, 'latent_dim', 64),
        state_encoder_hidden_dim=getattr(config.policy, 'state_encoder_hidden_dim', 256),
        
        # Set ACT and Q-chunking parameters
        use_act_actor=getattr(config.policy, 'use_act_actor', True),
        use_sequence_act_actor=getattr(config.policy, 'use_sequence_act_actor', True),
        enable_q_chunking=getattr(config.policy, 'enable_q_chunking', True),
        q_chunking_horizon=getattr(config.policy, 'q_chunking_horizon', 8),
    )
    
    # Set normalization mapping
    if hasattr(config.policy, 'normalization_mapping'):
        norm_mapping = config.policy.normalization_mapping
        if isinstance(norm_mapping, dict):
            converted = {}
            for key, value in norm_mapping.items():
                if isinstance(value, str):
                    converted[key] = NormalizationMode(value)
                else:
                    converted[key] = value
            sac_config.normalization_mapping = converted
    
    # Set dataset stats
    if hasattr(config.policy, 'dataset_stats') and config.policy.dataset_stats:
        sac_config.dataset_stats = config.policy.dataset_stats
    
    # Create input normalizer
    norm_map = {}
    if hasattr(config.policy, 'normalization_mapping'):
        norm_mapping = config.policy.normalization_mapping
        if isinstance(norm_mapping, dict):
            for key, value in norm_mapping.items():
                if isinstance(value, str):
                    norm_map[key] = NormalizationMode(value)
                else:
                    norm_map[key] = value
    
    input_normalizer = NormalizeBuffer(
        features=input_features,
        norm_map=norm_map,
        stats=getattr(config.policy, 'dataset_stats', None)
    )
    
    return SACObservationEncoder(sac_config, input_normalizer)

def compute_sequence_bc_loss(
    actor: SequenceACTSACActorV2,
    observations_sequence: Dict[str, torch.Tensor],
    target_actions: torch.Tensor,
    chunk_size: int,
    obs_history_length: int
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute behavior cloning loss for sequence ACT actor with proper temporal observations
    
    Args:
        actor: SequenceACTSACActorV2 model
        observations_sequence: Batch of observation sequences (batch_size, seq_len, ...) or (batch_size, ...)
        target_actions: Target actions (batch_size, action_dim)
        chunk_size: Action sequence length
        obs_history_length: Observation history length
        
    Returns:
        loss: Total BC loss
        metrics: Dictionary of metrics
    """
    batch_size = target_actions.shape[0]
    device = target_actions.device
    
    # Check if we have temporal sequences or simple observations
    first_key = next(iter(observations_sequence.keys()))
    first_tensor = observations_sequence[first_key]
    
    # Debug: Log tensor shapes (only once per epoch)
    if not hasattr(compute_sequence_bc_loss, '_logged_shapes'):
        logging.info(f"First tensor shape: {first_tensor.shape}")
        logging.info(f"Expected obs_history_length: {obs_history_length}")
        compute_sequence_bc_loss._logged_shapes = True
    
    # Determine if we have temporal dimension
    # Check if the tensor has a temporal dimension by looking at the second dimension
    has_temporal_dim = len(first_tensor.shape) > 2  # (batch_size, seq_len, ...) vs (batch_size, ...)
    
    # Additional check: if we have temporal dimension, verify the sequence length matches
    if has_temporal_dim and first_tensor.shape[1] != obs_history_length:
        logging.warning(f"Temporal dimension mismatch: expected {obs_history_length}, got {first_tensor.shape[1]}. Falling back to simple mode.")
        has_temporal_dim = False
    
    if has_temporal_dim:
        # Convert observations_sequence to the format expected by the actor
        # observations_sequence: Dict[str, Tensor] where Tensor.shape = (batch_size, seq_len, ...)
        # We need to convert to List[Dict[str, Tensor]] where each Dict contains (batch_size, ...)
        
        obs_sequence_list = []
        for t in range(obs_history_length):
            obs_t = {}
            for key, tensor in observations_sequence.items():
                # tensor.shape = (batch_size, seq_len, ...)
                # We want obs_t[key].shape = (batch_size, ...)
                obs_t[key] = tensor[:, t, ...]
            obs_sequence_list.append(obs_t)
    else:
        # Simple case: no temporal dimension, repeat the same observation
        # observations_sequence: Dict[str, Tensor] where Tensor.shape = (batch_size, ...)
        obs_sequence_list = []
        for t in range(obs_history_length):
            obs_t = {}
            for key, tensor in observations_sequence.items():
                # tensor.shape = (batch_size, ...)
                # We want obs_t[key].shape = (batch_size, ...) - same observation repeated
                obs_t[key] = tensor
            obs_sequence_list.append(obs_t)
    
    # Forward pass through the actor
    try:
        # Get action sequence prediction
        predicted_actions, log_probs, action_means = actor.forward(
            obs_sequence_list,
            return_sequence=True
        )
        
        # For BC loss, we compare the first action in the sequence with the target
        predicted_first_action = predicted_actions[:, 0, :]  # (batch_size, action_dim)
        
        # Compute MSE loss between predicted and target actions
        mse_loss = F.mse_loss(predicted_first_action, target_actions)
        
        # Optional: Add regularization loss for the full sequence
        # This encourages the model to predict reasonable action sequences
        sequence_consistency_loss = 0.0
        if chunk_size > 1:
            # Compute consistency loss between consecutive actions in the sequence
            for t in range(chunk_size - 1):
                action_diff = predicted_actions[:, t+1, :] - predicted_actions[:, t, :]
                sequence_consistency_loss += F.mse_loss(action_diff, torch.zeros_like(action_diff))
            sequence_consistency_loss /= (chunk_size - 1)
        
        # Total loss - reduce sequence consistency weight for stability
        total_loss = mse_loss + 0.01 * sequence_consistency_loss
        
        # Compute metrics
        with torch.no_grad():
            mse_metric = F.mse_loss(predicted_first_action, target_actions).item()
            mae_metric = F.l1_loss(predicted_first_action, target_actions).item()
            
            # Compute action magnitude metrics
            pred_magnitude = torch.norm(predicted_first_action, dim=1).mean().item()
            target_magnitude = torch.norm(target_actions, dim=1).mean().item()
            
            # Compute sequence smoothness metric
            sequence_smoothness = 0.0
            if chunk_size > 1:
                action_diffs = []
                for t in range(chunk_size - 1):
                    diff = torch.norm(predicted_actions[:, t+1, :] - predicted_actions[:, t, :], dim=1)
                    action_diffs.append(diff)
                sequence_smoothness = torch.stack(action_diffs).mean().item()
        
        metrics = {
            'bc/mse': mse_metric,
            'bc/mae': mae_metric,
            'bc/pred_magnitude': pred_magnitude,
            'bc/target_magnitude': target_magnitude,
            'bc/sequence_consistency': sequence_consistency_loss.item() if chunk_size > 1 else 0.0,
            'bc/sequence_smoothness': sequence_smoothness
        }
        
        return total_loss, metrics
        
    except Exception as e:
        logging.error(f"Error in forward pass: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a dummy loss to prevent training from crashing
        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        dummy_metrics = {
            'bc/mse': 0.0,
            'bc/mae': 0.0,
            'bc/pred_magnitude': 0.0,
            'bc/target_magnitude': 0.0,
            'bc/sequence_consistency': 0.0,
            'bc/sequence_smoothness': 0.0
        }
        return dummy_loss, dummy_metrics

def train_sequence_act_actor(config_path: str):
    """Main training function for Sequence ACT Actor warmup"""
    logger = setup_logging()
    logger.info("Starting Sequence ACT Actor warmup training")
    
    # Load configuration
    cfg = load_config_from_json(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Setup device
    device = torch.device(cfg.policy.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if enabled
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.job_name,
            dir=str(output_dir),
            config=cfg.__dict__ if hasattr(cfg, '__dict__') else {}
        )
        logger.info(f"Initialized wandb logging in directory: {output_dir}")
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load dataset
    logger.info(f"Loading dataset: {cfg.dataset.repo_id}")
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        episodes=None,  # Load all episodes
        download_videos=True
    )
    
    # Create data loaders
    train_loader, val_loader = create_sequence_behavior_cloning_dataset(
        dataset, 
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        obs_history_length=getattr(cfg.policy, 'obs_history_length', 5),
        chunk_size=getattr(cfg.policy, 'act_chunk_size', 8)
    )
    logger.info(f"Created dataloaders - Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
    
    # Create observation encoder
    logger.info("Creating SACObservationEncoder with vision support")
    encoder = create_observation_encoder(cfg)
    encoder = encoder.to(device)
    
    # Get action dimension from output features
    action_feature = getattr(cfg.policy.output_features, 'action')
    action_dim = action_feature.shape[0]
    
    # Initialize Sequence ACT Actor
    logger.info("Initializing SequenceACTSACActorV2")
    actor = SequenceACTSACActorV2(
        encoder=encoder,
        action_dim=action_dim,
        chunk_size=getattr(cfg.policy, 'act_chunk_size', 8),
        obs_history_length=getattr(cfg.policy, 'obs_history_length', 5),
        dim_model=getattr(cfg.policy, 'act_dim_model', 512),
        n_heads=getattr(cfg.policy, 'act_n_heads', 8),
        dim_feedforward=getattr(cfg.policy, 'act_dim_feedforward', 3200),
        n_encoder_layers=getattr(cfg.policy, 'act_n_encoder_layers', 4),
        n_decoder_layers=getattr(cfg.policy, 'act_n_decoder_layers', 4),
        dropout=getattr(cfg.policy, 'act_dropout', 0.1),
        feedforward_activation=getattr(cfg.policy, 'act_feedforward_activation', 'relu'),
        pre_norm=getattr(cfg.policy, 'act_pre_norm', False),
        std_min=getattr(cfg.policy, 'std_min', -5.0),
        std_max=getattr(cfg.policy, 'std_max', 2.0),
        use_tanh_squash=getattr(cfg.policy, 'use_tanh_squash', True),
        encoder_is_shared=getattr(cfg.policy, 'encoder_is_shared', False)
    ).to(device)
    
    actor.train()
    logger.info("Initialized SequenceACTSACActorV2")
    
    # Setup optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Add learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Training loop
    logger.info("Starting training loop")
    total_steps = cfg.steps
    step = 0
    epoch = 0
    
    # Training metrics
    epoch_losses = []
    
    while step < total_steps:
        epoch += 1
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        # Reset debug logging for new epoch
        if hasattr(compute_sequence_bc_loss, '_logged_shapes'):
            delattr(compute_sequence_bc_loss, '_logged_shapes')
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Move batch to device
            observations_sequence = {k: v.to(device) for k, v in batch['observations_sequence'].items()}
            target_actions = batch['target_actions'].to(device)
            
            # Compute loss
            loss, batch_metrics = compute_sequence_bc_loss(
                actor=actor,
                observations_sequence=observations_sequence,
                target_actions=target_actions,
                chunk_size=getattr(cfg.policy, 'act_chunk_size', 8),
                obs_history_length=getattr(cfg.policy, 'obs_history_length', 5)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping - use smaller value for stability
            grad_clip_norm = getattr(cfg.policy, 'grad_clip_norm', 1.0)  # Default to 1.0 instead of 10.0
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'mse': f"{batch_metrics.get('bc/mse', 0):.6f}",
                'step': step,
                'epoch_avg_loss': f"{epoch_loss/num_batches:.6f}"
            })
            
            # Log to wandb
            if cfg.wandb.enable and step % cfg.log_freq == 0:
                log_dict = {
                    'train/bc_loss': loss.item(),
                    'train/step': step,
                    'train/epoch': epoch,
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                }
                
                # Add all batch metrics
                for key, value in batch_metrics.items():
                    log_dict[f'train/{key}'] = value
                
                wandb.log(log_dict, step=step)
            
            # Save checkpoint
            if cfg.save_checkpoint and step % cfg.save_freq == 0:
                checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
                
                # Convert config to dict to avoid pickle issues
                config_dict = {}
                def convert_config_to_dict(obj, prefix=""):
                    if hasattr(obj, '__dict__'):
                        for key, value in obj.__dict__.items():
                            new_key = f"{prefix}.{key}" if prefix else key
                            if hasattr(value, '__dict__'):
                                convert_config_to_dict(value, new_key)
                            else:
                                config_dict[new_key] = value
                    else:
                        config_dict[prefix] = obj
                
                convert_config_to_dict(cfg)
                
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'actor_state_dict': actor.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config_dict,
                    'loss': loss.item(),
                }, checkpoint_path)
                
                logger.info(f"Saved checkpoint at step {step}")
            
            # Early stopping if reached total steps
            if step >= total_steps:
                break
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.6f}")
        
        if cfg.wandb.enable:
            wandb.log({
                'train/epoch_loss': avg_epoch_loss,
                'train/epoch': epoch
            }, step=step)
        
        # Run validation at the end of each epoch
        if len(val_loader) > 0:
            actor.eval()
            val_loss = 0.0
            val_metrics = {}
            val_batches = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_observations_sequence = {k: v.to(device) for k, v in val_batch['observations_sequence'].items()}
                    val_target_actions = val_batch['target_actions'].to(device)
                    
                    val_batch_loss, val_batch_metrics = compute_sequence_bc_loss(
                        actor=actor,
                        observations_sequence=val_observations_sequence,
                        target_actions=val_target_actions,
                        chunk_size=getattr(cfg.policy, 'act_chunk_size', 8),
                        obs_history_length=getattr(cfg.policy, 'obs_history_length', 5)
                    )
                    val_loss += val_batch_loss.item()
                    val_batches += 1
                    
                    # Accumulate metrics
                    for key, value in val_batch_metrics.items():
                        if key not in val_metrics:
                            val_metrics[key] = 0.0
                        val_metrics[key] += value
            
            # Average validation metrics
            avg_val_loss = val_loss / val_batches
            for key in val_metrics:
                val_metrics[key] /= val_batches
            
            logger.info(f"Validation - Loss: {avg_val_loss:.6f}, MSE: {val_metrics.get('bc/mse', 0):.6f}")
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Log validation metrics
            if cfg.wandb.enable:
                val_log_dict = {'val/loss': avg_val_loss}
                for key, value in val_metrics.items():
                    val_log_dict[f'val/{key}'] = value
                val_log_dict['train/learning_rate'] = optimizer.param_groups[0]['lr']
                wandb.log(val_log_dict, step=step)
            
            actor.train()
        
        if step >= total_steps:
            break
    
    # Save final model
    if cfg.save_checkpoint:
        final_model_path = output_dir / "final_sequence_act_actor.pt"
        
        # Convert config to dict to avoid pickle issues
        config_dict = {}
        def convert_config_to_dict(obj, prefix=""):
            if hasattr(obj, '__dict__'):
                for key, value in obj.__dict__.items():
                    new_key = f"{prefix}.{key}" if prefix else key
                    if hasattr(value, '__dict__'):
                        convert_config_to_dict(value, new_key)
                    else:
                        config_dict[new_key] = value
            else:
                config_dict[prefix] = obj
        
        convert_config_to_dict(cfg)
        
        torch.save({
            'step': step,
            'epoch': epoch,
            'actor_state_dict': actor.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config_dict,
            'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
        }, final_model_path)
        
        logger.info(f"Saved final model to {final_model_path}")
    
    # Finish wandb
    if cfg.wandb.enable:
        wandb.finish()
    
    logger.info("Sequence ACT Actor warmup training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Sequence ACT Actor for warmup")
    parser.add_argument(
        "--config", 
        type=str, 
        default="lerobot/config/Isaac_lab_kuavo_env/train/only_on_line_learning/sequence_act_actor_warmup.json",
        help="Path to the training configuration JSON file"
    )
    
    args = parser.parse_args()
    
    train_sequence_act_actor(args.config)

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
        observations_sequence = []
        actions_sequence = []
        
        for item in batch:
            # Collect observations
            obs_dict = {}
            for key, value in item.items():
                if key.startswith('observation.'):
                    obs_dict[key] = value
            
            # Collect action
            action = item['action']
            
            observations_sequence.append(obs_dict)
            actions_sequence.append(action)
        
        # Stack observations and actions
        stacked_observations = {}
        for key in observations_sequence[0].keys():
            stacked_observations[key] = torch.stack([obs[key] for obs in observations_sequence], dim=0)
        
        stacked_actions = torch.stack(actions_sequence, dim=0)
        
        return {
            'observations': stacked_observations,
            'actions': stacked_actions
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
    observations: Dict[str, torch.Tensor],
    target_actions: torch.Tensor,
    chunk_size: int,
    obs_history_length: int
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute behavior cloning loss for sequence ACT actor
    
    Args:
        actor: SequenceACTSACActorV2 model
        observations: Batch of observations
        target_actions: Target actions (batch_size, action_dim)
        chunk_size: Action sequence length
        obs_history_length: Observation history length
        
    Returns:
        loss: Total BC loss
        metrics: Dictionary of metrics
    """
    batch_size = target_actions.shape[0]
    device = target_actions.device
    
    # Create observation sequence by repeating the current observation
    # In a real implementation, you would use actual observation history
    obs_sequence = []
    for _ in range(obs_history_length):
        obs_sequence.append(observations)
    
    # Forward pass through the actor
    try:
        # Get action sequence prediction
        predicted_actions, log_probs, action_means = actor.forward(
            obs_sequence,
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
        
        # Total loss
        total_loss = mse_loss + 0.1 * sequence_consistency_loss
        
        # Compute metrics
        with torch.no_grad():
            mse_metric = F.mse_loss(predicted_first_action, target_actions).item()
            mae_metric = F.l1_loss(predicted_first_action, target_actions).item()
            
            # Compute action magnitude metrics
            pred_magnitude = torch.norm(predicted_first_action, dim=1).mean().item()
            target_magnitude = torch.norm(target_actions, dim=1).mean().item()
        
        metrics = {
            'bc/mse': mse_metric,
            'bc/mae': mae_metric,
            'bc/pred_magnitude': pred_magnitude,
            'bc/target_magnitude': target_magnitude,
            'bc/sequence_consistency': sequence_consistency_loss.item() if chunk_size > 1 else 0.0
        }
        
        return total_loss, metrics
        
    except Exception as e:
        logging.error(f"Error in forward pass: {e}")
        # Return a dummy loss to prevent training from crashing
        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        dummy_metrics = {
            'bc/mse': 0.0,
            'bc/mae': 0.0,
            'bc/pred_magnitude': 0.0,
            'bc/target_magnitude': 0.0,
            'bc/sequence_consistency': 0.0
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
    
    # Setup optimizer
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    
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
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Move batch to device
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            target_actions = batch['actions'].to(device)
            
            # Compute loss
            loss, batch_metrics = compute_sequence_bc_loss(
                actor=actor,
                observations=observations,
                target_actions=target_actions,
                chunk_size=getattr(cfg.policy, 'act_chunk_size', 8),
                obs_history_length=getattr(cfg.policy, 'obs_history_length', 5)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(cfg.policy, 'grad_clip_norm') and cfg.policy.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.policy.grad_clip_norm)
            
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
                    val_observations = {k: v.to(device) for k, v in val_batch['observations'].items()}
                    val_target_actions = val_batch['actions'].to(device)
                    
                    val_batch_loss, val_batch_metrics = compute_sequence_bc_loss(
                        actor=actor,
                        observations=val_observations,
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
            
            # Log validation metrics
            if cfg.wandb.enable:
                val_log_dict = {'val/loss': avg_val_loss}
                for key, value in val_metrics.items():
                    val_log_dict[f'val/{key}'] = value
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

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
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from lerobot.common.policies.sac.mlp_bc_model.configuration_mlp_bc import MLPBCConfig
from lerobot.common.policies.sac.mlp_bc_model.modeling_mlp_bc import MLPBCPolicy
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

def create_behavior_cloning_dataset(dataset: LeRobotDataset, batch_size: int = 256, num_workers: int = 4, split_ratio: float = 0.9) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for behavior cloning training"""
    # Filter out episodes that are not completed (next.done == 1)
    # For behavior cloning, we typically want all state-action pairs from expert demonstrations
    
    def collate_fn(batch):
        """Custom collate function to handle the LeRobotDataset format"""
        observations = {}
        actions = []
        
        for item in batch:
            # Collect observations
            for key, value in item.items():
                if key.startswith('observation.'):
                    if key not in observations:
                        observations[key] = []
                    observations[key].append(value)
                elif key == 'action':
                    actions.append(value)
        
        # Stack observations and actions
        for key in observations:
            observations[key] = torch.stack(observations[key], dim=0)
        
        actions = torch.stack(actions, dim=0)
        
        return {
            'observations': observations,
            'actions': actions
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

def train_behavior_cloning(config_path: str):
    """Main training function for behavior cloning"""
    logger = setup_logging()
    logger.info("Starting MLP Behavior Cloning training")
    
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
            dir=str(output_dir),  # Set wandb directory to output_dir
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
    train_loader, val_loader = create_behavior_cloning_dataset(
        dataset, 
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    logger.info(f"Created dataloaders - Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
    
    # Create MLP BC config from the loaded config
    policy_config = MLPBCConfig()
    
    # Update policy config with values from JSON
    # Skip read-only attributes that can't be modified
    readonly_attrs = {'type', 'name'}
    
    # Helper function to convert ConfigObj to dict recursively
    def config_obj_to_dict(obj):
        if hasattr(obj, '__dict__'):
            result = {}
            for k, v in obj.__dict__.items():
                if hasattr(v, '__dict__'):
                    result[k] = config_obj_to_dict(v)
                else:
                    result[k] = v
            return result
        return obj
    
    # Helper function to create PolicyFeature objects from config dict
    def create_policy_features(features_dict):
        if isinstance(features_dict, dict):
            features = {}
            for key, feature_config in features_dict.items():
                if isinstance(feature_config, dict):
                    feature_type = FeatureType(feature_config["type"])
                    shape = tuple(feature_config["shape"])
                    features[key] = PolicyFeature(type=feature_type, shape=shape)
                else:
                    # Already converted
                    features[key] = feature_config
            return features
        return features_dict
    
    # Helper function to convert normalization mapping strings to enums
    def convert_normalization_mapping(norm_mapping):
        if isinstance(norm_mapping, dict):
            converted = {}
            for key, value in norm_mapping.items():
                if isinstance(value, str):
                    converted[key] = NormalizationMode(value)
                else:
                    converted[key] = value
            return converted
        return norm_mapping
    
    for key, value in cfg.policy.__dict__.items():
        if hasattr(policy_config, key) and key not in readonly_attrs:
            try:
                # Convert ConfigObj to dict if needed
                if hasattr(value, '__dict__'):
                    value = config_obj_to_dict(value)
                
                # Special handling for input_features and output_features
                if key in ['input_features', 'output_features']:
                    value = create_policy_features(value)
                # Special handling for normalization_mapping
                elif key == 'normalization_mapping':
                    value = convert_normalization_mapping(value)
                    
                setattr(policy_config, key, value)
            except AttributeError as e:
                # Skip attributes that can't be set
                print(f"Warning: Could not set attribute '{key}': {e}")
                continue
    
    # Set up normalization stats
    if hasattr(cfg.policy, 'dataset_stats') and cfg.policy.dataset_stats:
        # Convert dataset_stats from ConfigObj to dict if needed
        if hasattr(cfg.policy.dataset_stats, '__dict__'):
            policy_config.dataset_stats = config_obj_to_dict(cfg.policy.dataset_stats)
        else:
            policy_config.dataset_stats = cfg.policy.dataset_stats
    
    # Initialize policy
    policy = MLPBCPolicy(config=policy_config).to(device)
    policy.train()
    logger.info("Initialized MLP BC Policy")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
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
            
            # Forward pass and compute loss using policy's compute_loss method
            loss, batch_metrics = policy.compute_loss(observations, target_actions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(cfg.policy, 'grad_clip_norm') and cfg.policy.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.policy.grad_clip_norm)
            
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
                
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': policy_config,
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
            policy.eval()
            val_loss = 0.0
            val_metrics = {}
            val_batches = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_observations = {k: v.to(device) for k, v in val_batch['observations'].items()}
                    val_target_actions = val_batch['actions'].to(device)
                    
                    val_batch_loss, val_batch_metrics = policy.compute_loss(val_observations, val_target_actions)
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
            
            policy.train()
        
        if step >= total_steps:
            break
    
    # Save final model
    if cfg.save_checkpoint:
        final_model_path = output_dir / "final_model.pt"
        
        torch.save({
            'step': step,
            'epoch': epoch,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': policy_config,
            'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
        }, final_model_path)
        
        logger.info(f"Saved final model to {final_model_path}")
    
    # Finish wandb
    if cfg.wandb.enable:
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MLP Behavior Cloning Policy")
    parser.add_argument(
        "--config", 
        type=str, 
        default="lerobot/config/Isaac_lab_kuavo_env/train/only_on_line_learning/mlp_bc_train_grasp.json",
        help="Path to the training configuration JSON file"
    )
    
    args = parser.parse_args()
    
    train_behavior_cloning(args.config)
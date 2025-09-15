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

"""
Evaluation script for Sequence ACT Actor models.

This script loads a trained Sequence ACT Actor model and evaluates its performance
in a robot manipulation environment. It supports both simulation and real robot evaluation.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import draccus
import rospy
from std_msgs.msg import Float64MultiArray

from lerobot.common.policies.sac.modeling_sac_sequence_act_actor import SequenceACTSACActorV2
from lerobot.common.policies.sac.modeling_sac import SACObservationEncoder
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.normalize import NormalizeBuffer
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
from lerobot.scripts.rl.gym_manipulator import make_robot_env, plot_episode_rewards, print_rewards_summary
from lerobot.common.envs.configs import EnvConfig


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


def load_sequence_act_actor(checkpoint_path: str, config: TrainPipelineConfig, device: torch.device) -> SequenceACTSACActorV2:
    """Load trained Sequence ACT Actor from checkpoint"""
    logger = logging.getLogger(__name__)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Create observation encoder
    logger.info("Creating SACObservationEncoder")
    encoder = create_observation_encoder(config)
    encoder = encoder.to(device)
    
    # Get action dimension from output features
    action_feature = getattr(config.policy.output_features, 'action')
    action_dim = action_feature.shape[0]
    
    # Initialize Sequence ACT Actor
    logger.info("Initializing SequenceACTSACActorV2")
    actor = SequenceACTSACActorV2(
        encoder=encoder,
        action_dim=action_dim,
        chunk_size=getattr(config.policy, 'act_chunk_size', 8),
        obs_history_length=getattr(config.policy, 'obs_history_length', 5),
        dim_model=getattr(config.policy, 'act_dim_model', 512),
        n_heads=getattr(config.policy, 'act_n_heads', 8),
        dim_feedforward=getattr(config.policy, 'act_dim_feedforward', 3200),
        n_encoder_layers=getattr(config.policy, 'act_n_encoder_layers', 4),
        n_decoder_layers=getattr(config.policy, 'act_n_decoder_layers', 4),
        dropout=getattr(config.policy, 'act_dropout', 0.1),
        feedforward_activation=getattr(config.policy, 'act_feedforward_activation', 'relu'),
        pre_norm=getattr(config.policy, 'act_pre_norm', False),
        std_min=getattr(config.policy, 'std_min', -5.0),
        std_max=getattr(config.policy, 'std_max', 2.0),
        use_tanh_squash=getattr(config.policy, 'use_tanh_squash', True),
        encoder_is_shared=getattr(config.policy, 'encoder_is_shared', False)
    ).to(device)
    
    # Load model weights
    if 'actor_state_dict' in checkpoint:
        actor.load_state_dict(checkpoint['actor_state_dict'])
        logger.info("Loaded actor weights from checkpoint")
    else:
        logger.warning("No actor_state_dict found in checkpoint, using randomly initialized weights")
    
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        logger.info("Loaded encoder weights from checkpoint")
    else:
        logger.warning("No encoder_state_dict found in checkpoint, using randomly initialized weights")
    
    actor.eval()
    logger.info("Sequence ACT Actor loaded and set to evaluation mode")
    
    return actor


class SequenceACTActorEvaluator:
    """Evaluator for Sequence ACT Actor models"""
    
    def __init__(self, actor: SequenceACTSACActorV2, device: torch.device, chunk_size: int = 8, enable_visualization: bool = True):
        self.actor = actor
        self.device = device
        self.chunk_size = chunk_size
        self.obs_history = []
        self.action_buffer = []
        self.current_action_index = 0
        
        # ROS visualization setup
        self.enable_visualization = enable_visualization
        if self.enable_visualization:
            try:
                # Initialize ROS node if not already initialized
                if not rospy.core.is_initialized():
                    rospy.init_node('sequence_act_evaluator', anonymous=True)
                
                # Publisher for action buffer visualization - 增加队列大小减少消息丢失
                self.action_buffer_pub = rospy.Publisher('/policy/action/eef_pose_marker_all', Float64MultiArray, queue_size=10)
                rospy.loginfo("Sequence ACT Evaluator: ROS visualization enabled")
            except Exception as e:
                rospy.logwarn(f"Failed to initialize ROS visualization: {e}")
                self.enable_visualization = False
        
    def select_action(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Select action using the Sequence ACT Actor
        
        Args:
            observation: Current observation dictionary
            
        Returns:
            Action tensor for the current step
        """
        # If we have actions in buffer, use the next one
        if self.current_action_index < len(self.action_buffer):
            action = self.action_buffer[self.current_action_index]
            self.current_action_index += 1
            return action
        
        # Only update observation history when we need to generate new actions
        # This ensures temporal consistency between observations and actions
        self.obs_history.append(observation)
        
        # Keep only the required history length
        obs_history_length = getattr(self.actor, 'obs_history_length', 5)
        if len(self.obs_history) > obs_history_length:
            self.obs_history = self.obs_history[-obs_history_length:]
        
        # Generate new action sequence
        with torch.no_grad():
            # Convert observation to the format expected by the actor
            obs_for_actor = []
            for obs in self.obs_history:
                # Ensure all tensors are on the correct device
                obs_device = {k: v.to(self.device) for k, v in obs.items()}
                obs_for_actor.append(obs_device)
            
            # Get action sequence from actor
            action_sequence, _, _ = self.actor.forward(
                obs_for_actor,
                return_sequence=True
            )
            
            # Store the action sequence in buffer
            self.action_buffer = action_sequence[0].cpu()  # Take first batch element
            self.current_action_index = 1
            
            # Publish action buffer for visualization
            if self.enable_visualization:
                self._publish_action_buffer_for_visualization(observation)
            
            # Return the first action
            return self.action_buffer[0]
    
    def _publish_action_buffer_for_visualization(self, current_observation: Dict[str, torch.Tensor]):
        """
        Publish action buffer for visualization.
        
        Data format: [left_x, left_y, left_z, right_x, right_y, right_z, cmd_vel_linear_z] * chunk_size
        where each step contains the current base_link eef position + accumulated increments
        
        Note: Actions are scaled to match the environment's incremental control range.
        """
        try:
            # Extract current base_link eef positions from observation
            # obs[23:26] = left eef position in base_link frame
            # obs[26:29] = right eef position in base_link frame
            obs_state = current_observation["observation.state"].cpu().numpy()[0]
            current_left_eef_base = obs_state[23:26]  # [x, y, z]
            current_right_eef_base = obs_state[26:29]  # [x, y, z]
            
            # Prepare data array for the entire chunk
            chunk_data = []
            
            # Initialize current positions
            left_pos = current_left_eef_base.copy()
            right_pos = current_right_eef_base.copy()
            
            # 🔥 增量控制缩放参数 - 与环境中保持一致
            INCREMENT_SCALE = 0.01  # 将action[-1,1]缩放到±0.01m的增量范围
            
            # For each action in the buffer, accumulate the increments
            for i, action in enumerate(self.action_buffer):
                # Extract increments from action and apply scaling
                # action[0:3] = left hand increments (normalized [-1,1])
                # action[3:6] = right hand increments (normalized [-1,1])
                left_increment_action = action[0:3].numpy()
                right_increment_action = action[3:6].numpy()
                
                # 🔥 应用增量缩放 - 与环境中保持一致
                left_increment = left_increment_action * INCREMENT_SCALE
                right_increment = right_increment_action * INCREMENT_SCALE
                
                # Apply increments to current positions
                left_pos += left_increment
                right_pos += right_increment
                
                # Add cmd_vel_linear_z (assuming it's 0 for now, can be extracted from action if available)
                cmd_vel_linear_z = 0.0
                
                # Append to chunk data: [left_x, left_y, left_z, right_x, right_y, right_z, cmd_vel_linear_z]
                chunk_data.extend([
                    left_pos[0], left_pos[1], left_pos[2],  # left eef position
                    right_pos[0], right_pos[1], right_pos[2],  # right eef position
                    cmd_vel_linear_z  # cmd_vel_linear_z
                ])
            
            # Create and publish ROS message
            msg = Float64MultiArray()
            msg.data = chunk_data
            
            if hasattr(self, 'action_buffer_pub'):
                self.action_buffer_pub.publish(msg)
                rospy.logdebug(f"Published action buffer with {len(self.action_buffer)} steps, total data points: {len(chunk_data)}")
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish action buffer for visualization: {e}")

    def reset(self):
        """Reset the evaluator state"""
        self.obs_history = []
        self.action_buffer = []
        self.current_action_index = 0


def evaluate_sequence_act_actor(
    checkpoint_path: str,
    config_path: str,
    env_config_path: str,
    num_episodes: int = 10,
    save_results: bool = True,
    output_dir: str = "./eval_results"
):
    """
    Evaluate a trained Sequence ACT Actor model
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        config_path: Path to the training configuration JSON file
        env_config_path: Path to the environment configuration file
        num_episodes: Number of episodes to evaluate
        save_results: Whether to save evaluation results
        output_dir: Directory to save results
    """
    logger = setup_logging()
    logger.info("Starting Sequence ACT Actor evaluation")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load training configuration
    config = load_config_from_json(config_path)
    logger.info(f"Loaded training config from {config_path}")
    
    # Load environment configuration using draccus parser
    logger.info(f"Loading environment config from {env_config_path}")
    
    # Determine config file type and load accordingly
    config_path = Path(env_config_path)
    if config_path.suffix.lower() == '.json':
        config_type = "json"
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        config_type = "yaml"
    else:
        raise ValueError(f"Unsupported config file type: {config_path.suffix}")
    
    # Load environment configuration using draccus without parsing command line args
    with draccus.config_type(config_type):
        env_cfg = draccus.parse(EnvConfig, env_config_path, args=[])
    logger.info(f"Loaded environment config from {env_config_path}")
    
    # Create environment using the same factory function as gym_manipulator.py
    logger.info("Creating environment")
    env = make_robot_env(env_cfg)
    
    # Load trained model
    logger.info("Loading trained Sequence ACT Actor")
    actor = load_sequence_act_actor(checkpoint_path, config, device)
    
    # Create evaluator with visualization enabled
    evaluator = SequenceACTActorEvaluator(
        actor=actor,
        device=device,
        chunk_size=getattr(config.policy, 'act_chunk_size', 8),
        enable_visualization=True
    )
    
    # Evaluation loop
    logger.info(f"Starting evaluation for {num_episodes} episodes")
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_outcomes = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        # Reset environment and evaluator
        obs, _ = env.reset()
        evaluator.reset()
        
        episode_reward = 0.0
        episode_length = 0
        episode_success = False
        
        # Run episode
        while True:
            # Get action from evaluator
            action = evaluator.select_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # # 检验一下obs的长度是多少
            # print( " === obs === : ", obs["observation.state"].cpu().numpy()[0])
            # print( " === obs length === : ", len(obs["observation.state"].cpu().numpy()[0]))
            
            episode_reward += float(reward)
            episode_length += 1
            
            # Check for success
            if info.get("success", False) or reward >= 1.0:
                episode_success = True
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(episode_success)
        
        # Determine episode outcome
        if episode_success:
            episode_outcomes.append("success")
        elif info.get("box_fallen", False):
            episode_outcomes.append("box_fallen")
        elif truncated and not terminated:
            episode_outcomes.append("timeout")
        else:
            episode_outcomes.append("other_failure")
        
        logger.info(f"Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}, success={episode_success}")
    
    # Calculate statistics
    success_count = sum(episode_successes)
    success_rate = success_count / num_episodes
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_reward = np.std(episode_rewards)
    
    # Print results
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Total Episodes: {num_episodes}")
    logger.info(f"Success Rate: {success_rate:.1%} ({success_count}/{num_episodes})")
    logger.info(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"Average Episode Length: {avg_length:.1f}")
    logger.info(f"Max Reward: {max(episode_rewards):.2f}")
    logger.info(f"Min Reward: {min(episode_rewards):.2f}")
    
    # Outcome breakdown
    outcome_counts = {}
    for outcome in episode_outcomes:
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    
    logger.info("\nOutcome Breakdown:")
    for outcome, count in outcome_counts.items():
        logger.info(f"  {outcome}: {count} ({count/num_episodes:.1%})")
    
    # Save results if requested
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results = {
            'checkpoint_path': str(checkpoint_path),
            'config_path': str(config_path),
            'env_config_path': str(env_config_path),
            'num_episodes': num_episodes,
            'success_rate': success_rate,
            'success_count': success_count,
            'avg_reward': float(avg_reward),
            'std_reward': float(std_reward),
            'avg_length': float(avg_length),
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_lengths': [int(l) for l in episode_lengths],
            'episode_successes': [bool(s) for s in episode_successes],
            'episode_outcomes': episode_outcomes,
            'outcome_counts': outcome_counts
        }
        
        results_file = output_path / f"eval_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        # Generate and save reward analysis plot
        plot_save_path = output_path / f"eval_rewards_analysis_{num_episodes}eps.png"
        plot_episode_rewards(
            episode_rewards, 
            save_path=str(plot_save_path), 
            show_plot=False,
            success_count=success_count,
            total_episodes=num_episodes
        )
        logger.info(f"Reward analysis plot saved to {plot_save_path}")
        
        # Print detailed summary
        print_rewards_summary(episode_rewards, success_count=success_count, total_episodes=num_episodes)
    
    # Close environment
    env.close()
    
    logger.info("Evaluation completed!")
    return results if save_results else None


def main():
    """Main entry point for evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate Sequence ACT Actor model")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to the trained model checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the training configuration JSON file"
    )
    
    parser.add_argument(
        "--env_config", 
        type=str, 
        required=True,
        help="Path to the environment configuration file"
    )
    
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=10,
        help="Number of episodes to evaluate (default: 10)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./eval_results",
        help="Directory to save evaluation results (default: ./eval_results)"
    )
    
    parser.add_argument(
        "--no_save", 
        action="store_true",
        help="Don't save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not Path(args.env_config).exists():
        raise FileNotFoundError(f"Environment config file not found: {args.env_config}")
    
    # Run evaluation
    evaluate_sequence_act_actor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        env_config_path=args.env_config,
        num_episodes=args.num_episodes,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
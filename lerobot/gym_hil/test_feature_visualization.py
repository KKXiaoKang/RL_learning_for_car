#!/usr/bin/env python3
"""
Test script for feature visualization functionality.
This script demonstrates how to use the ResNet10 feature visualization with SAC policy.

Usage:
    python test_feature_visualization.py
    
In another terminal, you can view the feature visualization using:
    rosrun image_view image_view image:=/vision_features/resnet10_features
    
Or use rqt:
    rqt_image_view
"""

import torch
import numpy as np
import rospy
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.gym_hil.envs.rl_kuavo_gym_env import RLKuavoGymEnv


def create_sac_config_with_feature_viz():
    """Create a SAC configuration with feature visualization enabled."""
    config = SACConfig(
        # Vision encoder settings
        vision_encoder_name="helper2424/resnet10",
        freeze_vision_encoder=True,
        shared_encoder=True,
        enable_feature_visualization=True,  # Enable feature visualization
        
        # Input features - must match your environment's observation space
        input_features={
            "observation.image.front": {
                "shape": [3, 224, 224],  # RGB image, 224x224
                "normalization_mode": "VISUAL"
            },
            "observation.state": {
                "shape": [46],  # Agent state dimension (when WBC enabled)
                "normalization_mode": "MIN_MAX"  
            },
            "observation.environment_state": {
                "shape": [7],  # Environment state dimension (when WBC enabled)
                "normalization_mode": "MIN_MAX"
            }
        },
        
        # Output features
        output_features={
            "action": {
                "shape": [18],  # Action dimension (vel: 4 + arm: 14)
                "normalization_mode": "MIN_MAX"
            }
        },
        
        # Device settings
        device="cpu",  # Use CPU for this test
        storage_device="cpu",
        
        # Other important settings
        image_encoder_hidden_dim=64,
        latent_dim=256,
        state_encoder_hidden_dim=256,
        
        # Training parameters (not used for inference but required)
        online_steps=1000,
        online_buffer_capacity=1000,
        offline_buffer_capacity=1000,
    )
    
    return config


def test_feature_visualization():
    """Test the feature visualization functionality."""
    print("Starting feature visualization test...")
    
    # Initialize ROS node
    if not rospy.core.is_initialized():
        rospy.init_node('feature_visualization_test', anonymous=True)
        print("ROS node initialized.")
    
    # Create environment with feature visualization enabled
    print("Creating RLKuavoGymEnv with feature visualization...")
    env = RLKuavoGymEnv(
        debug=True,
        enable_feature_visualization=True,
        wbc_observation_enabled=True
    )
    
    # Create SAC policy with feature visualization
    print("Creating SAC policy with ResNet10 encoder...")
    config = create_sac_config_with_feature_viz()
    policy = SACPolicy(config=config)
    
    print(f"Policy device: {policy.device}")
    print(f"Feature visualization enabled in config: {config.enable_feature_visualization}")
    
    try:
        print("Waiting for initial observation...")
        obs, _ = env.reset()
        print("Initial observation received!")
        print(f"  Pixels shape: {obs['pixels']['front'].shape}")
        print(f"  Agent_pos shape: {obs['agent_pos'].shape}")
        print(f"  Environment_state shape: {obs['environment_state'].shape}")
        
        # Convert observation to the format expected by the policy
        policy_obs = {
            "observation.image.front": torch.from_numpy(obs['pixels']['front']).float().permute(2, 0, 1).unsqueeze(0) / 255.0,  # (1, 3, H, W), normalized to [0,1]
            "observation.state": torch.from_numpy(obs['agent_pos']).float().unsqueeze(0),  # (1, state_dim)
            "observation.environment_state": torch.from_numpy(obs['environment_state']).float().unsqueeze(0)  # (1, env_state_dim)
        }
        
        print("\nRunning policy inference with feature visualization...")
        for step in range(100):  # Run for 100 steps
            print(f"Step {step + 1}/100", end='\r')
            
            # Run policy inference - this will trigger feature extraction and visualization
            with torch.no_grad():
                action = policy.select_action(policy_obs)
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(action.cpu().numpy().flatten())
            
            # Update observation for next step
            policy_obs = {
                "observation.image.front": torch.from_numpy(obs['pixels']['front']).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
                "observation.state": torch.from_numpy(obs['agent_pos']).float().unsqueeze(0),
                "observation.environment_state": torch.from_numpy(obs['environment_state']).float().unsqueeze(0)
            }
            
            # Sleep to make visualization visible
            rospy.sleep(0.1)
            
            if done:
                print(f"\nEpisode finished at step {step + 1}")
                obs, _ = env.reset()
                policy_obs = {
                    "observation.image.front": torch.from_numpy(obs['pixels']['front']).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
                    "observation.state": torch.from_numpy(obs['agent_pos']).float().unsqueeze(0),
                    "observation.environment_state": torch.from_numpy(obs['environment_state']).float().unsqueeze(0)
                }
        
        print("\nFeature visualization test completed successfully!")
        print("\nTo view the feature visualization:")
        print("1. In another terminal, run: rosrun image_view image_view image:=/vision_features/resnet10_features")
        print("2. Or use rqt_image_view and subscribe to /vision_features/resnet10_features")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    test_feature_visualization() 
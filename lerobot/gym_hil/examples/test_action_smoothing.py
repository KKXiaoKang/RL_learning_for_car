#!/usr/bin/env python3
"""
Test script for action smoothing functionality in RLKuavoGymEnv.

This script demonstrates how action smoothing reduces sudden changes in robot commands,
making the robot control more stable and preventing controller failures.

Usage:
    python test_action_smoothing.py
"""

import sys
import rospy
import numpy as np
import time
from typing import Dict, Any

# Add the parent directory to allow imports
sys.path.append('..')

import gym_hil
from gym_hil.envs.rl_kuavo_gym_env import RLKuavoGymEnv


def test_action_smoothing():
    """
    Test function to demonstrate action smoothing.
    """
    print("Testing Action Smoothing in RLKuavoGymEnv...")
    
    # Initialize ROS node (if not already initialized)
    if not rospy.core.is_initialized():
        rospy.init_node('test_action_smoothing', anonymous=True)
    
    # Test different smoothing factors
    smoothing_configs = [
        {"vel_smoothing_factor": 0.0, "arm_smoothing_factor": 0.0, "name": "No Smoothing"},
        {"vel_smoothing_factor": 0.3, "arm_smoothing_factor": 0.4, "name": "Medium Smoothing"},
        {"vel_smoothing_factor": 0.1, "arm_smoothing_factor": 0.2, "name": "High Smoothing"},
    ]
    
    for config in smoothing_configs:
        print(f"\n--- Testing {config['name']} ---")
        print(f"Vel smoothing factor: {config['vel_smoothing_factor']}")
        print(f"Arm smoothing factor: {config['arm_smoothing_factor']}")
        
        # Create environment with specific smoothing parameters
        env = RLKuavoGymEnv(
            debug=True,
            vel_smoothing_factor=config['vel_smoothing_factor'],
            arm_smoothing_factor=config['arm_smoothing_factor']
        )
        
        try:
            # Reset environment
            obs, info = env.reset()
            print("Environment reset successfully")
            
            # Test action smoothing with random actions
            num_steps = 50
            action_changes = []
            
            for step in range(num_steps):
                # Generate random action with some sudden changes
                if step % 10 == 0:  # Every 10 steps, make a sudden change
                    action = np.random.uniform(-1.0, 1.0, size=(18,))
                else:
                    # Small random changes
                    action = np.random.normal(0, 0.1, size=(18,))
                    action = np.clip(action, -1.0, 1.0)
                
                # Step the environment (this will apply smoothing internally)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Calculate action change magnitude
                if step > 0:
                    action_change = np.linalg.norm(action - env.last_action)
                    action_changes.append(action_change)
                
                if step % 10 == 0:
                    print(f"Step {step}: Action change magnitude: {action_change:.4f}")
                
                # Small delay to allow ROS messages to be processed
                time.sleep(0.1)
                
                if terminated or truncated:
                    break
            
            # Print statistics
            if action_changes:
                avg_change = np.mean(action_changes)
                max_change = np.max(action_changes)
                print(f"Average action change: {avg_change:.4f}")
                print(f"Maximum action change: {max_change:.4f}")
            
        except Exception as e:
            print(f"Error during testing: {e}")
        finally:
            env.close()
            print(f"Environment closed for {config['name']}")
    
    print("\n--- Action Smoothing Test Complete ---")


def test_smoothing_parameters():
    """
    Test function to demonstrate the effect of different smoothing parameters.
    """
    print("\n--- Testing Smoothing Parameter Effects ---")
    
    # Initialize ROS node
    if not rospy.core.is_initialized():
        rospy.init_node('test_smoothing_params', anonymous=True)
    
    # Create environment with default smoothing
    env = RLKuavoGymEnv(debug=True)
    
    try:
        obs, info = env.reset()
        
        # Test sequence of actions
        test_actions = [
            np.array([0.5, 0.0, 0.0, 0.0] + [0.0] * 14),  # Forward motion
            np.array([0.0, 0.5, 0.0, 0.0] + [0.0] * 14),  # Side motion
            np.array([0.0, 0.0, 0.5, 0.0] + [0.0] * 14),  # Up motion
            np.array([0.0, 0.0, 0.0, 0.5] + [0.0] * 14),  # Turn motion
            np.array([0.0, 0.0, 0.0, 0.0] + [0.5] * 14),  # Arm motion
        ]
        
        for i, action in enumerate(test_actions):
            print(f"\nStep {i+1}: Testing action {action[:4]} (vel) + {action[4:8]} (arm)")
            
            # Apply smoothing manually to see the effect
            smoothed_action = env._smooth_action(action)
            
            print(f"Original action (vel): {action[:4]}")
            print(f"Smoothed action (vel): {smoothed_action[:4]}")
            print(f"Original action (arm): {action[4:8]}")
            print(f"Smoothed action (arm): {smoothed_action[4:8]}")
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.2)
            
            if terminated or truncated:
                break
    
    except Exception as e:
        print(f"Error during parameter testing: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        test_action_smoothing()
        test_smoothing_parameters()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 
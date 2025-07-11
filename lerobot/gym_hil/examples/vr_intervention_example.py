#!/usr/bin/env python3
"""
Example script demonstrating VR intervention with the RLKuavo environment.

This example shows how to:
1. Set up the RLKuavo environment with VR intervention capability
2. Run a policy while allowing VR intervention
3. Monitor intervention state and VR data availability

Prerequisites:
- ROS environment with Quest3 VR system running
- monitor_quest3.py and quest3_node_incremental.py should be running
- Proper VR setup and calibration

Usage:
    python vr_intervention_example.py
"""

import sys
import rospy
import numpy as np
import gymnasium as gym
from typing import Dict, Any

# Add the parent directory to allow imports
sys.path.append('..')

import gym_hil
from gym_hil.wrappers.factory import make_rl_kuavo_meta_vr_env


def simple_policy(observation: Dict) -> np.ndarray:
    """
    A simple random policy for demonstration.
    In practice, this would be your trained RL policy.
    """
    # Extract environment info for policy decision
    agent_pos = observation['agent_pos']
    env_state = observation['environment_state']
    
    # Simple policy: small random movements
    action = np.random.normal(0, 0.1, size=(18,))  # For 4D vel + 14 arm joints
    action = np.clip(action, -1.0, 1.0)
    
    return action.astype(np.float32)


def run_vr_intervention_demo():
    """
    Main demo function showing VR intervention in action.
    """
    print("Starting VR Intervention Demo with RLKuavo environment...")
    
    # Initialize ROS node (if not already initialized)
    if not rospy.core.is_initialized():
        rospy.init_node('vr_intervention_demo', anonymous=True)
    
    # Create environment with VR intervention capability
    env = make_rl_kuavo_meta_vr_env(
        auto_reset=False,
        intervention_threshold=1.0,  # Right grip threshold for intervention
        rerecord_threshold=1.0,      # Left grip threshold for rerecord
        debug=True,                  # Enable debug logging
    )
    
    print("Environment created successfully!")
    print("\nVR Intervention Instructions:")
    print("- Hold RIGHT grip >= 1.0 to activate intervention")
    print("- Hold LEFT grip >= 1.0 to trigger rerecord")
    print("- Press left second button (Y) for success")
    print("- Press left first button (X) for failure")
    print("\nDuring intervention:")
    print("- Your VR movements will control the robot")
    print("- Environment actions will be ignored")
    print("- VR system publishes to /cmd_vel and /mm/kuavo_arm_traj")
    print("-" * 50)
    
    try:
        num_episodes = 3
        
        for episode in range(num_episodes):
            print(f"\n{'='*20} Episode {episode + 1} {'='*20}")
            
            # Reset environment
            obs, info = env.reset()
            print("Environment reset complete")
            
            episode_reward = 0
            step_count = 0
            max_steps = 200
            
            while step_count < max_steps:
                # Generate policy action
                policy_action = simple_policy(obs)
                
                # Step environment (VR intervention will override if active)
                obs, reward, terminated, truncated, info = env.step(policy_action)
                
                episode_reward += reward
                step_count += 1
                
                # Log intervention status
                if info.get("is_intervention", False):
                    vr_cmd_available = info.get("vr_cmd_vel_available", False)
                    vr_arm_available = info.get("vr_arm_traj_available", False)
                    grip_values = info.get("vr_grip_values", (0.0, 0.0))
                    
                    print(f"Step {step_count}: VR INTERVENTION ACTIVE")
                    print(f"  - Grip values (L/R): {grip_values}")
                    print(f"  - CMD_VEL available: {vr_cmd_available}")
                    print(f"  - ARM_TRAJ available: {vr_arm_available}")
                    print(f"  - Reward: {reward:.3f}")
                
                # Check for episode end conditions
                if terminated or truncated:
                    success = info.get("next.success", False)
                    rerecord = info.get("rerecord_episode", False)
                    
                    if rerecord:
                        print(f"Episode {episode + 1}: RERECORD requested")
                    elif success:
                        print(f"Episode {episode + 1}: SUCCESS (reward: {reward:.3f})")
                    else:
                        print(f"Episode {episode + 1}: TERMINATED")
                    break
                
                # Small delay for readability
                rospy.sleep(0.05)
            
            print(f"Episode {episode + 1} summary:")
            print(f"  - Steps: {step_count}/{max_steps}")
            print(f"  - Total reward: {episode_reward:.3f}")
            print(f"  - Final success: {info.get('succeed', False)}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        env.close()
        print("\nDemo finished and environment closed")


if __name__ == "__main__":
    print("RLKuavo VR Intervention Demo")
    print("Make sure the following are running:")
    print("1. ROS master (roscore)")
    print("2. Quest3 monitor: python monitor_quest3.py")
    print("3. Quest3 node: python quest3_node_incremental.py")
    print("4. Isaac Lab simulation with RLKuavo robot")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    
    try:
        input()
        run_vr_intervention_demo()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0) 
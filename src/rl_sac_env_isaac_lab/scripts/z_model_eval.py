#!/usr/bin/env python3
import torch
import numpy as np
import time
import rospy
from algo.sac import SAC, GymEnvWrapper
from algo.policies import SACPolicy
import os

def main():
    """Main function to evaluate the trained SAC model."""    
    # Define the path to the saved model
    model_path = "/home/lab/RL/src/rl_sac_env_isaac_lab/logs/sac_kuavo_navigation/run_20250618_211000/model.pth"
    
    if not os.path.exists(model_path):
        rospy.logerr(f"Model file not found at {model_path}")
        return

    # Instantiate the environment
    try:
        env = GymEnvWrapper()
        # A small delay to allow ROS connections to be established.
        rospy.loginfo("Waiting for ROS connections to be established...")
        time.sleep(2)
        rospy.loginfo("Environment initialized.")
    except Exception as e:
        rospy.logerr(f"Failed to initialize GymEnvWrapper: {e}")
        return

    # Instantiate the SAC model.
    # Most parameters are for training and can be left as default for evaluation.
    model = SAC(
        policy=SACPolicy,
        env=env,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Load the trained model parameters
    try:
        rospy.loginfo(f"Loading model from {model_path}...")
        model.load(model_path)
        rospy.loginfo("Model loaded successfully.")
    except Exception as e:
        rospy.logerr(f"Failed to load model: {e}")
        return

    # Set the debug flag in the environment to see more output
    env.debug = True

    # Evaluation loop
    num_episodes = 10
    successful_episodes = 0
    collision_episodes = 0
    rospy.loginfo(f"Starting evaluation for {num_episodes} episodes.")

    for i in range(num_episodes):
        rospy.loginfo(f"--- Starting Episode {i+1}/{num_episodes} ---")
        try:
            obs = env.reset()
            if obs is None:
                rospy.logerr("Failed to get initial observation from env.reset().")
                continue
        except (RuntimeError, TimeoutError) as e:
            rospy.logerr(f"Failed to reset environment: {e}")
            # If reset fails, maybe we should stop.
            break

        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        max_episode_steps = 1000  # Same as in training
        info = {}

        while not done and not truncated:
            # Normalize observation using the loaded running mean and std
            obs = np.array(obs, dtype=np.float32)
            
            # Use the statistics loaded with the model
            mean = model.obs_rms.mean
            var = model.obs_rms.var
            
            # Avoid division by zero or NaN
            normalized_obs = (obs - mean) / np.sqrt(var + 1e-8)
            clipped_obs = np.clip(normalized_obs, -10.0, 10.0)
            
            # Predict action deterministically for evaluation
            action = model.policy.predict(clipped_obs, deterministic=True)
            
            try:
                # Step the environment
                next_obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs

                if episode_length >= max_episode_steps:
                    truncated = True
                    rospy.logwarn("Episode truncated due to max steps.")
                
            except (RuntimeError, TimeoutError) as e:
                rospy.logerr(f"Error during env.step: {e}")
                done = True # Exit episode loop on step error

        rospy.loginfo(f"Episode {i+1} finished.")
        rospy.loginfo(f"  - Reward: {episode_reward:.2f}")
        rospy.loginfo(f"  - Length: {episode_length}")
        if info and info.get("collided"):
            rospy.logwarn("  - Termination: Collision")
            collision_episodes += 1
        elif done:
            rospy.loginfo("  - Termination: Reached Goal (done=True)")
            successful_episodes += 1

    rospy.loginfo(f"Evaluation finished. Reached goal {successful_episodes}/{num_episodes} times.")
    rospy.loginfo(f"Collision {collision_episodes}/{num_episodes} times.")

if __name__ == "__main__":
    main() 
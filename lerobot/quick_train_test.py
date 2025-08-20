#!/usr/bin/env python3

import sys
sys.path.append('/home/lab/RL/lerobot')

from lerobot.scripts.rl.train_mlp_bc import train_behavior_cloning

if __name__ == "__main__":
    # Quick test training for just a few steps
    config_path = "/home/lab/RL/lerobot/config/Isaac_lab_kuavo_env/train/only_on_line_learning/mlp_bc_train_grasp_aligned.json"
    
    # Load and modify config for quick test
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Modify for quick testing
    config["steps"] = 5  # Just 5 steps
    config["save_freq"] = 2  # Save every 2 steps
    config["wandb"]["enable"] = False  # Disable wandb for quick test
    
    # Save modified config
    test_config_path = "/home/lab/RL/lerobot/quick_test_config.json"
    with open(test_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Starting quick training test...")
    train_behavior_cloning(test_config_path)
    print("Quick training test completed!")

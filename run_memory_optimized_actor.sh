#!/bin/bash

# Memory optimized Q-chunking ACT-SAC training script

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0

# Set Python memory optimization
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4

# Monitor memory usage
echo "Starting Q-chunking ACT-SAC training with memory optimization..."
echo "Memory usage before training:"
free -h

# Run the actor with optimized config
python3 /home/lab/RL/lerobot/lerobot/scripts/rl/actor.py \
    --config_path /home/lab/RL/lerobot/config/Isaac_lab_kuavo_env/train/only_on_line_learning/eef_obs_32_action_06_memory_optimized.json \

# Monitor memory after
echo "Memory usage after training:"
free -h

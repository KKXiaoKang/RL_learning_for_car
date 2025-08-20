#!/usr/bin/env python3
"""
Script to inspect actor MLP network parameters from SAC model checkpoint
"""

import os
import json
import torch
from safetensors import safe_open
import numpy as np

def inspect_actor_parameters(checkpoint_path):
    """
    检查actor MLP网络参数
    
    Args:
        checkpoint_path: checkpoint目录路径
    """
    print("=" * 60)
    print("Actor MLP Network Parameters Inspector")
    print("=" * 60)
    
    # 检查模型文件路径
    model_path = os.path.join(checkpoint_path, "pretrained_model", "model.safetensors")
    config_path = os.path.join(checkpoint_path, "pretrained_model", "config.json")
    train_config_path = os.path.join(checkpoint_path, "pretrained_model", "train_config.json")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    print(f"Model file: {model_path}")
    print(f"Config file: {config_path}")
    print(f"Train config file: {train_config_path}")
    print()
    
    # 读取配置文件
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        with open(train_config_path, 'r') as f:
            train_config = json.load(f)
        
        print("Model Configuration:")
        print(f"  Policy Type: {config.get('type', 'Unknown')}")
        print(f"  Actor Learning Rate: {config.get('actor_lr', 'Unknown')}")
        print(f"  Actor Network Config: {config.get('actor_network_kwargs', {})}")
        print()
        
    except Exception as e:
        print(f"Error reading config: {e}")
        return
    
    # 读取模型参数
    try:
        print("Loading model parameters...")
        with safe_open(model_path, framework="pt", device="cpu") as f:
            # 获取所有参数键
            keys = f.keys()
            actor_keys = [key for key in keys if 'actor' in key.lower()]
            
            print(f"Total parameters in model: {len(keys)}")
            print(f"Actor-related parameters: {len(actor_keys)}")
            print()
            
            if actor_keys:
                print("Actor MLP Network Parameters:")
                print("-" * 40)
                
                for key in sorted(actor_keys):
                    param = f.get_tensor(key)
                    print(f"Parameter: {key}")
                    print(f"  Shape: {param.shape}")
                    print(f"  Data type: {param.dtype}")
                    print(f"  Device: {param.device}")
                    print(f"  Mean: {param.mean().item():.6f}")
                    print(f"  Std: {param.std().item():.6f}")
                    print(f"  Min: {param.min().item():.6f}")
                    print(f"  Max: {param.max().item():.6f}")
                    
                    # 如果是权重矩阵，显示更多信息
                    if len(param.shape) == 2:
                        print(f"  Weight matrix: {param.shape[0]} x {param.shape[1]}")
                        print(f"  Norm: {torch.norm(param).item():.6f}")
                    elif len(param.shape) == 1:
                        print(f"  Bias vector: {param.shape[0]} elements")
                    
                    print()
                
                # 显示网络结构信息
                print("Network Architecture Analysis:")
                print("-" * 40)
                
                # 尝试推断网络结构
                weight_layers = [key for key in actor_keys if 'weight' in key and len(f.get_tensor(key).shape) == 2]
                bias_layers = [key for key in actor_keys if 'bias' in key]
                
                print(f"Number of weight layers: {len(weight_layers)}")
                print(f"Number of bias layers: {len(bias_layers)}")
                
                if weight_layers:
                    print("\nLayer dimensions:")
                    for i, key in enumerate(sorted(weight_layers)):
                        weight = f.get_tensor(key)
                        print(f"  Layer {i+1}: {weight.shape[1]} -> {weight.shape[0]} (input -> output)")
                
            else:
                print("No actor-related parameters found!")
                print("\nAll available parameters:")
                for key in sorted(keys)[:20]:  # 显示前20个参数
                    param = f.get_tensor(key)
                    print(f"  {key}: {param.shape}")
                if len(keys) > 20:
                    print(f"  ... and {len(keys) - 20} more parameters")
                    
    except Exception as e:
        print(f"Error loading model parameters: {e}")
        return

def main():
    # 默认checkpoint路径
    default_checkpoint = "/home/lab/RL/lerobot/outputs/train/2025-08-18/17-36-30_15_grasp_box_kuavo_reward_mse_demo01_action_06_yes_dataset_temp01_discount095_fps10_seed1000s/checkpoints/last"
    
    # 检查默认路径是否存在
    if os.path.exists(default_checkpoint):
        print(f"Using default checkpoint: {default_checkpoint}")
        inspect_actor_parameters(default_checkpoint)
    else:
        print(f"Default checkpoint not found: {default_checkpoint}")
        print("Please provide the correct checkpoint path.")

if __name__ == "__main__":
    main()

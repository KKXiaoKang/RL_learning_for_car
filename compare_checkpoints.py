#!/usr/bin/env python3
import torch
import numpy as np
from safetensors import safe_open
import json
import os

def load_safetensors_weights(path):
    """加载safetensors格式的模型权重"""
    weights = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    return weights

def compare_model_weights(checkpoint1_path, checkpoint2_path):
    """比较两个checkpoint的模型权重"""
    
    # 加载两个checkpoint的权重
    weights1 = load_safetensors_weights(checkpoint1_path)
    weights2 = load_safetensors_weights(checkpoint2_path)
    
    print("=" * 60)
    print("Policy权重对比分析")
    print("=" * 60)
    
    # 比较权重差异
    total_params = 0
    total_diff = 0
    param_stats = {}
    
    for key in weights1.keys():
        if key in weights2:
            w1 = weights1[key]
            w2 = weights2[key]
            
            # 计算差异
            diff = torch.abs(w1 - w2)
            mean_diff = torch.mean(diff).item()
            max_diff = torch.max(diff).item()
            std_diff = torch.std(diff).item()
            
            # 计算相对变化
            w1_norm = torch.norm(w1).item()
            w2_norm = torch.norm(w2).item()
            relative_change = abs(w2_norm - w1_norm) / (w1_norm + 1e-8)
            
            total_params += w1.numel()
            total_diff += torch.sum(diff).item()
            
            param_stats[key] = {
                'shape': list(w1.shape),
                'mean_abs_diff': mean_diff,
                'max_abs_diff': max_diff,
                'std_diff': std_diff,
                'w1_norm': w1_norm,
                'w2_norm': w2_norm,
                'relative_change': relative_change
            }
            
            print(f"\n{key}:")
            print(f"  Shape: {w1.shape}")
            print(f"  Mean absolute difference: {mean_diff:.6f}")
            print(f"  Max absolute difference: {max_diff:.6f}")
            print(f"  Standard deviation of diff: {std_diff:.6f}")
            print(f"  Weight norm change: {w1_norm:.6f} -> {w2_norm:.6f}")
            print(f"  Relative change: {relative_change:.6f}")
    
    print("\n" + "=" * 60)
    print("总体统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"平均绝对差异: {total_diff/total_params:.6f}")
    
    # 找出变化最大的参数
    print("\n变化最大的参数层 (按相对变化排序):")
    sorted_params = sorted(param_stats.items(), key=lambda x: x[1]['relative_change'], reverse=True)
    for i, (key, stats) in enumerate(sorted_params[:5]):
        print(f"{i+1}. {key}: {stats['relative_change']:.6f}")
    
    return param_stats

def analyze_policy_specific_weights(param_stats):
    """分析与policy相关的特定权重"""
    print("\n" + "=" * 60)
    print("Policy特定组件分析:")
    print("=" * 60)
    
    # 分析actor相关的权重
    actor_weights = {k: v for k, v in param_stats.items() if 'actor' in k.lower()}
    if actor_weights:
        print("\nActor网络权重变化:")
        for key, stats in actor_weights.items():
            print(f"  {key}: 相对变化 {stats['relative_change']:.6f}")
    
    # 分析critic相关的权重  
    critic_weights = {k: v for k, v in param_stats.items() if 'critic' in k.lower()}
    if critic_weights:
        print("\nCritic网络权重变化:")
        for key, stats in critic_weights.items():
            print(f"  {key}: 相对变化 {stats['relative_change']:.6f}")
    
    # 分析temperature相关的权重
    temp_weights = {k: v for k, v in param_stats.items() if 'temperature' in k.lower() or 'temp' in k.lower()}
    if temp_weights:
        print("\nTemperature参数变化:")
        for key, stats in temp_weights.items():
            print(f"  {key}: 相对变化 {stats['relative_change']:.6f}")

def main():
    base_path = "lerobot/outputs/train/2025-07-25/14-35-10_rl_kuavo_MetaVR_hil_1_0725_1415_last_dance/checkpoints"
    
    checkpoint1_path = os.path.join(base_path, "0002000/pretrained_model/model.safetensors")
    checkpoint2_path = os.path.join(base_path, "0010000/pretrained_model/model.safetensors")
    
    if not os.path.exists(checkpoint1_path):
        print(f"找不到checkpoint 2000: {checkpoint1_path}")
        return
    
    if not os.path.exists(checkpoint2_path):
        print(f"找不到checkpoint 10000: {checkpoint2_path}")
        return
    
    print(f"比较checkpoint:")
    print(f"Checkpoint 2000: {checkpoint1_path}")
    print(f"Checkpoint 10000: {checkpoint2_path}")
    
    # 比较权重
    param_stats = compare_model_weights(checkpoint1_path, checkpoint2_path)
    
    # 分析policy特定的权重
    analyze_policy_specific_weights(param_stats)
    
    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    print("1. 如果Actor网络权重变化较大，说明policy分布发生了显著变化")
    print("2. 如果Temperature参数变化较大，说明exploration策略发生了变化")
    print("3. 如果Critic网络权重变化较大，说明价值评估发生了变化")
    print("4. 总体权重变化反映了模型学习的程度")

if __name__ == "__main__":
    main() 
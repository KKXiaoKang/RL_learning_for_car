#!/usr/bin/env python3
"""
Script to visualize the actor MLP network structure and parameter distribution
"""

import os
import json
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
import numpy as np

def visualize_actor_network(checkpoint_path):
    """
    可视化actor MLP网络结构和参数分布
    
    Args:
        checkpoint_path: checkpoint目录路径
    """
    print("=" * 60)
    print("Actor MLP Network Visualizer")
    print("=" * 60)
    
    # 模型和配置文件路径
    model_path = os.path.join(checkpoint_path, "pretrained_model", "model.safetensors")
    config_path = os.path.join(checkpoint_path, "pretrained_model", "config.json")
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print("Error: Model or config file not found!")
        return
    
    # 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Network Configuration:")
    print(f"  Actor Network: {config.get('actor_network_kwargs', {})}")
    print(f"  Learning Rate: {config.get('actor_lr', 'Unknown')}")
    print()
    
    # 加载模型参数
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        actor_keys = [key for key in keys if 'actor' in key.lower()]
        
        # 分析网络结构
        print("Actor Network Structure:")
        print("-" * 40)
        
        # 收集权重和偏置参数
        weights = {}
        biases = {}
        
        for key in actor_keys:
            param = f.get_tensor(key)
            if 'weight' in key and len(param.shape) == 2:
                weights[key] = param
            elif 'bias' in key:
                biases[key] = param
        
        # 打印网络层结构
        print("Network Layers:")
        sorted_weight_keys = sorted(weights.keys())
        for i, key in enumerate(sorted_weight_keys):
            weight = weights[key]
            layer_name = key.replace('actor.', '').replace('.weight', '')
            print(f"  Layer {i+1} ({layer_name}): {weight.shape[1]} → {weight.shape[0]}")
        
        print(f"\nTotal trainable parameters: {sum(param.numel() for param in weights.values()) + sum(param.numel() for param in biases.values())}")
        
        # 可视化参数分布
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Actor Network Parameter Analysis', fontsize=16)
            
            # 1. 权重分布直方图
            ax1 = axes[0, 0]
            all_weights = torch.cat([param.flatten() for param in weights.values()])
            ax1.hist(all_weights.numpy(), bins=50, alpha=0.7, color='blue')
            ax1.set_title('Weight Distribution')
            ax1.set_xlabel('Weight Value')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # 2. 偏置分布直方图
            ax2 = axes[0, 1]
            all_biases = torch.cat([param.flatten() for param in biases.values()])
            ax2.hist(all_biases.numpy(), bins=30, alpha=0.7, color='red')
            ax2.set_title('Bias Distribution')
            ax2.set_xlabel('Bias Value')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # 3. 每层权重范数
            ax3 = axes[1, 0]
            layer_names = []
            layer_norms = []
            for key in sorted_weight_keys:
                weight = weights[key]
                layer_name = key.replace('actor.', '').replace('.weight', '')
                layer_names.append(layer_name.replace('.', '\n'))
                layer_norms.append(torch.norm(weight).item())
            
            bars = ax3.bar(range(len(layer_names)), layer_norms, color='green', alpha=0.7)
            ax3.set_title('Weight Norm by Layer')
            ax3.set_xlabel('Layer')
            ax3.set_ylabel('Frobenius Norm')
            ax3.set_xticks(range(len(layer_names)))
            ax3.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # 在柱状图上显示数值
            for bar, norm in zip(bars, layer_norms):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{norm:.1f}', ha='center', va='bottom', fontsize=8)
            
            # 4. 参数统计表
            ax4 = axes[1, 1]
            ax4.axis('tight')
            ax4.axis('off')
            
            # 创建参数统计表
            stats_data = []
            stats_data.append(['Parameter Type', 'Count', 'Mean', 'Std', 'Min', 'Max'])
            
            # 权重统计
            stats_data.append([
                'Weights',
                f'{all_weights.numel()}',
                f'{all_weights.mean().item():.4f}',
                f'{all_weights.std().item():.4f}',
                f'{all_weights.min().item():.4f}',
                f'{all_weights.max().item():.4f}'
            ])
            
            # 偏置统计
            stats_data.append([
                'Biases',
                f'{all_biases.numel()}',
                f'{all_biases.mean().item():.4f}',
                f'{all_biases.std().item():.4f}',
                f'{all_biases.min().item():.4f}',
                f'{all_biases.max().item():.4f}'
            ])
            
            table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax4.set_title('Parameter Statistics', y=0.8)
            
            plt.tight_layout()
            
            # 保存图像
            save_path = './actor_network_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
            
            # 显示图像（如果在支持的环境中）
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("You may need to install matplotlib: pip install matplotlib")
        
        # 详细的层级分析
        print("\nDetailed Layer Analysis:")
        print("-" * 40)
        
        for key in sorted_weight_keys:
            weight = weights[key]
            layer_name = key.replace('actor.', '').replace('.weight', '')
            
            print(f"\n{layer_name}:")
            print(f"  Shape: {weight.shape}")
            print(f"  Parameters: {weight.numel()}")
            print(f"  Mean: {weight.mean().item():.6f}")
            print(f"  Std: {weight.std().item():.6f}")
            print(f"  Norm: {torch.norm(weight).item():.6f}")
            
            # 对应的偏置
            bias_key = key.replace('weight', 'bias')
            if bias_key in biases:
                bias = biases[bias_key]
                print(f"  Bias shape: {bias.shape}")
                print(f"  Bias mean: {bias.mean().item():.6f}")
                print(f"  Bias std: {bias.std().item():.6f}")

def main():
    # 使用相同的默认checkpoint路径
    default_checkpoint = "/home/lab/RL/lerobot/outputs/train/2025-08-18/17-36-30_15_grasp_box_kuavo_reward_mse_demo01_action_06_yes_dataset_temp01_discount095_fps10_seed1000s/checkpoints/last"
    
    if os.path.exists(default_checkpoint):
        print(f"Analyzing checkpoint: {default_checkpoint}")
        visualize_actor_network(default_checkpoint)
    else:
        print(f"Checkpoint not found: {default_checkpoint}")

if __name__ == "__main__":
    main()

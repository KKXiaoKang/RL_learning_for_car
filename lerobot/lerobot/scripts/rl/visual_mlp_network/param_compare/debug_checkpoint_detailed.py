#!/usr/bin/env python3
"""
Detailed debug script to check checkpoint structure
"""

import torch
import sys
import os
sys.path.insert(0, '/home/lab/RL/lerobot')
os.chdir('/home/lab/RL/lerobot')

try:
    from lerobot.common.policies.sac.mlp_bc_model.configuration_mlp_bc import MLPBCConfig
    torch.serialization.add_safe_globals([MLPBCConfig])
    print("Successfully imported MLPBCConfig")
except ImportError as e:
    print(f"Warning: Could not import MLPBCConfig: {e}")

def debug_checkpoint_detailed(checkpoint_path):
    print(f"=== Loading checkpoint: {checkpoint_path} ===")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"\nTop-level keys: {list(checkpoint.keys())}")
            
            for key, value in checkpoint.items():
                print(f"\n--- {key} ---")
                print(f"Type: {type(value)}")
                
                if key == 'model_state_dict' and isinstance(value, dict):
                    print(f"Contains {len(value)} parameters:")
                    for i, (param_key, param_value) in enumerate(value.items()):
                        if isinstance(param_value, torch.Tensor):
                            print(f"  {i+1:2d}. {param_key:40s}: {str(param_value.shape):15s} | {param_value.dtype} | {param_value.numel():,} params")
                        else:
                            print(f"  {i+1:2d}. {param_key:40s}: {type(param_value)} (not tensor)")
                    
                    # 统计参数
                    total_params = sum(p.numel() for p in value.values() if isinstance(p, torch.Tensor))
                    print(f"\nTotal tensor parameters in model_state_dict: {total_params:,}")
                    
                    # 按类型分组
                    weight_params = [k for k in value.keys() if 'weight' in k]
                    bias_params = [k for k in value.keys() if 'bias' in k]
                    other_params = [k for k in value.keys() if 'weight' not in k and 'bias' not in k]
                    
                    print(f"\nParameter breakdown:")
                    print(f"  Weight parameters: {len(weight_params)}")
                    for w in weight_params:
                        if isinstance(value[w], torch.Tensor):
                            print(f"    {w}: {value[w].shape}")
                    
                    print(f"  Bias parameters: {len(bias_params)}")
                    for b in bias_params:
                        if isinstance(value[b], torch.Tensor):
                            print(f"    {b}: {value[b].shape}")
                    
                    if other_params:
                        print(f"  Other parameters: {len(other_params)}")
                        for o in other_params:
                            print(f"    {o}: {type(value[o])}")
                
                elif key == 'config' and hasattr(value, '__dict__'):
                    print(f"Config attributes:")
                    config_dict = value.__dict__
                    for conf_key, conf_value in config_dict.items():
                        if conf_key in ['actor_network_kwargs', 'policy_kwargs', 'input_features', 'output_features']:
                            print(f"  {conf_key}: {conf_value}")
                
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    try:
                        print(f"Length: {len(value)}")
                    except:
                        pass
                        
                if hasattr(value, 'shape'):
                    print(f"Shape: {value.shape}")
                    
        else:
            print("Checkpoint is not a dictionary")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    checkpoint_path = "/home/lab/RL/lerobot/outputs/mlp_bc_grasp_training_aligned_2/checkpoint_step_2000.pt"
    debug_checkpoint_detailed(checkpoint_path)

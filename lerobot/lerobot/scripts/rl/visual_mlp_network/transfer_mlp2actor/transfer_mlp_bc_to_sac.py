#!/usr/bin/env python3
"""
智能参数转移脚本：从MLP BC模型转移参数到SAC Actor
处理归一化参数差异，确保参数正确映射
"""

import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
from typing import Dict, Any
import json

# Import the MLPBCConfig to make it available for torch.load
try:
    from lerobot.common.policies.sac.mlp_bc_model.configuration_mlp_bc import MLPBCConfig
    torch.serialization.add_safe_globals([MLPBCConfig])
except ImportError:
    print("Warning: Could not import MLPBCConfig")

class MLPBCToSACTransfer:
    """智能参数转移器"""
    
    def __init__(self):
        self.mlp_bc_params = {}
        self.sac_params = {}
        self.transfer_mapping = {}
        
    def load_mlp_bc_checkpoint(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """加载MLP BC checkpoint"""
        print(f"Loading MLP BC checkpoint from: {checkpoint_path}")
        
        try:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except Exception:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model_params = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                model_params = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_params = checkpoint['state_dict']
            else:
                model_params = checkpoint
            
            # 只保留tensor参数
            tensor_params = {}
            for key, value in model_params.items():
                if isinstance(value, torch.Tensor):
                    tensor_params[key] = value
            
            self.mlp_bc_params = tensor_params
            print(f"Loaded {len(tensor_params)} parameters from MLP BC")
            return tensor_params
            
        except Exception as e:
            raise RuntimeError(f"Error loading MLP BC checkpoint: {e}")
    
    def load_sac_checkpoint(self, checkpoint_dir: str) -> Dict[str, torch.Tensor]:
        """加载SAC checkpoint的完整参数"""
        print(f"Loading SAC checkpoint from: {checkpoint_dir}")
        
        model_path = os.path.join(checkpoint_dir, "pretrained_model", "model.safetensors")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            all_params = {}
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_params[key] = f.get_tensor(key)
            
            self.sac_params = all_params
            print(f"Loaded {len(all_params)} parameters from SAC")
            return all_params
            
        except Exception as e:
            raise RuntimeError(f"Error loading SAC parameters: {e}")
    
    def create_transfer_mapping(self) -> Dict[str, str]:
        """创建参数转移映射"""
        print("\nCreating parameter transfer mapping...")
        
        mapping = {}
        
        # 核心网络参数直接映射
        core_mappings = [
            # 编码器参数
            ("actor.encoder.input_normalization.observation_state_min", "actor.encoder.input_normalization.observation_state_min"),
            ("actor.encoder.input_normalization.observation_state_max", "actor.encoder.input_normalization.observation_state_max"),
            ("actor.encoder.state_encoder.0.weight", "actor.encoder.state_encoder.0.weight"),
            ("actor.encoder.state_encoder.0.bias", "actor.encoder.state_encoder.0.bias"),
            ("actor.encoder.state_encoder.1.weight", "actor.encoder.state_encoder.1.weight"),
            ("actor.encoder.state_encoder.1.bias", "actor.encoder.state_encoder.1.bias"),
            
            # 主干网络参数
            ("actor.network.net.0.weight", "actor.network.net.0.weight"),
            ("actor.network.net.0.bias", "actor.network.net.0.bias"),
            ("actor.network.net.1.weight", "actor.network.net.1.weight"),
            ("actor.network.net.1.bias", "actor.network.net.1.bias"),
            ("actor.network.net.3.weight", "actor.network.net.3.weight"),
            ("actor.network.net.3.bias", "actor.network.net.3.bias"),
            ("actor.network.net.4.weight", "actor.network.net.4.weight"),
            ("actor.network.net.4.bias", "actor.network.net.4.bias"),
            
            # 输出层参数
            ("actor.mean_layer.weight", "actor.mean_layer.weight"),
            ("actor.mean_layer.bias", "actor.mean_layer.bias"),
            ("actor.std_layer.weight", "actor.std_layer.weight"),
            ("actor.std_layer.bias", "actor.std_layer.bias"),
        ]
        
        for mlp_key, sac_key in core_mappings:
            if mlp_key in self.mlp_bc_params and sac_key in self.sac_params:
                mapping[mlp_key] = sac_key
                print(f"  {mlp_key} → {sac_key}")
            else:
                if mlp_key not in self.mlp_bc_params:
                    print(f"  WARNING: {mlp_key} not found in MLP BC")
                if sac_key not in self.sac_params:
                    print(f"  WARNING: {sac_key} not found in SAC")
        
        self.transfer_mapping = mapping
        print(f"\nCreated {len(mapping)} parameter mappings")
        
        # 识别需要特殊处理的参数
        mlp_only_params = set(self.mlp_bc_params.keys()) - set(mapping.keys())
        if mlp_only_params:
            print(f"\nParameters only in MLP BC (will be skipped):")
            for param in sorted(mlp_only_params):
                print(f"  - {param}: {self.mlp_bc_params[param].shape}")
        
        return mapping
    
    def transfer_parameters(self) -> Dict[str, torch.Tensor]:
        """执行参数转移"""
        print("\nTransferring parameters...")
        
        # 从SAC模型开始，作为基础
        transferred_params = dict(self.sac_params)
        
        # 转移兼容的参数
        transfer_count = 0
        for mlp_key, sac_key in self.transfer_mapping.items():
            mlp_param = self.mlp_bc_params[mlp_key]
            sac_param = self.sac_params[sac_key]
            
            # 检查形状兼容性
            if mlp_param.shape == sac_param.shape:
                transferred_params[sac_key] = mlp_param.clone()
                transfer_count += 1
                print(f"  ✅ Transferred {mlp_key} → {sac_key} {mlp_param.shape}")
            else:
                print(f"  ❌ Shape mismatch: {mlp_key} {mlp_param.shape} vs {sac_key} {sac_param.shape}")
        
        print(f"\nSuccessfully transferred {transfer_count}/{len(self.transfer_mapping)} parameters")
        return transferred_params
    
    def validate_transfer(self, transferred_params: Dict[str, torch.Tensor]) -> bool:
        """验证转移结果"""
        print("\nValidating parameter transfer...")
        
        # 检查参数数量
        original_actor_params = {k: v for k, v in self.sac_params.items() if 'actor' in k}
        transferred_actor_params = {k: v for k, v in transferred_params.items() if 'actor' in k}
        
        print(f"Original actor parameters: {len(original_actor_params)}")
        print(f"Transferred actor parameters: {len(transferred_actor_params)}")
        
        # 检查所有actor参数都存在
        missing_params = set(original_actor_params.keys()) - set(transferred_actor_params.keys())
        if missing_params:
            print(f"❌ Missing parameters: {missing_params}")
            return False
        
        # 检查形状一致性
        shape_mismatches = []
        for key in original_actor_params.keys():
            if key in transferred_actor_params:
                orig_shape = original_actor_params[key].shape
                new_shape = transferred_actor_params[key].shape
                if orig_shape != new_shape:
                    shape_mismatches.append((key, orig_shape, new_shape))
        
        if shape_mismatches:
            print("❌ Shape mismatches found:")
            for key, orig, new in shape_mismatches:
                print(f"  {key}: {orig} → {new}")
            return False
        
        print("✅ Parameter transfer validation passed!")
        return True
    
    def save_transferred_model(self, transferred_params: Dict[str, torch.Tensor], 
                             output_path: str, metadata: Dict[str, Any] = None):
        """保存转移后的模型"""
        print(f"\nSaving transferred model to: {output_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 准备metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "transfer_info": "Transferred from MLP BC to SAC format",
            "original_mlp_bc_params": str(len(self.mlp_bc_params)),
            "transferred_params": str(len([k for k in self.transfer_mapping.keys()])),
            "total_sac_params": str(len(self.sac_params)),
        })
        
        # 保存为safetensors格式
        save_file(transferred_params, output_path, metadata=metadata)
        print(f"✅ Model saved successfully!")
        
        # 同时保存转移报告
        report_path = output_path.replace('.safetensors', '_transfer_report.json')
        with open(report_path, 'w') as f:
            report = {
                "transfer_mapping": self.transfer_mapping,
                "mlp_bc_only_params": list(set(self.mlp_bc_params.keys()) - set(self.transfer_mapping.keys())),
                "metadata": metadata
            }
            json.dump(report, f, indent=2)
        
        print(f"📄 Transfer report saved to: {report_path}")

def main():
    """主函数"""
    print("🔄 MLP BC to SAC Parameter Transfer Tool")
    print("=" * 60)
    
    # 配置路径
    mlp_bc_checkpoint = "/home/lab/RL/lerobot/outputs/mlp_bc_grasp_training_aligned_2/checkpoint_step_2000.pt"
    sac_checkpoint_dir = "/home/lab/RL/lerobot/outputs/train/2025-08-18/17-36-30_15_grasp_box_kuavo_reward_mse_demo01_action_06_yes_dataset_temp01_discount095_fps10_seed1000s/checkpoints/last"
    output_path = "/home/lab/RL/lerobot/lerobot/scripts/rl/visual_mlp_network/transfer_mlp2actor/transferred_sac_model.safetensors"
    
    # 创建转移器
    transfer_tool = MLPBCToSACTransfer()
    
    try:
        # 加载模型
        transfer_tool.load_mlp_bc_checkpoint(mlp_bc_checkpoint)
        transfer_tool.load_sac_checkpoint(sac_checkpoint_dir)
        
        # 创建转移映射
        transfer_tool.create_transfer_mapping()
        
        # 执行参数转移
        transferred_params = transfer_tool.transfer_parameters()
        
        # 验证转移结果
        if transfer_tool.validate_transfer(transferred_params):
            # 保存转移后的模型
            transfer_tool.save_transferred_model(transferred_params, output_path)
            print("\n🎉 Parameter transfer completed successfully!")
        else:
            print("\n❌ Parameter transfer validation failed!")
            
    except Exception as e:
        print(f"❌ Error during parameter transfer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

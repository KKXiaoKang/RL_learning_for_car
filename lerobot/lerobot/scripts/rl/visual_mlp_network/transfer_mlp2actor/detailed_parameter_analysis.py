#!/usr/bin/env python3
"""
详细分析MLP BC和SAC Actor参数差异
找出具体哪些参数层导致了参数数目不一致
"""

import os
import torch
from safetensors import safe_open
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# Import the MLPBCConfig to make it available for torch.load
try:
    from lerobot.common.policies.sac.mlp_bc_model.configuration_mlp_bc import MLPBCConfig
    torch.serialization.add_safe_globals([MLPBCConfig])
except ImportError:
    print("Warning: Could not import MLPBCConfig")

class DetailedParameterAnalyzer:
    """详细参数分析器"""
    
    def __init__(self):
        self.mlp_bc_params = {}
        self.sac_actor_params = {}
    
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
            return tensor_params
            
        except Exception as e:
            raise RuntimeError(f"Error loading MLP BC checkpoint: {e}")
    
    def load_sac_actor_parameters(self, checkpoint_dir: str) -> Dict[str, torch.Tensor]:
        """加载SAC actor参数"""
        print(f"Loading SAC actor parameters from: {checkpoint_dir}")
        
        model_path = os.path.join(checkpoint_dir, "pretrained_model", "model.safetensors")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            actor_params = {}
            with safe_open(model_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                actor_keys = [key for key in keys if 'actor' in key.lower()]
                
                for key in actor_keys:
                    actor_params[key] = f.get_tensor(key)
            
            self.sac_actor_params = actor_params
            return actor_params
            
        except Exception as e:
            raise RuntimeError(f"Error loading SAC parameters: {e}")
    
    def analyze_parameter_structure(self):
        """分析参数结构"""
        print("\n" + "="*80)
        print("DETAILED PARAMETER STRUCTURE ANALYSIS")
        print("="*80)
        
        # 分析MLP BC参数
        print(f"\n📊 MLP BC Parameters (Total: {len(self.mlp_bc_params)})")
        print("-" * 60)
        mlp_bc_total_params = 0
        mlp_bc_groups = self._group_parameters(self.mlp_bc_params, "MLP_BC")
        
        for group, params in mlp_bc_groups.items():
            group_params = sum(p.numel() for p in params.values())
            mlp_bc_total_params += group_params
            print(f"\n{group}:")
            for name, param in sorted(params.items()):
                print(f"  {name:50} {str(param.shape):15} {param.numel():>8} params")
            print(f"  {'='*50} {'Subtotal:':15} {group_params:>8} params")
        
        print(f"\n🎯 MLP BC Total Parameters: {mlp_bc_total_params:,}")
        
        # 分析SAC Actor参数
        print(f"\n📊 SAC Actor Parameters (Total: {len(self.sac_actor_params)})")
        print("-" * 60)
        sac_actor_total_params = 0
        sac_actor_groups = self._group_parameters(self.sac_actor_params, "SAC_Actor")
        
        for group, params in sac_actor_groups.items():
            group_params = sum(p.numel() for p in params.values())
            sac_actor_total_params += group_params
            print(f"\n{group}:")
            for name, param in sorted(params.items()):
                print(f"  {name:50} {str(param.shape):15} {param.numel():>8} params")
            print(f"  {'='*50} {'Subtotal:':15} {group_params:>8} params")
        
        print(f"\n🎯 SAC Actor Total Parameters: {sac_actor_total_params:,}")
        
        # 计算差异
        param_diff = abs(mlp_bc_total_params - sac_actor_total_params)
        print(f"\n❗ Parameter Difference: {param_diff:,} parameters")
        
        return mlp_bc_groups, sac_actor_groups, param_diff
    
    def _group_parameters(self, params: Dict[str, torch.Tensor], model_name: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """按功能模块分组参数"""
        groups = defaultdict(dict)
        
        for name, param in params.items():
            if 'normalization' in name or 'normalize' in name:
                # 归一化相关参数
                groups['Normalization'][name] = param
            elif 'encoder' in name and 'state_encoder' in name:
                # 状态编码器
                groups['State Encoder'][name] = param
            elif 'encoder' in name and 'image' in name:
                # 图像编码器
                groups['Image Encoder'][name] = param
            elif 'network' in name:
                # 主干网络
                groups['Main Network'][name] = param
            elif 'mean_layer' in name:
                # 均值层
                groups['Mean Layer'][name] = param
            elif 'std_layer' in name:
                # 标准差层
                groups['Std Layer'][name] = param
            else:
                # 其他参数
                groups['Other'][name] = param
        
        return dict(groups)
    
    def find_missing_parameters(self):
        """找出缺失的参数"""
        print("\n" + "="*80)
        print("MISSING PARAMETER ANALYSIS")
        print("="*80)
        
        # 提取参数名（去掉前缀）
        mlp_bc_simple_names = set()
        sac_actor_simple_names = set()
        
        for name in self.mlp_bc_params.keys():
            # 提取核心参数名
            simple_name = self._extract_core_param_name(name)
            mlp_bc_simple_names.add(simple_name)
        
        for name in self.sac_actor_params.keys():
            simple_name = self._extract_core_param_name(name)
            sac_actor_simple_names.add(simple_name)
        
        # 找出差异
        only_in_mlp_bc = mlp_bc_simple_names - sac_actor_simple_names
        only_in_sac_actor = sac_actor_simple_names - mlp_bc_simple_names
        
        print(f"\n🔍 Parameters only in MLP BC ({len(only_in_mlp_bc)}):")
        for name in sorted(only_in_mlp_bc):
            # 找到原始参数名和形状
            original_names = [k for k in self.mlp_bc_params.keys() if self._extract_core_param_name(k) == name]
            for orig_name in original_names:
                param = self.mlp_bc_params[orig_name]
                print(f"  {orig_name:50} {str(param.shape):15} {param.numel():>8} params")
        
        print(f"\n🔍 Parameters only in SAC Actor ({len(only_in_sac_actor)}):")
        for name in sorted(only_in_sac_actor):
            original_names = [k for k in self.sac_actor_params.keys() if self._extract_core_param_name(k) == name]
            for orig_name in original_names:
                param = self.sac_actor_params[orig_name]
                print(f"  {orig_name:50} {str(param.shape):15} {param.numel():>8} params")
        
        return only_in_mlp_bc, only_in_sac_actor
    
    def _extract_core_param_name(self, full_name: str) -> str:
        """提取核心参数名"""
        # 去掉模型前缀，保留核心结构
        name = full_name
        
        # 去掉常见前缀
        prefixes_to_remove = ['actor.', 'critic.', 'policy.', 'model.']
        for prefix in prefixes_to_remove:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        
        return name
    
    def analyze_shape_compatibility(self):
        """分析形状兼容性"""
        print("\n" + "="*80)
        print("SHAPE COMPATIBILITY ANALYSIS")
        print("="*80)
        
        # 找到对应的参数对
        matched_pairs = []
        mlp_bc_unmatched = []
        sac_actor_unmatched = []
        
        # 简化的匹配策略
        for mlp_name, mlp_param in self.mlp_bc_params.items():
            mlp_core = self._extract_core_param_name(mlp_name)
            
            # 寻找SAC中的对应参数
            sac_match = None
            for sac_name, sac_param in self.sac_actor_params.items():
                sac_core = self._extract_core_param_name(sac_name)
                if mlp_core == sac_core:
                    sac_match = (sac_name, sac_param)
                    break
            
            if sac_match:
                matched_pairs.append((mlp_name, mlp_param, sac_match[0], sac_match[1]))
            else:
                mlp_bc_unmatched.append((mlp_name, mlp_param))
        
        # 找到SAC中未匹配的参数
        matched_sac_names = {pair[2] for pair in matched_pairs}
        for sac_name, sac_param in self.sac_actor_params.items():
            if sac_name not in matched_sac_names:
                sac_actor_unmatched.append((sac_name, sac_param))
        
        # 打印匹配结果
        print(f"\n✅ Matched Parameter Pairs ({len(matched_pairs)}):")
        print(f"{'MLP BC':50} {'Shape':15} {'SAC Actor':50} {'Shape':15} {'Compatible':10}")
        print("-" * 140)
        
        compatible_count = 0
        for mlp_name, mlp_param, sac_name, sac_param in matched_pairs:
            compatible = mlp_param.shape == sac_param.shape
            if compatible:
                compatible_count += 1
            
            compat_str = "✅ Yes" if compatible else "❌ No"
            print(f"{mlp_name:50} {str(mlp_param.shape):15} {sac_name:50} {str(sac_param.shape):15} {compat_str}")
        
        print(f"\n📊 Compatibility Summary: {compatible_count}/{len(matched_pairs)} pairs are shape-compatible")
        
        # 打印未匹配的参数
        if mlp_bc_unmatched:
            print(f"\n❌ Unmatched MLP BC Parameters ({len(mlp_bc_unmatched)}):")
            for name, param in mlp_bc_unmatched:
                print(f"  {name:50} {str(param.shape):15} {param.numel():>8} params")
        
        if sac_actor_unmatched:
            print(f"\n❌ Unmatched SAC Actor Parameters ({len(sac_actor_unmatched)}):")
            for name, param in sac_actor_unmatched:
                print(f"  {name:50} {str(param.shape):15} {param.numel():>8} params")
        
        return matched_pairs, mlp_bc_unmatched, sac_actor_unmatched
    
    def generate_replacement_strategy(self):
        """生成替换策略"""
        print("\n" + "="*80)
        print("REPLACEMENT STRATEGY")
        print("="*80)
        
        matched_pairs, mlp_bc_unmatched, sac_actor_unmatched = self.analyze_shape_compatibility()
        
        print("\n🔄 Replacement Plan:")
        print("1. Direct Parameter Transfer (Shape Compatible):")
        
        transferable_pairs = []
        for mlp_name, mlp_param, sac_name, sac_param in matched_pairs:
            if mlp_param.shape == sac_param.shape:
                transferable_pairs.append((mlp_name, sac_name))
                print(f"   {mlp_name} → {sac_name}")
        
        print(f"\n2. Parameters Requiring Special Handling:")
        if mlp_bc_unmatched:
            print("   MLP BC Unique Parameters (may need to be ignored):")
            for name, param in mlp_bc_unmatched:
                print(f"     - {name} ({param.shape})")
        
        if sac_actor_unmatched:
            print("   SAC Actor Unique Parameters (may need default initialization):")
            for name, param in sac_actor_unmatched:
                print(f"     - {name} ({param.shape})")
        
        # 检查是否可以直接替换
        total_mlp_bc_transferable = sum(
            self.mlp_bc_params[mlp_name].numel() 
            for mlp_name, _ in transferable_pairs
        )
        
        total_sac_actor_params = sum(p.numel() for p in self.sac_actor_params.values())
        
        print(f"\n📊 Transfer Coverage:")
        print(f"   Transferable parameters: {total_mlp_bc_transferable:,}")
        print(f"   Total SAC Actor parameters: {total_sac_actor_params:,}")
        print(f"   Coverage: {total_mlp_bc_transferable/total_sac_actor_params*100:.1f}%")
        
        return transferable_pairs, mlp_bc_unmatched, sac_actor_unmatched


def main():
    """主函数"""
    print("🔍 Detailed Parameter Structure Analyzer")
    print("="*80)
    
    # 文件路径
    mlp_bc_checkpoint = "/home/lab/RL/lerobot/outputs/mlp_bc_grasp_training_aligned_2/checkpoint_step_2000.pt"
    sac_checkpoint_dir = "/home/lab/RL/lerobot/outputs/train/2025-08-18/17-36-30_15_grasp_box_kuavo_reward_mse_demo01_action_06_yes_dataset_temp01_discount095_fps10_seed1000s/checkpoints/last"
    
    # 创建分析器
    analyzer = DetailedParameterAnalyzer()
    
    try:
        # 加载参数
        analyzer.load_mlp_bc_checkpoint(mlp_bc_checkpoint)
        analyzer.load_sac_actor_parameters(sac_checkpoint_dir)
        
        # 执行分析
        analyzer.analyze_parameter_structure()
        analyzer.find_missing_parameters()
        analyzer.generate_replacement_strategy()
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

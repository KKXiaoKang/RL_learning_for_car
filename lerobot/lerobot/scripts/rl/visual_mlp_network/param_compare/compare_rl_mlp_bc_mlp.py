#!/usr/bin/env python3
"""
Script to compare MLP network parameters between different models
Compares MLP BC model with SAC actor network parameters
"""

import os
import json
import torch
from safetensors import safe_open
import numpy as np
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization will be disabled")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available, some visualizations will be simplified")
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# Import the MLPBCConfig to make it available for torch.load
try:
    from lerobot.common.policies.sac.mlp_bc_model.configuration_mlp_bc import MLPBCConfig
    # Add safe globals for torch loading
    torch.serialization.add_safe_globals([MLPBCConfig])
except ImportError:
    print("Warning: Could not import MLPBCConfig, using weights_only=False")

class NetworkParameterComparator:
    """Compare parameters between different MLP networks"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def load_mlp_bc_checkpoint(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """
        加载MLP BC模型的checkpoint参数
        
        Args:
            checkpoint_path: MLP BC checkpoint文件路径 (.pt文件)
        
        Returns:
            参数字典
        """
        print(f"Loading MLP BC checkpoint from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            # First try to load with weights_only=True (safer)
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except Exception:
                # If that fails, try with weights_only=False for custom classes
                print("Warning: Loading checkpoint with weights_only=False due to custom classes")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # 通常模型参数在'model'、'model_state_dict'或'state_dict'键下
            if 'model_state_dict' in checkpoint:
                model_params = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                model_params = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_params = checkpoint['state_dict']
            else:
                # 如果checkpoint就是参数字典
                model_params = checkpoint
            
            # 确保我们获得的是参数字典
            if not isinstance(model_params, dict):
                print(f"Warning: Unexpected model_params type: {type(model_params)}")
                print(f"Available keys in checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
                # 尝试从模型对象中提取state_dict
                if hasattr(model_params, 'state_dict'):
                    model_params = model_params.state_dict()
                else:
                    raise RuntimeError(f"Cannot extract parameters from checkpoint type: {type(model_params)}")
            
            # 调试信息：显示参数的类型
            print(f"Loaded {len(model_params)} items from MLP BC model")
            print("Parameter types:")
            for key, value in list(model_params.items())[:10]:  # 只显示前10个
                print(f"  {key}: {type(value)} - {getattr(value, 'shape', 'N/A')}")
            
            # 特别检查model_state_dict的内容
            if hasattr(model_params, 'items'):
                model_state_dict = None
                for key, value in model_params.items():
                    if key == 'model_state_dict' and isinstance(value, dict):
                        print(f"\nFound model_state_dict with {len(value)} items:")
                        for subkey, subvalue in list(value.items())[:5]:
                            print(f"    {subkey}: {type(subvalue)} - {getattr(subvalue, 'shape', 'N/A')}")
                        model_state_dict = value
                        break
                
                # 如果找到了model_state_dict，使用它作为模型参数
                if model_state_dict is not None:
                    model_params = model_state_dict
                    print(f"Using model_state_dict as model parameters")
                elif 'model_state_dict' in model_params:
                    # 直接从当前字典中提取model_state_dict
                    model_params = model_params['model_state_dict']
                    print(f"Extracted model_state_dict from checkpoint")
            
            # 过滤掉非tensor的参数
            tensor_params = {}
            for key, value in model_params.items():
                if isinstance(value, torch.Tensor):
                    tensor_params[key] = value
                else:
                    print(f"Skipping non-tensor parameter: {key} (type: {type(value)})")
            
            print(f"Filtered to {len(tensor_params)} tensor parameters")
            
            # 显示找到的tensor参数名称
            if tensor_params:
                print("Found tensor parameters:")
                for key in sorted(tensor_params.keys()):
                    print(f"  {key}: {tensor_params[key].shape}")
            
            return tensor_params
            
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
    
    def load_sac_actor_parameters(self, checkpoint_dir: str) -> Dict[str, torch.Tensor]:
        """
        加载SAC actor网络的参数
        
        Args:
            checkpoint_dir: SAC checkpoint目录路径
        
        Returns:
            actor相关参数字典
        """
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
            
            print(f"Loaded {len(actor_params)} actor parameters from SAC model")
            return actor_params
            
        except Exception as e:
            raise RuntimeError(f"Error loading SAC parameters: {e}")
    
    def extract_layer_info(self, params: Dict[str, torch.Tensor], model_name: str) -> Dict[str, Any]:
        """
        提取网络层信息
        
        Args:
            params: 参数字典
            model_name: 模型名称
        
        Returns:
            层信息字典
        """
        layer_info = {
            'weight_layers': [],
            'bias_layers': [],
            'layer_dimensions': [],
            'total_params': 0,
            'model_name': model_name
        }
        
        # 按层名排序
        weight_keys = sorted([k for k in params.keys() if 'weight' in k and len(params[k].shape) == 2])
        bias_keys = sorted([k for k in params.keys() if 'bias' in k and len(params[k].shape) == 1])
        
        layer_info['weight_layers'] = weight_keys
        layer_info['bias_layers'] = bias_keys
        
        # 提取层维度
        for key in weight_keys:
            weight = params[key]
            layer_info['layer_dimensions'].append((weight.shape[1], weight.shape[0]))  # (input, output)
        
        # 计算总参数量 (确保都是tensor)
        layer_info['total_params'] = sum(param.numel() for param in params.values() if isinstance(param, torch.Tensor))
        
        return layer_info
    
    def compute_parameter_statistics(self, params: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        计算参数统计信息
        
        Args:
            params: 参数字典
        
        Returns:
            统计信息字典
        """
        stats = {}
        
        for name, param in params.items():
            # 确保只处理tensor参数
            if not isinstance(param, torch.Tensor):
                print(f"Warning: Skipping non-tensor parameter in statistics: {name} (type: {type(param)})")
                continue
                
            param_stats = {
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'norm': torch.norm(param).item(),
                'shape': list(param.shape),
                'num_elements': param.numel()
            }
            
            # 如果是权重矩阵，计算额外统计信息
            if len(param.shape) == 2:
                param_stats['frobenius_norm'] = torch.norm(param, 'fro').item()
                param_stats['spectral_norm'] = torch.norm(param, 2).item()
            
            stats[name] = param_stats
        
        return stats
    
    def compare_parameters(self, params1: Dict[str, torch.Tensor], params2: Dict[str, torch.Tensor], 
                          model1_name: str, model2_name: str) -> Dict[str, Any]:
        """
        对比两个模型的参数
        
        Args:
            params1: 第一个模型参数
            params2: 第二个模型参数
            model1_name: 第一个模型名称
            model2_name: 第二个模型名称
        
        Returns:
            对比结果字典
        """
        print(f"\nComparing parameters between {model1_name} and {model2_name}")
        print("=" * 60)
        
        # 获取层信息
        layer_info1 = self.extract_layer_info(params1, model1_name)
        layer_info2 = self.extract_layer_info(params2, model2_name)
        
        # 计算统计信息
        stats1 = self.compute_parameter_statistics(params1)
        stats2 = self.compute_parameter_statistics(params2)
        
        comparison = {
            'model1_info': layer_info1,
            'model2_info': layer_info2,
            'model1_stats': stats1,
            'model2_stats': stats2,
            'layer_comparison': [],
            'global_comparison': {}
        }
        
        # 网络结构对比
        print(f"\nNetwork Architecture Comparison:")
        print(f"  {model1_name}:")
        print(f"    Total parameters: {layer_info1['total_params']:,}")
        print(f"    Weight layers: {len(layer_info1['weight_layers'])}")
        print(f"    Bias layers: {len(layer_info1['bias_layers'])}")
        print(f"    Layer dimensions: {layer_info1['layer_dimensions']}")
        
        print(f"  {model2_name}:")
        print(f"    Total parameters: {layer_info2['total_params']:,}")
        print(f"    Weight layers: {len(layer_info2['weight_layers'])}")
        print(f"    Bias layers: {len(layer_info2['bias_layers'])}")
        print(f"    Layer dimensions: {layer_info2['layer_dimensions']}")
        
        # 全局统计对比
        self._compare_global_statistics(stats1, stats2, model1_name, model2_name, comparison)
        
        # 逐层对比
        self._compare_layer_by_layer(stats1, stats2, model1_name, model2_name, comparison)
        
        return comparison
    
    def _compare_global_statistics(self, stats1: Dict, stats2: Dict, 
                                 model1_name: str, model2_name: str, 
                                 comparison: Dict):
        """对比全局统计信息"""
        print(f"\nGlobal Statistics Comparison:")
        print("-" * 40)
        
        global_stats = {
            'total_params_diff': abs(sum(s['num_elements'] for s in stats1.values()) - 
                                   sum(s['num_elements'] for s in stats2.values())),
            'mean_param_values': {
                model1_name: np.mean([s['mean'] for s in stats1.values()]),
                model2_name: np.mean([s['mean'] for s in stats2.values()])
            },
            'mean_param_stds': {
                model1_name: np.mean([s['std'] for s in stats1.values()]),
                model2_name: np.mean([s['std'] for s in stats2.values()])
            }
        }
        
        comparison['global_comparison'] = global_stats
        
        print(f"Parameter count difference: {global_stats['total_params_diff']:,}")
        print(f"Average parameter mean:")
        print(f"  {model1_name}: {global_stats['mean_param_values'][model1_name]:.6f}")
        print(f"  {model2_name}: {global_stats['mean_param_values'][model2_name]:.6f}")
        print(f"Average parameter std:")
        print(f"  {model1_name}: {global_stats['mean_param_stds'][model1_name]:.6f}")
        print(f"  {model2_name}: {global_stats['mean_param_stds'][model2_name]:.6f}")
    
    def _compare_layer_by_layer(self, stats1: Dict, stats2: Dict, 
                               model1_name: str, model2_name: str, 
                               comparison: Dict):
        """逐层参数对比"""
        print(f"\nLayer-by-Layer Comparison:")
        print("-" * 40)
        
        # 尝试匹配对应的层
        weight_layers1 = [k for k in stats1.keys() if 'weight' in k]
        weight_layers2 = [k for k in stats2.keys() if 'weight' in k]
        
        for i, (layer1, layer2) in enumerate(zip(sorted(weight_layers1), sorted(weight_layers2))):
            if i < min(len(weight_layers1), len(weight_layers2)):
                layer_comp = self._compare_single_layer(stats1[layer1], stats2[layer2], 
                                                       layer1, layer2, model1_name, model2_name)
                comparison['layer_comparison'].append(layer_comp)
    
    def _compare_single_layer(self, layer1_stats: Dict, layer2_stats: Dict, 
                            layer1_name: str, layer2_name: str,
                            model1_name: str, model2_name: str) -> Dict:
        """对比单个层的参数"""
        print(f"\nComparing layers: {layer1_name} vs {layer2_name}")
        
        layer_comparison = {
            'layer1_name': layer1_name,
            'layer2_name': layer2_name,
            'shape_match': layer1_stats['shape'] == layer2_stats['shape'],
            'differences': {}
        }
        
        # 对比统计量
        metrics = ['mean', 'std', 'min', 'max', 'norm']
        for metric in metrics:
            if metric in layer1_stats and metric in layer2_stats:
                diff = abs(layer1_stats[metric] - layer2_stats[metric])
                rel_diff = diff / (abs(layer1_stats[metric]) + 1e-8) * 100
                
                layer_comparison['differences'][metric] = {
                    'absolute_diff': diff,
                    'relative_diff_percent': rel_diff,
                    f'{model1_name}_value': layer1_stats[metric],
                    f'{model2_name}_value': layer2_stats[metric]
                }
                
                print(f"  {metric:>15}: {layer1_stats[metric]:>10.6f} vs {layer2_stats[metric]:>10.6f} "
                      f"(diff: {diff:>8.6f}, {rel_diff:>6.2f}%)")
        
        return layer_comparison
    
    def visualize_comparison(self, comparison: Dict, save_path: str = None):
        """可视化对比结果"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping visualization")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MLP Network Parameter Comparison', fontsize=16)
        
        model1_name = comparison['model1_info']['model_name']
        model2_name = comparison['model2_info']['model_name']
        
        # 1. 参数分布对比
        ax1 = axes[0, 0]
        metrics = ['mean', 'std', 'min', 'max']
        model1_values = []
        model2_values = []
        
        for layer_comp in comparison['layer_comparison']:
            if layer_comp['differences']:
                for metric in metrics:
                    if metric in layer_comp['differences']:
                        model1_values.append(layer_comp['differences'][metric][f'{model1_name}_value'])
                        model2_values.append(layer_comp['differences'][metric][f'{model2_name}_value'])
        
        if model1_values and model2_values:
            ax1.scatter(model1_values, model2_values, alpha=0.6)
            ax1.plot([min(model1_values + model2_values), max(model1_values + model2_values)],
                    [min(model1_values + model2_values), max(model1_values + model2_values)], 'r--')
            ax1.set_xlabel(f'{model1_name} Parameter Values')
            ax1.set_ylabel(f'{model2_name} Parameter Values')
            ax1.set_title('Parameter Value Correlation')
            ax1.grid(True, alpha=0.3)
        
        # 2. 相对差异热力图
        ax2 = axes[0, 1]
        if comparison['layer_comparison']:
            diff_matrix = []
            layer_names = []
            
            for layer_comp in comparison['layer_comparison']:
                if layer_comp['differences']:
                    layer_names.append(layer_comp['layer1_name'].split('.')[-1])
                    row = []
                    for metric in metrics:
                        if metric in layer_comp['differences']:
                            row.append(layer_comp['differences'][metric]['relative_diff_percent'])
                        else:
                            row.append(0)
                    diff_matrix.append(row)
            
            if diff_matrix:
                if SEABORN_AVAILABLE:
                    sns.heatmap(diff_matrix, annot=True, fmt='.2f', 
                               xticklabels=metrics, yticklabels=layer_names,
                               ax=ax2, cmap='viridis')
                else:
                    # Fallback to matplotlib only
                    im = ax2.imshow(diff_matrix, cmap='viridis', aspect='auto')
                    ax2.set_xticks(range(len(metrics)))
                    ax2.set_xticklabels(metrics)
                    ax2.set_yticks(range(len(layer_names)))
                    ax2.set_yticklabels(layer_names)
                    plt.colorbar(im, ax=ax2)
                ax2.set_title('Relative Differences (%) by Layer')
        
        # 3. 网络架构对比
        ax3 = axes[1, 0]
        dims1 = comparison['model1_info']['layer_dimensions']
        dims2 = comparison['model2_info']['layer_dimensions']
        
        if dims1 and dims2:
            x = range(max(len(dims1), len(dims2)))
            
            input_dims1 = [d[0] for d in dims1] + [0] * (len(x) - len(dims1))
            output_dims1 = [d[1] for d in dims1] + [0] * (len(x) - len(dims1))
            input_dims2 = [d[0] for d in dims2] + [0] * (len(x) - len(dims2))
            output_dims2 = [d[1] for d in dims2] + [0] * (len(x) - len(dims2))
            
            ax3.plot(x, input_dims1, 'b-o', label=f'{model1_name} Input', alpha=0.7)
            ax3.plot(x, output_dims1, 'b--s', label=f'{model1_name} Output', alpha=0.7)
            ax3.plot(x, input_dims2, 'r-o', label=f'{model2_name} Input', alpha=0.7)
            ax3.plot(x, output_dims2, 'r--s', label=f'{model2_name} Output', alpha=0.7)
            
            ax3.set_xlabel('Layer Index')
            ax3.set_ylabel('Dimension Size')
            ax3.set_title('Network Architecture Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 全局统计对比
        ax4 = axes[1, 1]
        global_comp = comparison['global_comparison']
        
        categories = ['Mean Parameter Values', 'Mean Parameter Stds']
        model1_global = [global_comp['mean_param_values'][model1_name], 
                        global_comp['mean_param_stds'][model1_name]]
        model2_global = [global_comp['mean_param_values'][model2_name], 
                        global_comp['mean_param_stds'][model2_name]]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x_pos - width/2, model1_global, width, label=model1_name, alpha=0.7)
        ax4.bar(x_pos + width/2, model2_global, width, label=model2_name, alpha=0.7)
        
        ax4.set_xlabel('Statistics')
        ax4.set_ylabel('Values')
        ax4.set_title('Global Statistics Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison visualization saved to: {save_path}")
        
        plt.show()
    
    def save_comparison_report(self, comparison: Dict, report_path: str):
        """保存对比报告"""
        with open(report_path, 'w') as f:
            f.write("MLP Network Parameter Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            # 模型信息
            f.write("Model Information:\n")
            f.write(f"  Model 1: {comparison['model1_info']['model_name']}\n")
            f.write(f"    Total parameters: {comparison['model1_info']['total_params']:,}\n")
            f.write(f"    Weight layers: {len(comparison['model1_info']['weight_layers'])}\n")
            f.write(f"    Architecture: {comparison['model1_info']['layer_dimensions']}\n\n")
            
            f.write(f"  Model 2: {comparison['model2_info']['model_name']}\n")
            f.write(f"    Total parameters: {comparison['model2_info']['total_params']:,}\n")
            f.write(f"    Weight layers: {len(comparison['model2_info']['weight_layers'])}\n")
            f.write(f"    Architecture: {comparison['model2_info']['layer_dimensions']}\n\n")
            
            # 全局对比
            f.write("Global Comparison:\n")
            global_comp = comparison['global_comparison']
            f.write(f"  Parameter count difference: {global_comp['total_params_diff']:,}\n\n")
            
            # 逐层对比
            f.write("Layer-by-Layer Comparison:\n")
            for i, layer_comp in enumerate(comparison['layer_comparison']):
                f.write(f"  Layer {i+1}: {layer_comp['layer1_name']} vs {layer_comp['layer2_name']}\n")
                f.write(f"    Shape match: {layer_comp['shape_match']}\n")
                
                if layer_comp['differences']:
                    for metric, diff_info in layer_comp['differences'].items():
                        f.write(f"    {metric}: {diff_info['relative_diff_percent']:.2f}% difference\n")
                f.write("\n")
        
        print(f"Comparison report saved to: {report_path}")


def main():
    """主函数"""
    print("MLP Network Parameter Comparator")
    print("=" * 50)
    
    # 默认路径
    mlp_bc_checkpoint = \
        "/home/lab/RL/lerobot/outputs/mlp_bc_grasp_training_aligned_2_vision_random/checkpoint_step_2000.pt"
    
    sac_checkpoint_dir = \
        "/home/lab/RL/lerobot/outputs/train/2025-08-22/18-56-12_20_warm_up_grasp_box_kuavo_reward_mse_demo01_action_06_yes_dataset_temp01_discount095_fps10_seed1000s/checkpoints/0034000"
    
    # 创建对比器
    comparator = NetworkParameterComparator()
    
    try:
        # 加载模型参数
        mlp_bc_params = comparator.load_mlp_bc_checkpoint(mlp_bc_checkpoint)
        sac_actor_params = comparator.load_sac_actor_parameters(sac_checkpoint_dir)
        
        # 执行对比
        comparison = comparator.compare_parameters(
            mlp_bc_params, sac_actor_params,
            "MLP_BC", "SAC_Actor"
        )
        
        # 可视化结果
        viz_path = "/home/lab/RL/lerobot/lerobot/scripts/rl/visual_mlp_network/param_compare/parameter_comparison.png"
        comparator.visualize_comparison(comparison, viz_path)
        
        # 保存报告
        report_path = "/home/lab/RL/lerobot/lerobot/scripts/rl/visual_mlp_network/param_compare/comparison_report.txt"
        comparator.save_comparison_report(comparison, report_path)
        
        print("\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
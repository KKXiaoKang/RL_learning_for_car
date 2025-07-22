#!/usr/bin/env python3
"""
SAC 熵分析和调优工具

这个工具帮助分析和解决SAC训练中熵下降慢的问题。
专门针对专家演示数据下熵下降缓慢的问题进行诊断和优化建议。

Usage:
    python entropy_analysis_tool.py --config_path your_config.json
    python entropy_analysis_tool.py --analyze_wandb_run your_run_id
"""

import json
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

class SACEntropyAnalyzer:
    """SAC熵分析器，用于诊断熵下降慢的问题"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def analyze_entropy_parameters(self) -> Dict[str, Any]:
        """分析当前配置的熵相关参数"""
        if not self.config:
            raise ValueError("请先加载配置文件")
        
        policy_config = self.config.get('policy', {})
        
        # 提取熵相关参数
        temperature_init = policy_config.get('temperature_init', 1.0)
        temperature_lr = policy_config.get('temperature_lr', 3e-4)
        target_entropy = policy_config.get('target_entropy', None)
        use_backup_entropy = policy_config.get('use_backup_entropy', True)
        
        # 动作维度
        action_dim = policy_config.get('output_features', {}).get('action', {}).get('shape', [1])[0]
        
        # 计算自动目标熵值
        auto_target_entropy = -action_dim / 2
        
        # 学习率相关
        actor_lr = policy_config.get('actor_lr', 3e-4)
        critic_lr = policy_config.get('critic_lr', 3e-4)
        
        # 网络参数
        policy_kwargs = policy_config.get('policy_kwargs', {})
        std_min = policy_kwargs.get('std_min', 1e-5)
        std_max = policy_kwargs.get('std_max', 10.0)
        
        analysis_result = {
            'current_config': {
                'temperature_init': temperature_init,
                'temperature_lr': temperature_lr,
                'target_entropy': target_entropy,
                'auto_target_entropy': auto_target_entropy,
                'use_backup_entropy': use_backup_entropy,
                'action_dim': action_dim,
                'actor_lr': actor_lr,
                'critic_lr': critic_lr,
                'std_min': std_min,
                'std_max': std_max,
            }
        }
        
        return analysis_result
    
    def diagnose_entropy_issues(self) -> Dict[str, Any]:
        """诊断熵下降慢的可能问题"""
        analysis = self.analyze_entropy_parameters()
        config = analysis['current_config']
        
        issues = []
        recommendations = []
        
        # 问题1: 初始温度过低
        if config['temperature_init'] < 0.1:
            issues.append("初始温度过低 ({:.4f})".format(config['temperature_init']))
            recommendations.append("建议将 temperature_init 设置为 0.2-1.0 之间")
        
        # 问题2: 温度学习率过低
        if config['temperature_lr'] < 1e-4:
            issues.append("温度学习率过低 ({:.0e})".format(config['temperature_lr']))
            recommendations.append("建议将 temperature_lr 提高到 3e-4 或更高")
        
        # 问题3: 目标熵值设置不当
        if config['target_entropy'] is not None:
            if config['target_entropy'] > -1:  # 目标熵过高
                issues.append("目标熵值过高 ({:.2f})".format(config['target_entropy']))
                recommendations.append("建议将 target_entropy 设置为 {:.2f} 左右".format(config['auto_target_entropy']))
        
        # 问题4: 学习率比例不当
        lr_ratio = config['temperature_lr'] / config['actor_lr']
        if lr_ratio < 0.5:
            issues.append("温度学习率相对于actor学习率过低 (比例: {:.2f})".format(lr_ratio))
            recommendations.append("建议保持温度学习率与actor学习率相近")
        
        # 问题5: 动作标准差范围问题
        if config['std_max'] < 1.0:
            issues.append("最大标准差限制过低 ({:.4f})".format(config['std_max']))
            recommendations.append("建议将 std_max 提高到 2.0 或更高")
        
        # 问题6: 高维动作空间
        if config['action_dim'] > 15:
            issues.append("动作维度较高 ({:d}维)".format(config['action_dim']))
            recommendations.append("高维动作空间需要更长训练时间，考虑调整温度参数")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'severity': 'HIGH' if len(issues) >= 3 else 'MEDIUM' if len(issues) >= 1 else 'LOW'
        }
    
    def generate_optimized_config(self) -> Dict[str, Any]:
        """生成优化后的配置建议"""
        analysis = self.analyze_entropy_parameters()
        config = analysis['current_config']
        
        # 基于动作维度调整参数
        action_dim = config['action_dim']
        
        # 优化建议
        optimized_params = {
            'temperature_init': max(0.2, min(1.0, action_dim * 0.05)),  # 基于动作维度调整
            'temperature_lr': max(3e-4, config['actor_lr']),  # 至少等于actor学习率
            'target_entropy': -action_dim * 0.5,  # 标准公式
            'use_backup_entropy': True,
            'policy_kwargs': {
                'std_min': 1e-4,
                'std_max': min(5.0, max(2.0, action_dim * 0.2)),  # 基于动作维度
                'init_final': 0.01
            }
        }
        
        return optimized_params
    
    def create_comparison_plot(self, original_params: Dict, optimized_params: Dict) -> str:
        """创建参数对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 温度初始值对比
        params = ['Original', 'Optimized']
        temp_values = [original_params['temperature_init'], optimized_params['temperature_init']]
        ax1.bar(params, temp_values, color=['red', 'green'])
        ax1.set_title('Temperature Init Comparison')
        ax1.set_ylabel('Temperature Init Value')
        
        # 学习率对比
        lr_original = [original_params['actor_lr'], original_params['temperature_lr']]
        lr_optimized = [original_params['actor_lr'], optimized_params['temperature_lr']]
        
        x = np.arange(len(['Actor LR', 'Temperature LR']))
        width = 0.35
        
        ax2.bar(x - width/2, lr_original, width, label='Original', color='red')
        ax2.bar(x + width/2, lr_optimized, width, label='Optimized', color='green')
        ax2.set_title('Learning Rate Comparison')
        ax2.set_ylabel('Learning Rate')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Actor LR', 'Temperature LR'])
        ax2.legend()
        ax2.set_yscale('log')
        
        # 目标熵值对比
        target_entropy_original = original_params.get('target_entropy')
        if target_entropy_original is None:
            target_entropy_original = -original_params['action_dim']/2
        target_entropy_optimized = optimized_params['target_entropy']
        
        ax3.bar(params, [target_entropy_original, target_entropy_optimized], color=['red', 'green'])
        ax3.set_title('Target Entropy Comparison')
        ax3.set_ylabel('Target Entropy Value')
        
        # 标准差范围对比
        std_ranges = ['Original Min', 'Original Max', 'Optimized Min', 'Optimized Max']
        std_values = [
            original_params['std_min'], 
            original_params['std_max'],
            optimized_params['policy_kwargs']['std_min'],
            optimized_params['policy_kwargs']['std_max']
        ]
        
        colors = ['red', 'red', 'green', 'green']
        ax4.bar(std_ranges, std_values, color=colors)
        ax4.set_title('Standard Deviation Range Comparison')
        ax4.set_ylabel('Std Value')
        ax4.set_yscale('log')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = "entropy_analysis_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def print_analysis_report(self):
        """打印详细的分析报告"""
        print("=" * 60)
        print("🔍 SAC熵下降慢问题分析报告")
        print("=" * 60)
        
        # 当前配置分析
        analysis = self.analyze_entropy_parameters()
        config = analysis['current_config']
        
        print(f"\n📊 当前配置参数:")
        print(f"  • 初始温度: {config['temperature_init']}")
        print(f"  • 温度学习率: {config['temperature_lr']:.0e}")
        print(f"  • 目标熵值: {config['target_entropy']} (自动计算: {config['auto_target_entropy']:.2f})")
        print(f"  • 动作维度: {config['action_dim']}维")
        print(f"  • Actor学习率: {config['actor_lr']:.0e}")
        print(f"  • 标准差范围: [{config['std_min']:.0e}, {config['std_max']:.1f}]")
        
        # 问题诊断
        diagnosis = self.diagnose_entropy_issues()
        print(f"\n⚠️  发现问题 (严重程度: {diagnosis['severity']}):")
        for i, issue in enumerate(diagnosis['issues'], 1):
            print(f"  {i}. {issue}")
        
        print(f"\n💡 优化建议:")
        for i, rec in enumerate(diagnosis['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # 优化配置
        optimized = self.generate_optimized_config()
        print(f"\n🚀 建议的优化配置:")
        print(f"  • temperature_init: {optimized['temperature_init']:.3f}")
        print(f"  • temperature_lr: {optimized['temperature_lr']:.0e}")
        print(f"  • target_entropy: {optimized['target_entropy']:.2f}")
        print(f"  • std_max: {optimized['policy_kwargs']['std_max']:.1f}")
        
        # 创建对比图
        plot_path = self.create_comparison_plot(config, optimized)
        print(f"\n📈 参数对比图已保存至: {plot_path}")
        
        # 专家演示相关建议
        print(f"\n🎯 专家演示数据的特别建议:")
        print(f"  1. 专家演示通常动作比较确定，熵值本身就较低")
        print(f"  2. 可以适当提高初始温度 (0.5-1.0) 来增加探索")
        print(f"  3. 使用更高的温度学习率 (5e-4 或更高)")
        print(f"  4. 考虑在训练初期使用更高的目标熵值")
        print(f"  5. 监控 log_prob 和 temperature 的变化趋势")
        
        print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description='SAC熵分析工具')
    parser.add_argument('--config_path', type=str, help='配置文件路径')
    parser.add_argument('--output_optimized_config', type=str, help='输出优化配置文件路径')
    
    args = parser.parse_args()
    
    if not args.config_path:
        print("请提供配置文件路径: --config_path your_config.json")
        return
    
    # 分析熵问题
    analyzer = SACEntropyAnalyzer(args.config_path)
    analyzer.print_analysis_report()
    
    # 输出优化配置
    if args.output_optimized_config:
        optimized_config = analyzer.generate_optimized_config()
        
        # 加载原始配置并更新
        original_config = analyzer.config.copy()
        original_config['policy'].update(optimized_config)
        
        with open(args.output_optimized_config, 'w') as f:
            json.dump(original_config, f, indent=2)
        
        print(f"\n✅ 优化配置已保存至: {args.output_optimized_config}")

if __name__ == "__main__":
    main() 
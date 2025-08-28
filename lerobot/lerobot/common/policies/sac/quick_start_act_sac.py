#!/usr/bin/env python3
"""
ACT-SAC Quick Start 快速开始

最简单的ACT-SAC使用示例，5分钟上手
"""

import torch
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.configs.types import PolicyFeature, FeatureType


def quick_start_example():
    """5分钟快速上手ACT-SAC"""
    
    print("🚀 ACT-SAC Quick Start")
    print("=" * 50)
    
    # 步骤1: 创建配置
    print("\n📝 步骤1: 创建配置")
    config = SACConfig(
        # 定义输入输出特征
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        },
        
        # 启用ACT Actor（关键步骤！）
        use_act_actor=True,
        
        # 轻量化配置用于快速测试
        act_dim_model=128,
        act_n_heads=4,
        act_n_encoder_layers=2,
        
        # BC混合训练
        bc_initial_weight=0.5,
        bc_final_weight=0.1,
        bc_decay_steps=1000,
        
        # 归一化配置
        dataset_stats={
            "observation.state": {"min": [0.0] * 10, "max": [1.0] * 10},
            "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
        },
        
        # 禁用视觉功能简化测试
        disable_vision_features=True,
    )
    print("  ✅ 配置创建完成")
    
    # 步骤2: 创建策略
    print("\n🤖 步骤2: 创建策略")
    policy = SACPolicy(config=config)
    print(f"  ✅ 策略创建完成，Actor类型: {type(policy.actor).__name__}")
    
    # 步骤3: 动作推理
    print("\n🎯 步骤3: 动作推理")
    batch_size = 2
    observations = {
        "observation.state": torch.randn(batch_size, 10)
    }
    
    with torch.no_grad():
        actions = policy.select_action(observations)
    
    print(f"  📊 输入观测形状: {observations['observation.state'].shape}")
    print(f"  📊 输出动作形状: {actions.shape}")
    print(f"  📊 动作值范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    print("  ✅ 动作推理成功")
    
    # 步骤4: 训练损失计算
    print("\n📈 步骤4: 训练损失计算")
    
    # 模拟训练数据
    expert_actions = torch.randn(batch_size, 4)
    rewards = torch.randn(batch_size, 1)
    done = torch.zeros(batch_size, 1, dtype=torch.bool)
    next_observations = {
        "observation.state": torch.randn(batch_size, 10)
    }
    
    # 构建训练批次
    batch = {
        "state": observations,
        "action": actions,
        "expert_action": expert_actions,  # 专家动作用于BC损失
        "reward": rewards,
        "next_state": next_observations,
        "done": done,
        "training_step": 500,  # 当前训练步数
    }
    
    # 计算Actor损失
    policy.train()  # 设置为训练模式
    loss_dict = policy.forward(batch, model="actor")
    
    print(f"  📊 Actor损失: {loss_dict['loss_actor'].item():.4f}")
    
    # 检查BC权重
    bc_weight = policy._compute_dynamic_bc_weight(500)
    print(f"  📊 当前BC权重: {bc_weight:.3f}")
    print("  ✅ 损失计算成功")
    
    # 步骤5: 对比传统SAC
    print("\n⚖️ 步骤5: 对比传统SAC")
    
    # 创建传统SAC配置
    traditional_config = SACConfig(
        input_features=config.input_features,
        output_features=config.output_features,
        use_act_actor=False,  # 使用传统MLP Actor
        dataset_stats=config.dataset_stats,
        disable_vision_features=True,
    )
    
    traditional_policy = SACPolicy(config=traditional_config)
    
    # 参数量对比
    act_params = sum(p.numel() for p in policy.actor.parameters())
    mlp_params = sum(p.numel() for p in traditional_policy.actor.parameters())
    
    print(f"  📊 ACT Actor参数量: {act_params:,}")
    print(f"  📊 MLP Actor参数量: {mlp_params:,}")
    print(f"  📊 参数增长倍数: {act_params / mlp_params:.1f}x")
    print("  ✅ 对比完成")
    
    print("\n🎉 Quick Start 完成！")
    print("=" * 50)
    print("📝 下一步:")
    print("  1. 查看完整文档: README_ACT_SAC.md")
    print("  2. 运行详细示例: how_to_use_act_sac.py")
    print("  3. 自定义配置: act_sac_config_example.py")
    print("  4. 运行完整测试: test_act_sac.py")


def compare_act_vs_mlp():
    """快速对比ACT和MLP Actor"""
    
    print("\n🔬 ACT vs MLP 快速对比")
    print("-" * 30)
    
    # 通用配置
    base_config = {
        "input_features": {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,))},
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        "dataset_stats": {
            "observation.state": {"min": [0.0] * 15, "max": [1.0] * 15},
            "action": {"min": [-1.0] * 6, "max": [1.0] * 6},
        },
        "disable_vision_features": True,
    }
    
    configs = {
        "MLP Actor": SACConfig(**base_config, use_act_actor=False),
        "ACT Actor": SACConfig(**base_config, use_act_actor=True, act_dim_model=256),
    }
    
    print(f"{'类型':<12} {'参数量':<10} {'推理时间':<10} {'内存占用':<10}")
    print("-" * 50)
    
    for name, config in configs.items():
        policy = SACPolicy(config=config)
        params = sum(p.numel() for p in policy.actor.parameters())
        
        # 简单的推理时间测试
        obs = {"observation.state": torch.randn(1, 15)}
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                policy.select_action(obs)
        
        # 计时
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                policy.select_action(obs)
        inference_time = (time.time() - start_time) / 100 * 1000  # ms
        
        # 内存使用（简单估计）
        memory_mb = params * 4 / 1024 / 1024  # 假设float32
        
        print(f"{name:<12} {params:<10,} {inference_time:<8.2f}ms {memory_mb:<8.1f}MB")


if __name__ == "__main__":
    try:
        quick_start_example()
        compare_act_vs_mlp()
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 解决建议:")
        print("  1. 确保已安装所有依赖")
        print("  2. 检查配置是否正确")
        print("  3. 查看详细文档获取帮助")

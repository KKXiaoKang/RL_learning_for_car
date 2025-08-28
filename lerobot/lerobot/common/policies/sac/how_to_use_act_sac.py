#!/usr/bin/env python3
"""
ACT-SAC 使用示例

展示如何在实际项目中使用ACT-SAC混合架构
"""

import torch
import numpy as np
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.configs.types import PolicyFeature, FeatureType


def example_basic_usage():
    """基础使用示例"""
    print("📖 基础ACT-SAC使用示例")
    
    # 1. 创建配置
    config = SACConfig(
        # 输入输出特征定义
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,)),  # 20维机器人状态
            "observation.image.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),  # 前置摄像头
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),  # 7维动作（6D pose + 1D gripper）
        },
        
        # 启用ACT Actor
        use_act_actor=True,
        
        # ACT架构配置
        act_dim_model=512,
        act_n_heads=8,
        act_n_encoder_layers=4,
        act_n_decoder_layers=1,
        
        # 视觉编码器
        vision_encoder_name="helper2424/resnet10",
        freeze_vision_encoder=True,
        
        # BC混合训练
        bc_initial_weight=0.6,
        bc_final_weight=0.05,
        bc_decay_steps=100000,
        
        # SAC参数
        num_critics=2,
        critic_lr=3e-4,
        actor_lr=1e-4,  # ACT Actor可能需要较小的学习率
        temperature_lr=3e-4,
        
        # 归一化配置（匹配我们的特征维度）
        dataset_stats={
            "observation.state": {
                "min": [0.0] * 20,  # 20维状态的最小值
                "max": [1.0] * 20,  # 20维状态的最大值
            },
            "observation.image.front": {
                "mean": [0.485, 0.456, 0.406],  # RGB通道均值
                "std": [0.229, 0.224, 0.225],   # RGB通道标准差
            },
            "action": {
                "min": [-1.0] * 7,  # 7维动作的最小值
                "max": [1.0] * 7,   # 7维动作的最大值
            },
        },
    )
    
    # 2. 创建策略
    policy = SACPolicy(config=config)
    print(f"  ✅ 创建策略成功，Actor类型: {type(policy.actor).__name__}")
    
    # 3. 模拟推理
    batch_size = 4
    observations = {
        "observation.state": torch.randn(batch_size, 20),
        "observation.image.front": torch.randn(batch_size, 3, 224, 224),
    }
    
    # 选择动作
    with torch.no_grad():
        actions = policy.select_action(observations)
    print(f"  ✅ 动作推理成功，形状: {actions.shape}")
    
    # 4. 模拟训练
    expert_actions = torch.randn(batch_size, 7)
    
    batch = {
        "state": observations,
        "action": actions,
        "expert_action": expert_actions,
        "training_step": 1000,
    }
    
    loss_dict = policy.forward(batch, model="actor")
    print(f"  ✅ 损失计算成功，Actor loss: {loss_dict['loss_actor'].item():.4f}")


def example_sequence_usage():
    """序列处理示例"""
    print("\n📚 序列ACT-SAC使用示例")
    
    config = SACConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(5,)),
        },
        
        # 启用序列ACT Actor
        use_act_actor=True,
        use_sequence_act_actor=True,
        obs_history_length=8,  # 使用8步历史
        
        # 适应序列处理的架构
        act_dim_model=256,
        act_n_heads=8,
        act_n_encoder_layers=6,  # 更多编码器层处理序列
        act_max_seq_length=10,
        
        # 其他配置
        disable_vision_features=True,  # 仅使用状态观测
        
        # 归一化配置
        dataset_stats={
            "observation.state": {
                "min": [0.0] * 15,  # 15维状态的最小值
                "max": [1.0] * 15,  # 15维状态的最大值
            },
            "action": {
                "min": [-1.0] * 5,  # 5维动作的最小值
                "max": [1.0] * 5,   # 5维动作的最大值
            },
        },
    )
    
    policy = SACPolicy(config=config)
    print(f"  ✅ 创建序列策略成功，Actor类型: {type(policy.actor).__name__}")
    
    # 测试单个观测
    observations = {"observation.state": torch.randn(2, 15)}
    actions = policy.select_action(observations)
    print(f"  ✅ 单观测推理成功，形状: {actions.shape}")


def example_training_loop():
    """训练循环示例"""
    print("\n🔄 训练循环示例")
    
    config = SACConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        use_act_actor=True,
        act_dim_model=128,  # 轻量化用于演示
        bc_decay_steps=5000,
        
        # 归一化配置
        dataset_stats={
            "observation.state": {
                "min": [0.0] * 10,
                "max": [1.0] * 10,
            },
            "action": {
                "min": [-1.0] * 4,
                "max": [1.0] * 4,
            },
        },
    )
    
    policy = SACPolicy(config=config)
    
    print("  开始模拟训练...")
    
    for step in range(0, 6000, 1000):
        # 模拟数据
        observations = {"observation.state": torch.randn(8, 10)}
        actions = torch.randn(8, 4)
        expert_actions = torch.randn(8, 4)
        
        # 前向传播
        batch = {
            "state": observations,
            "action": actions,
            "expert_action": expert_actions,
            "training_step": step,
        }
        
        loss_dict = policy.forward(batch, model="actor")
        
        # 获取BC权重
        bc_weight = policy._compute_dynamic_bc_weight(step)
        
        print(f"    步数 {step:4d}: Actor Loss = {loss_dict['loss_actor'].item():.4f}, "
              f"BC权重 = {bc_weight:.3f}")
    
    print("  ✅ 训练循环演示完成")


def example_config_comparison():
    """配置对比示例"""
    print("\n⚖️ 配置对比示例")
    
    base_features = {
        "input_features": {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(12,))},
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        "dataset_stats": {
            "observation.state": {
                "min": [0.0] * 12,
                "max": [1.0] * 12,
            },
            "action": {
                "min": [-1.0] * 6,
                "max": [1.0] * 6,
            },
        },
    }
    
    configs = {
        "传统MLP": SACConfig(**base_features, use_act_actor=False),
        "ACT基础": SACConfig(**base_features, use_act_actor=True, act_dim_model=256),
        "ACT序列": SACConfig(**base_features, use_act_actor=True, use_sequence_act_actor=True, act_dim_model=256),
        "ACT轻量": SACConfig(**base_features, use_act_actor=True, act_dim_model=128, act_n_encoder_layers=2),
    }
    
    for name, config in configs.items():
        try:
            policy = SACPolicy(config=config)
            
            # 计算参数数量
            total_params = sum(p.numel() for p in policy.actor.parameters())
            
            print(f"  {name:8s}: ✅ 参数量 = {total_params:,}")
            
        except Exception as e:
            print(f"  {name:8s}: ❌ 错误 = {e}")


def example_best_practices():
    """最佳实践示例"""
    print("\n💡 最佳实践示例")
    
    # 推荐配置
    config = SACConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(25,)),
            "observation.image.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        
        # ============ ACT配置 ============
        use_act_actor=True,
        
        # 架构参数（平衡性能和效率）
        act_dim_model=512,
        act_n_heads=8,
        act_n_encoder_layers=4,
        act_n_decoder_layers=1,
        act_dropout=0.1,
        
        # ============ 视觉配置 ============
        vision_encoder_name="helper2424/resnet10",
        freeze_vision_encoder=True,
        shared_encoder=True,
        
        # ============ BC配置 ============
        # 渐进式衰减策略
        bc_initial_weight=0.7,      # 初期重视模仿学习
        bc_final_weight=0.02,       # 后期主要依靠RL
        bc_decay_steps=80000,       # 在总训练的前80%步数内完成衰减
        
        # ============ SAC配置 ============
        num_critics=2,
        critic_lr=3e-4,
        actor_lr=1e-4,              # ACT Actor使用较小学习率
        temperature_lr=3e-4,
        discount=0.99,
        critic_target_update_weight=0.005,
        
        # ============ 优化配置 ============
        use_torch_compile=True,     # 启用编译加速
        grad_clip_norm=10.0,        # 梯度裁剪防止梯度爆炸
        
        # ============ 归一化配置 ============
        dataset_stats={
            "observation.state": {
                "min": [0.0] * 25,
                "max": [1.0] * 25,
            },
            "observation.image.front": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "action": {
                "min": [-1.0] * 8,
                "max": [1.0] * 8,
            },
        },
    )
    
    print("  📋 推荐配置要点:")
    print(f"    - ACT维度: {config.act_dim_model}")
    print(f"    - BC初始权重: {config.bc_initial_weight}")
    print(f"    - Actor学习率: {config.actor_lr}")
    print(f"    - 梯度裁剪: {config.grad_clip_norm}")
    print("  ✅ 这是一个平衡性能和稳定性的推荐配置")


if __name__ == "__main__":
    print("🚀 ACT-SAC 使用示例\n")
    
    try:
        example_basic_usage()
        example_sequence_usage()
        example_training_loop()
        example_config_comparison()
        example_best_practices()
        
        print("\n🎉 所有示例运行成功！")
        print("\n📝 总结:")
        print("  1. ACT-SAC成功融合了Transformer和强化学习")
        print("  2. 支持单步和序列两种模式")
        print("  3. BC混合训练提供了强大的初始化能力")
        print("  4. 配置灵活，可根据任务需求调整")
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()

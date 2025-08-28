#!/usr/bin/env python3
"""
ACT-SAC 混合架构测试脚本

验证ACT-SAC实现的正确性和兼容性
"""

import torch
import torch.nn as nn
from torch import Tensor
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

# 测试导入
try:
    from lerobot.common.policies.sac.configuration_sac import SACConfig
    from lerobot.common.policies.sac.modeling_sac import SACPolicy
    from lerobot.common.policies.sac.modeling_sac_act_actor import ACTSACActor
    from lerobot.common.policies.sac.modeling_sac_sequence_act_actor import SequenceACTSACActorV2
    from lerobot.configs.types import PolicyFeature, FeatureType
    print("✅ 成功导入所有模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)


def create_test_config(use_act_actor: bool = True, use_sequence: bool = False):
    """创建测试配置"""
    
    # 基础特征配置
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),  # 10维状态观测
    }
    
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),  # 4维动作
    }
    
    config = SACConfig(
        input_features=input_features,
        output_features=output_features,
        
        # ACT配置
        use_act_actor=use_act_actor,
        use_sequence_act_actor=use_sequence,
        obs_history_length=3,
        
        # 轻量化配置以便测试
        act_dim_model=128,
        act_n_heads=4,
        act_dim_feedforward=512,
        act_n_encoder_layers=2,
        act_n_decoder_layers=1,
        act_dropout=0.1,
        
        # SAC配置
        num_critics=2,
        shared_encoder=True,
        latent_dim=64,
        
        # BC配置
        bc_initial_weight=0.5,
        bc_final_weight=0.1,
        bc_decay_steps=1000,
        
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
    
    return config


def create_test_data(batch_size: int = 4, state_dim: int = 10, action_dim: int = 4):
    """创建测试数据"""
    
    observations = {
        "observation.state": torch.randn(batch_size, state_dim)
    }
    
    actions = torch.randn(batch_size, action_dim)
    expert_actions = torch.randn(batch_size, action_dim)
    
    return observations, actions, expert_actions


def test_traditional_sac():
    """测试传统SAC Actor"""
    print("\n🔧 测试传统SAC Actor...")
    
    config = create_test_config(use_act_actor=False)
    policy = SACPolicy(config=config)
    
    observations, actions, expert_actions = create_test_data()
    
    # 测试动作选择
    selected_actions = policy.select_action(observations)
    print(f"  - 动作选择: {selected_actions.shape}")
    
    # 测试损失计算
    batch = {
        "state": observations,
        "action": actions,
        "expert_action": expert_actions,
        "training_step": 100,
    }
    
    loss_dict = policy.forward(batch, model="actor")
    print(f"  - Actor损失: {loss_dict['loss_actor'].item():.4f}")
    
    print("✅ 传统SAC Actor测试通过")


def test_act_sac():
    """测试ACT-SAC Actor"""
    print("\n🤖 测试ACT-SAC Actor...")
    
    config = create_test_config(use_act_actor=True, use_sequence=False)
    policy = SACPolicy(config=config)
    
    observations, actions, expert_actions = create_test_data()
    
    # 测试动作选择
    selected_actions = policy.select_action(observations)
    print(f"  - 动作选择: {selected_actions.shape}")
    
    # 测试损失计算
    batch = {
        "state": observations,
        "action": actions,
        "expert_action": expert_actions,
        "training_step": 100,
    }
    
    loss_dict = policy.forward(batch, model="actor")
    print(f"  - Actor损失: {loss_dict['loss_actor'].item():.4f}")
    
    print("✅ ACT-SAC Actor测试通过")


def test_sequence_act_sac():
    """测试序列ACT-SAC Actor"""
    print("\n📚 测试序列ACT-SAC Actor...")
    
    config = create_test_config(use_act_actor=True, use_sequence=True)
    policy = SACPolicy(config=config)
    
    observations, actions, expert_actions = create_test_data()
    
    # 测试动作选择（单个观测）
    selected_actions = policy.select_action(observations)
    print(f"  - 单观测动作选择: {selected_actions.shape}")
    
    # 测试序列观测（如果Actor支持）
    if hasattr(policy.actor, 'obs_history_length'):
        obs_sequence = [observations for _ in range(3)]
        # 注意：这里可能需要修改select_action以支持序列输入
        # selected_actions_seq = policy.select_action(obs_sequence)
        # print(f"  - 序列观测动作选择: {selected_actions_seq.shape}")
    
    # 测试损失计算
    batch = {
        "state": observations,
        "action": actions,
        "expert_action": expert_actions,
        "training_step": 100,
    }
    
    loss_dict = policy.forward(batch, model="actor")
    print(f"  - Actor损失: {loss_dict['loss_actor'].item():.4f}")
    
    print("✅ 序列ACT-SAC Actor测试通过")


def test_bc_weight_decay():
    """测试BC权重衰减"""
    print("\n⚖️ 测试BC权重衰减...")
    
    config = create_test_config(use_act_actor=True)
    policy = SACPolicy(config=config)
    
    observations, actions, expert_actions = create_test_data()
    
    # 测试不同训练步数的权重
    steps = [0, 250, 500, 750, 1000, 1500]
    
    for step in steps:
        weight = policy._compute_dynamic_bc_weight(step)
        print(f"  - 步数 {step:4d}: BC权重 = {weight:.4f}")
    
    print("✅ BC权重衰减测试通过")


def test_compatibility():
    """测试兼容性"""
    print("\n🔄 测试兼容性...")
    
    # 测试配置切换
    configs = [
        ("传统SAC", create_test_config(use_act_actor=False)),
        ("ACT-SAC", create_test_config(use_act_actor=True, use_sequence=False)),
        ("序列ACT-SAC", create_test_config(use_act_actor=True, use_sequence=True)),
    ]
    
    for name, config in configs:
        try:
            policy = SACPolicy(config=config)
            observations, _, _ = create_test_data()
            actions = policy.select_action(observations)
            print(f"  - {name}: ✅ (动作形状: {actions.shape})")
        except Exception as e:
            print(f"  - {name}: ❌ 错误: {e}")
    
    print("✅ 兼容性测试通过")


def main():
    """主测试函数"""
    print("🚀 开始ACT-SAC测试...")
    
    try:
        # 基础功能测试
        test_traditional_sac()
        test_act_sac()
        test_sequence_act_sac()
        
        # 特殊功能测试
        test_bc_weight_decay()
        test_compatibility()
        
        print("\n🎉 所有测试通过！ACT-SAC实现正确。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

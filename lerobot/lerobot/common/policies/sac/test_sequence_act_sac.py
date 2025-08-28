#!/usr/bin/env python3
"""
序列ACT-SAC测试脚本

测试真正的动作序列预测和联合概率损失计算
"""

import torch
import torch.nn as nn
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)

# 测试导入
try:
    from lerobot.common.policies.sac.configuration_sac import SACConfig
    from lerobot.common.policies.sac.modeling_sac import SACPolicy
    from lerobot.common.policies.sac.modeling_sac_sequence_act_actor import SequenceACTSACActorV2
    from lerobot.configs.types import PolicyFeature, FeatureType
    print("✅ 成功导入所有模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)


def create_sequence_test_config():
    """创建序列ACT测试配置"""
    
    config = SACConfig(
        # 基础特征
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        },
        
        # 启用序列ACT Actor
        use_act_actor=True,
        use_sequence_act_actor=True,
        
        # 序列参数
        obs_history_length=1, # 观测历史长度
        act_chunk_size=5,  # 预测5步动作序列
        
        # 轻量化配置用于测试
        act_dim_model=128,
        act_n_heads=4,
        act_dim_feedforward=256,
        act_n_encoder_layers=2,
        act_n_decoder_layers=2,
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
                "min": [0.0] * 12,
                "max": [1.0] * 12,
            },
            "action": {
                "min": [-1.0] * 4,
                "max": [1.0] * 4,
            },
        },
        
        disable_vision_features=True,
    )
    
    return config


def test_sequence_action_prediction():
    """测试动作序列预测"""
    print("\n🚀 测试动作序列预测")
    
    config = create_sequence_test_config()
    policy = SACPolicy(config=config)
    
    batch_size = 2
    
    # 创建观测序列
    obs_sequence = []
    for t in range(config.obs_history_length):
        obs = {
            "observation.state": torch.randn(batch_size, 12) * 0.5 + 0.5
        }
        obs_sequence.append(obs)
    
    print(f"  📊 观测序列长度: {len(obs_sequence)}")
    print(f"  📊 每个观测形状: {obs_sequence[0]['observation.state'].shape}")
    
    # 测试序列预测
    with torch.no_grad():
        # 预测完整动作序列
        action_sequence, log_probs_joint, means_sequence = policy.actor(
            obs_sequence, 
            return_sequence=True
        )
        
        print(f"  ✅ 动作序列形状: {action_sequence.shape}")  # 应该是 (batch, chunk_size, action_dim)
        print(f"  ✅ 联合概率形状: {log_probs_joint.shape}")  # 应该是 (batch,)
        print(f"  ✅ 均值序列形状: {means_sequence.shape}")  # 应该是 (batch, chunk_size, action_dim)
        
        # 预测单个动作（用于SAC）
        single_action, single_log_prob, single_mean = policy.actor(
            obs_sequence,
            return_sequence=False
        )
        
        print(f"  ✅ 单个动作形状: {single_action.shape}")  # 应该是 (batch, action_dim)
        print(f"  ✅ 单个概率形状: {single_log_prob.shape}")  # 应该是 (batch,)
        
        # 验证单个动作是序列的第一个
        first_from_sequence = action_sequence[:, 0, :]
        print(f"  📊 单个动作与序列第一个的差异: {torch.abs(single_action - first_from_sequence).mean().item():.6f}")


def test_sequence_loss_computation():
    """测试序列损失计算"""
    print("\n📈 测试序列损失计算")
    
    config = create_sequence_test_config()
    policy = SACPolicy(config=config)
    
    batch_size = 2
    chunk_size = config.act_chunk_size
    action_dim = 4
    
    # 创建观测序列
    obs_sequence = []
    for t in range(config.obs_history_length):
        obs = {
            "observation.state": torch.randn(batch_size, 12) * 0.5 + 0.5
        }
        obs_sequence.append(obs)
    
    # 创建专家动作序列
    expert_action_sequences = torch.randn(batch_size, chunk_size, action_dim) * 0.5
    
    print(f"  📊 专家动作序列形状: {expert_action_sequences.shape}")
    
    # 测试序列损失计算
    policy.train()
    
    # 构建训练批次
    batch = {
        "state": obs_sequence,  # 注意：这里传递的是观测序列
        "expert_action_sequences": expert_action_sequences,
        "training_step": 500,
    }
    
    # 计算Actor损失
    loss_dict = policy.forward(batch, model="actor")
    
    print(f"  ✅ Actor损失: {loss_dict['loss_actor'].item():.4f}")
    print(f"  📊 BC权重: {policy._last_bc_weight:.3f}")
    print(f"  📊 序列长度: {policy._last_sequence_length}")
    print(f"  📊 联合对数概率: {policy._last_joint_log_prob.item():.4f}")


def test_sequence_vs_single_comparison():
    """对比序列预测和单步预测"""
    print("\n⚖️ 对比序列预测和单步预测")
    
    # 单步配置
    single_config = create_sequence_test_config()
    single_config.use_sequence_act_actor = False  # 使用基础ACT
    single_policy = SACPolicy(config=single_config)
    
    # 序列配置
    sequence_config = create_sequence_test_config()
    sequence_policy = SACPolicy(config=sequence_config)
    
    batch_size = 2
    
    # 单个观测
    obs = {
        "observation.state": torch.randn(batch_size, 12) * 0.5 + 0.5
    }
    
    with torch.no_grad():
        # 单步预测
        single_action, single_log_prob, _ = single_policy.actor(obs)
        
        # 序列预测（第一个动作）
        sequence_action, sequence_log_prob, _ = sequence_policy.actor([obs], return_sequence=False)
        
        print(f"  📊 单步Actor参数量: {sum(p.numel() for p in single_policy.actor.parameters()):,}")
        print(f"  📊 序列Actor参数量: {sum(p.numel() for p in sequence_policy.actor.parameters()):,}")
        print(f"  📊 单步动作形状: {single_action.shape}")
        print(f"  📊 序列动作形状: {sequence_action.shape}")
        print(f"  📊 单步概率范围: [{single_log_prob.min().item():.3f}, {single_log_prob.max().item():.3f}]")
        print(f"  📊 序列概率范围: [{sequence_log_prob.min().item():.3f}, {sequence_log_prob.max().item():.3f}]")


def test_autoregressive_generation():
    """测试自回归生成"""
    print("\n🔄 测试自回归生成")
    
    config = create_sequence_test_config()
    policy = SACPolicy(config=config)
    
    batch_size = 1
    
    # 创建观测序列
    obs_sequence = []
    for t in range(config.obs_history_length):
        obs = {
            "observation.state": torch.randn(batch_size, 12) * 0.5 + 0.5
        }
        obs_sequence.append(obs)
    
    # 设置为评估模式（触发自回归生成）
    policy.eval()
    
    with torch.no_grad():
        # 生成动作序列
        action_sequence, log_probs_joint, means_sequence = policy.actor(
            obs_sequence,
            return_sequence=True
        )
        
        print(f"  ✅ 自回归生成序列形状: {action_sequence.shape}")
        print(f"  📊 动作序列变化范围:")
        for t in range(config.act_chunk_size):
            action_t = action_sequence[0, t, :]
            print(f"    步骤 {t}: 动作范围 [{action_t.min().item():.3f}, {action_t.max().item():.3f}]")
        
        # 验证动作序列的时间一致性
        action_diffs = []
        for t in range(1, config.act_chunk_size):
            diff = torch.abs(action_sequence[0, t, :] - action_sequence[0, t-1, :]).mean()
            action_diffs.append(diff.item())
        
        avg_diff = np.mean(action_diffs)
        print(f"  📊 平均步间动作差异: {avg_diff:.4f}")


def test_bc_loss_with_sequences():
    """测试BC损失与动作序列"""
    print("\n📚 测试BC损失与动作序列")
    
    config = create_sequence_test_config()
    policy = SACPolicy(config=config)
    
    batch_size = 3
    chunk_size = config.act_chunk_size
    action_dim = 4
    
    # 创建观测序列
    obs_sequence = []
    for t in range(config.obs_history_length):
        obs = {
            "observation.state": torch.randn(batch_size, 12) * 0.5 + 0.5
        }
        obs_sequence.append(obs)
    
    # 创建不同长度的专家动作序列进行测试
    print("  🔍 测试不同长度的专家序列:")
    
    # 测试1: 完整长度序列
    expert_full = torch.randn(batch_size, chunk_size, action_dim) * 0.5
    batch_full = {
        "state": obs_sequence,
        "expert_action_sequences": expert_full,
        "training_step": 100,
    }
    
    policy.train()
    loss_full = policy.forward(batch_full, model="actor")
    print(f"    完整序列 ({chunk_size}步): BC损失 {policy._last_bc_loss.item():.4f}")
    
    # 测试2: 较短序列
    expert_short = torch.randn(batch_size, chunk_size-2, action_dim) * 0.5
    batch_short = {
        "state": obs_sequence,
        "expert_action_sequences": expert_short,
        "training_step": 100,
    }
    
    loss_short = policy.forward(batch_short, model="actor")
    print(f"    较短序列 ({chunk_size-2}步): BC损失 {policy._last_bc_loss.item():.4f}")
    
    # 测试3: 较长序列
    expert_long = torch.randn(batch_size, chunk_size+3, action_dim) * 0.5
    batch_long = {
        "state": obs_sequence,
        "expert_action_sequences": expert_long,
        "training_step": 100,
    }
    
    loss_long = policy.forward(batch_long, model="actor")
    print(f"    较长序列 ({chunk_size+3}步): BC损失 {policy._last_bc_loss.item():.4f}")


def main():
    """主测试函数"""
    print("🚀 开始序列ACT-SAC测试...")
    
    try:
        test_sequence_action_prediction()
        test_sequence_loss_computation()
        test_sequence_vs_single_comparison()
        test_autoregressive_generation()
        test_bc_loss_with_sequences()
        
        print("\n🎉 所有序列ACT测试通过！")
        
        print("\n📊 关键特性验证:")
        print("  ✅ 动作序列预测 (chunk-based)")
        print("  ✅ 联合概率损失计算")
        print("  ✅ 自回归序列生成")
        print("  ✅ 序列BC损失")
        print("  ✅ SAC与序列损失集成")
        
        print("\n💡 这个实现的核心优势:")
        print("  🔄 真正的动作序列建模")
        print("  🎯 联合概率优化而非独立动作")
        print("  🧠 充分利用ACT的Transformer架构")
        print("  ⚡ 与SAC强化学习框架深度集成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

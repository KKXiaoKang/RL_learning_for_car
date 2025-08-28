#!/usr/bin/env python3
"""
ACT-SAC 混合架构配置示例

这个文件展示了如何配置SAC以使用ACT Transformer Actor
"""

from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType


def create_act_sac_config():
    """创建ACT-SAC混合配置"""
    
    config = SACConfig(
        # ========== 基础SAC配置 ==========
        discount=0.99,
        temperature_init=1.0,
        num_critics=2,
        critic_lr=3e-4,
        actor_lr=3e-4,
        temperature_lr=3e-4,
        critic_target_update_weight=0.005,
        
        # ========== ACT Actor配置 ==========
        # 启用ACT Transformer Actor
        use_act_actor=True,
        # 是否使用序列版本（处理观测历史）
        use_sequence_act_actor=False,  # 可以设为True以启用序列处理
        obs_history_length=5,  # 序列长度
        
        # ACT Transformer架构参数
        act_dim_model=512,          # Transformer隐藏维度
        act_n_heads=8,              # 注意力头数
        act_dim_feedforward=3200,   # 前馈网络维度
        act_n_encoder_layers=4,     # 编码器层数
        act_n_decoder_layers=1,     # 解码器层数
        act_dropout=0.1,            # Dropout率
        act_feedforward_activation="relu",  # 激活函数
        act_pre_norm=False,         # 是否使用预归一化
        act_max_seq_length=10,      # 最大序列长度
        
        # ========== BC混合训练配置 ==========
        bc_initial_weight=0.5,      # BC损失初始权重
        bc_final_weight=0.01,       # BC损失最终权重
        bc_decay_steps=50000,       # BC权重衰减步数
        
        # ========== 网络架构配置 ==========
        # 观测编码器
        vision_encoder_name="helper2424/resnet10",  # 或者None使用默认编码器
        freeze_vision_encoder=True,
        shared_encoder=True,
        latent_dim=256,
        
        # ========== 训练配置 ==========
        online_steps=1000000,
        online_buffer_capacity=100000,
        offline_buffer_capacity=100000,
        policy_update_freq=1,
        utd_ratio=1,
        
        # ========== 优化配置 ==========
        use_torch_compile=True,
        grad_clip_norm=40.0,
    )
    
    return config


def create_sequence_act_sac_config():
    """创建支持序列处理的ACT-SAC配置"""
    
    config = create_act_sac_config()
    
    # 启用序列处理
    config.use_sequence_act_actor = True
    config.obs_history_length = 10  # 更长的观测历史
    
    # 调整ACT参数以适应序列处理
    config.act_n_encoder_layers = 6  # 更多编码器层
    config.act_max_seq_length = 20   # 更长的最大序列长度
    
    return config


def create_lightweight_act_sac_config():
    """创建轻量化的ACT-SAC配置（适合快速实验）"""
    
    config = create_act_sac_config()
    
    # 减少模型复杂度
    config.act_dim_model = 256
    config.act_n_heads = 4
    config.act_dim_feedforward = 1024
    config.act_n_encoder_layers = 2
    config.act_n_decoder_layers = 1
    
    return config


# 使用示例
if __name__ == "__main__":
    
    # 创建基础ACT-SAC配置
    config = create_act_sac_config()
    print("🤖 ACT-SAC基础配置:")
    print(f"  - ACT Actor: {config.use_act_actor}")
    print(f"  - Sequence Actor: {config.use_sequence_act_actor}")
    print(f"  - Transformer dim: {config.act_dim_model}")
    print(f"  - Encoder layers: {config.act_n_encoder_layers}")
    print(f"  - BC初始权重: {config.bc_initial_weight}")
    
    # 创建序列版本配置
    seq_config = create_sequence_act_sac_config()
    print("\n📚 序列ACT-SAC配置:")
    print(f"  - 观测历史长度: {seq_config.obs_history_length}")
    print(f"  - 最大序列长度: {seq_config.act_max_seq_length}")
    
    # 创建轻量化配置
    lite_config = create_lightweight_act_sac_config()
    print("\n⚡ 轻量化ACT-SAC配置:")
    print(f"  - Transformer dim: {lite_config.act_dim_model}")
    print(f"  - 前馈维度: {lite_config.act_dim_feedforward}")

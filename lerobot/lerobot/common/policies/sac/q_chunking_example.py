#!/usr/bin/env python3
"""
Q-chunking SAC使用示例

该脚本展示如何配置和使用Q-chunking SAC进行训练
"""

import torch
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy


def create_q_chunking_config():
    """创建Q-chunking SAC配置"""
    
    # 基础配置
    config = SACConfig(
        # 启用序列ACT Actor
        use_act_actor=True,
        use_sequence_act_actor=True,
        
        # Q-chunking配置
        enable_q_chunking=True,
        q_chunking_strategy="standard",  # 可选: "standard", "conservative", "temporal_weighted"
        q_chunking_horizon=3,
        q_chunking_decay=0.9,
        q_chunking_entropy_scaling="linear",  # 可选: "linear", "sqrt", "log", "none"
        
        # ACT参数
        act_chunk_size=8,
        obs_history_length=5,
        act_dim_model=512,
        act_n_heads=8,
        act_n_encoder_layers=4,
        act_n_decoder_layers=4,
        
        # SAC参数
        discount=0.99,
        temperature_init=1.0,
        critic_lr=3e-4,
        actor_lr=3e-4,
        temperature_lr=3e-4,
        
        # 输入输出特征（示例）
        input_features={
            "observation.state": torch.randn(1, 10),  # 状态维度
            "observation.image.front": torch.randn(1, 3, 224, 224),  # 图像
        },
        output_features={
            "action": torch.randn(1, 4),  # 动作维度
        }
    )
    
    return config


def demonstrate_q_chunking_differences():
    """演示Q-chunking与传统SAC的区别"""
    
    print("🔥 Q-chunking SAC vs 传统SAC 对比演示")
    print("=" * 60)
    
    # 1. 传统SAC配置
    traditional_config = SACConfig(
        use_act_actor=False,
        use_sequence_act_actor=False,
        enable_q_chunking=False,
        input_features={"observation.state": torch.randn(1, 10)},
        output_features={"action": torch.randn(1, 4)}
    )
    
    # 2. Q-chunking SAC配置
    q_chunking_config = create_q_chunking_config()
    
    print("📊 配置对比:")
    print(f"传统SAC - 动作预测: 单步")
    print(f"Q-chunking SAC - 动作预测: {q_chunking_config.act_chunk_size}步序列")
    print(f"Q-chunking SAC - 策略: {q_chunking_config.q_chunking_strategy}")
    print(f"Q-chunking SAC - 时间视野: {q_chunking_config.q_chunking_horizon}")
    print()
    
    # 3. 策略对比
    print("🎯 策略优化对比:")
    print("传统SAC: loss = E[α * log π(a_t|s_t) - Q(s_t, a_t)]")
    print("Q-chunking: loss = E[α * log π(a_1:t|s_1:t) - Q(s_1, a_1)]")
    print("         其中 a_1:t 是动作序列，π(a_1:t|s_1:t) 是联合概率")
    print()
    
    # 4. TD学习对比
    print("📈 TD学习对比:")
    print("传统SAC: 1-step TD target = r + γ * Q_target(s', a')")
    print("Q-chunking: n-step TD target = Σ(γ^i * r_{t+i}) + γ^n * Q_target(s_{t+n}, a_{t+n})")
    print()


def demonstrate_q_chunking_strategies():
    """演示不同Q-chunking策略"""
    
    print("🚀 Q-chunking策略对比")
    print("=" * 40)
    
    strategies = {
        "standard": {
            "描述": "使用第一个动作计算Q值，序列联合概率",
            "优点": "计算高效，论文标准方法",
            "适用": "一般任务",
            "公式": "Q(s, a_1), log π(a_1:k|s)"
        },
        "conservative": {
            "描述": "对多个动作计算Q值并取最小值",
            "优点": "更保守稳定的估计",
            "适用": "高风险任务",
            "公式": "min(Q(s, a_1), Q(s, a_2), ...), log π(a_1:k|s)"
        },
        "temporal_weighted": {
            "描述": "对不同时间步动作给予不同权重",
            "优点": "平衡近期和远期动作重要性",
            "适用": "需要时间平衡的任务",
            "公式": "Σ(w_t * Q(s, a_t)), log π(a_1:k|s)"
        }
    }
    
    for strategy, info in strategies.items():
        print(f"📋 {strategy.upper()}策略:")
        print(f"   描述: {info['描述']}")
        print(f"   优点: {info['优点']}")
        print(f"   适用: {info['适用']}")
        print(f"   公式: {info['公式']}")
        print()


def create_example_training_config():
    """创建示例训练配置文件"""
    
    config = {
        # 基础设置
        "env_name": "YourEnv-v0",
        "device": "cuda",
        "seed": 42,
        
        # SAC Q-chunking配置
        "use_act_actor": True,
        "use_sequence_act_actor": True,
        "enable_q_chunking": True,
        
        # Q-chunking参数
        "q_chunking_strategy": "standard",
        "q_chunking_horizon": 3,
        "q_chunking_decay": 0.9,
        "q_chunking_entropy_scaling": "linear",
        
        # ACT参数
        "act_chunk_size": 8,
        "obs_history_length": 5,
        "act_dim_model": 512,
        "act_n_heads": 8,
        "act_n_encoder_layers": 4,
        "act_n_decoder_layers": 4,
        
        # 训练参数
        "online_steps": 1000000,
        "critic_lr": 3e-4,
        "actor_lr": 3e-4,
        "temperature_lr": 3e-4,
        "batch_size": 256,
        "utd_ratio": 1,
        
        # 评估参数
        "eval_freq": 10000,
        "save_freq": 50000,
    }
    
    print("📝 示例训练配置:")
    print("=" * 30)
    for key, value in config.items():
        print(f"{key}: {value}")
    print()
    
    return config


def main():
    """主函数：运行所有演示"""
    
    print("🎯 Q-chunking SAC 实现演示")
    print("基于论文: https://arxiv.org/abs/2507.07969")
    print("=" * 80)
    print()
    
    # 1. 演示Q-chunking与传统SAC的区别
    demonstrate_q_chunking_differences()
    
    # 2. 演示不同Q-chunking策略
    demonstrate_q_chunking_strategies()
    
    # 3. 创建示例配置
    create_example_training_config()
    
    # 4. 创建Q-chunking配置示例
    try:
        config = create_q_chunking_config()
        print("✅ Q-chunking配置创建成功!")
        print(f"   序列长度: {config.act_chunk_size}")
        print(f"   Q-chunking策略: {config.q_chunking_strategy}")
        print(f"   时间视野: {config.q_chunking_horizon}")
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
    
    print()
    print("🚀 开始训练:")
    print("python lerobot/scripts/rl/learner.py --config-path='config/q_chunking_sac.json'")
    print()
    print("📊 监控关键指标:")
    print("- q_chunking/joint_log_prob: 动作序列联合概率")
    print("- q_chunking/chunk_size: 序列长度")
    print("- q_chunking/n_step_returns: n-step return值")
    print("- q_chunking/strategy: 使用的策略")


if __name__ == "__main__":
    main()

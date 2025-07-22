#!/usr/bin/env python3
"""
Reward Analysis Script for RLKuavo Environment

This script analyzes the reward design in RLKuavo environment to identify
why random exploration gets similar rewards to human demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_reward_components():
    """分析奖励组件，模拟不同距离下的奖励值"""
    
    # 模拟距离范围 (0.1m 到 3.0m)
    distances = np.linspace(0.1, 3.0, 100)
    
    # 计算各种奖励组件
    torso_rewards = np.exp(-2.0 * distances) * 8.0
    eef_rewards = np.exp(-2.0 * distances) * 3.0  # 单手奖励，双手翻倍
    
    # 对称性奖励（假设对称性误差从0到1）
    symmetry_errors = np.linspace(0, 1, 100)
    symmetry_rewards = np.exp(-5.0 * symmetry_errors) * 1.0
    
    # 箱子提升奖励（假设提升高度从0到0.3m）
    lift_heights = np.linspace(0, 0.3, 100)
    lift_rewards = lift_heights * 20.0
    
    # 成功奖励
    success_reward = 100.0
    
    print("=== RLKuavo奖励设计分析 ===")
    print(f"1. 躯干接近奖励范围: {torso_rewards.min():.2f} - {torso_rewards.max():.2f}")
    print(f"2. 单手末端接近奖励范围: {eef_rewards.min():.2f} - {eef_rewards.max():.2f}")
    print(f"3. 双手末端接近奖励范围: {(eef_rewards*2).min():.2f} - {(eef_rewards*2).max():.2f}")
    print(f"4. 对称性奖励范围: {symmetry_rewards.min():.2f} - {symmetry_rewards.max():.2f}")
    print(f"5. 箱子提升奖励范围: {lift_rewards.min():.2f} - {lift_rewards.max():.2f}")
    print(f"6. 成功奖励: {success_reward:.2f}")
    
    # 计算随机情况下可能的奖励
    print("\n=== 随机情况下可能的奖励分析 ===")
    
    # 假设随机情况：机器人距离箱子1-2米
    random_distance = 1.5
    random_torso_reward = np.exp(-2.0 * random_distance) * 8.0
    random_eef_reward = np.exp(-2.0 * random_distance) * 3.0 * 2  # 双手
    
    print(f"随机情况下(距离{random_distance}m)：")
    print(f"  - 躯干接近奖励: {random_torso_reward:.2f}")
    print(f"  - 双手末端接近奖励: {random_eef_reward:.2f}")
    print(f"  - 总计密集奖励: {random_torso_reward + random_eef_reward:.2f}")
    
    # 人工演示情况：机器人贴近箱子
    demo_distance = 0.3
    demo_torso_reward = np.exp(-2.0 * demo_distance) * 8.0
    demo_eef_reward = np.exp(-2.0 * demo_distance) * 3.0 * 2
    demo_symmetry_reward = np.exp(-5.0 * 0.1) * 1.0  # 假设较好对称性
    demo_lift_reward = 0.15 * 20.0  # 假设提升15cm
    
    print(f"\n人工演示情况下(距离{demo_distance}m)：")
    print(f"  - 躯干接近奖励: {demo_torso_reward:.2f}")
    print(f"  - 双手末端接近奖励: {demo_eef_reward:.2f}")
    print(f"  - 对称性奖励: {demo_symmetry_reward:.2f}")
    print(f"  - 提升奖励: {demo_lift_reward:.2f}")
    print(f"  - 成功奖励: {success_reward:.2f}")
    print(f"  - 总计奖励: {demo_torso_reward + demo_eef_reward + demo_symmetry_reward + demo_lift_reward + success_reward:.2f}")
    
    print("\n=== 问题诊断 ===")
    print("问题1: 密集奖励过高 - 即使随机动作也能获得可观的接近奖励")
    print("问题2: 奖励尺度不平衡 - 密集奖励与稀疏奖励(成功)比例不当")
    print("问题3: 缺乏惩罚机制 - 没有对无效动作的明确惩罚")
    
    return {
        'distances': distances,
        'torso_rewards': torso_rewards,
        'eef_rewards': eef_rewards,
        'symmetry_rewards': symmetry_rewards,
        'lift_rewards': lift_rewards
    }

def suggest_improvements():
    """建议改进方案"""
    print("\n=== 建议的改进方案 ===")
    print("1. 降低密集奖励权重：")
    print("   - 躯干接近奖励: 8.0 → 2.0")
    print("   - 末端接近奖励: 3.0 → 1.0") 
    print("   - 对称性奖励: 1.0 → 0.5")
    
    print("\n2. 增加稀疏奖励权重：")
    print("   - 成功奖励: 100.0 → 200.0")
    print("   - 增加阶段性里程碑奖励")
    
    print("\n3. 添加有效惩罚：")
    print("   - 时间惩罚: -0.1 per step")
    print("   - 无效动作惩罚: -1.0 for large actions")
    print("   - 远离惩罚: 当距离增加时给予负奖励")
    
    print("\n4. 改进奖励塑形：")
    print("   - 使用更陡峭的衰减函数")
    print("   - 引入任务阶段性奖励")
    print("   - 添加动作平滑性奖励")

if __name__ == "__main__":
    rewards_data = analyze_reward_components()
    suggest_improvements()
    
    # 可视化奖励曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards_data['distances'], rewards_data['torso_rewards'])
    plt.title('躯干接近奖励 vs 距离')
    plt.xlabel('距离 (m)')
    plt.ylabel('奖励')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(rewards_data['distances'], rewards_data['eef_rewards'] * 2)
    plt.title('双手末端接近奖励 vs 距离')
    plt.xlabel('距离 (m)')
    plt.ylabel('奖励')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(np.linspace(0, 1, 100), rewards_data['symmetry_rewards'])
    plt.title('对称性奖励 vs 误差')
    plt.xlabel('对称性误差')
    plt.ylabel('奖励')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(np.linspace(0, 0.3, 100), rewards_data['lift_rewards'])
    plt.title('提升奖励 vs 高度')
    plt.xlabel('提升高度 (m)')
    plt.ylabel('奖励')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reward_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n奖励分析图表已保存为 reward_analysis.png") 
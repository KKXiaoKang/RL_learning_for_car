from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch

def compute_image_stats(repo_id: str):
    """计算图像数据集的mean和std统计信息"""
    
    # 加载数据集
    dataset = LeRobotDataset(repo_id=repo_id)
    
    # 获取图像统计信息
    stats = dataset.meta.stats
    
    # 打印observation.images.front的统计信息
    if "observation.images.front" in stats:
        image_stats = stats["observation.images.front"]
        print("observation.images.front statistics:")
        print(f"Mean: {image_stats['mean'].tolist()}")
        print(f"Std: {image_stats['std'].tolist()}")
        print(f"Min: {image_stats['min'].tolist()}")
        print(f"Max: {image_stats['max'].tolist()}")
        
        return {
            "mean": image_stats['mean'].tolist(),
            "std": image_stats['std'].tolist()
        }
    else:
        print("observation.images.front not found in dataset stats")
        return None

def compute_observation_state_stats(repo_id: str):
    """计算observation.state的min和max统计信息"""
    
    # 加载数据集
    dataset = LeRobotDataset(repo_id=repo_id)
    
    # 获取observation.state统计信息
    stats = dataset.meta.stats
    
    if "observation.state" in stats:
        state_stats = stats["observation.state"]
        print("observation.state statistics:")
        print(f"Min: {state_stats['min'].tolist()}")
        print(f"Max: {state_stats['max'].tolist()}")
        print(f"Mean: {state_stats['mean'].tolist()}")
        print(f"Std: {state_stats['std'].tolist()}")
        
        return {
            "min": state_stats['min'].tolist(),
            "max": state_stats['max'].tolist(),
            "mean": state_stats['mean'].tolist(),
            "std": state_stats['std'].tolist()
        }
    else:
        print("observation.state not found in dataset stats")
        return None

def compute_action_stats(repo_id: str):
    """计算action的min和max统计信息"""
    
    # 加载数据集
    dataset = LeRobotDataset(repo_id=repo_id)
    
    # 获取action统计信息
    stats = dataset.meta.stats
    
    if "action" in stats:
        action_stats = stats["action"]
        print("action statistics:")
        print(f"Min: {action_stats['min'].tolist()}")
        print(f"Max: {action_stats['max'].tolist()}")
        print(f"Mean: {action_stats['mean'].tolist()}")
        print(f"Std: {action_stats['std'].tolist()}")
        
        return {
            "min": action_stats['min'].tolist(),
            "max": action_stats['max'].tolist(),
            "mean": action_stats['mean'].tolist(),
            "std": action_stats['std'].tolist()
        }
    else:
        print("action not found in dataset stats")
        return None

def manually_compute_state_action_stats(repo_id: str):
    """手动从数据集中计算observation.state和action的统计信息"""
    
    # 加载数据集
    dataset = LeRobotDataset(repo_id=repo_id)
    
    print("Manually computing statistics from dataset...")
    
    # 收集所有数据
    all_states = []
    all_actions = []
    
    for i in range(len(dataset.hf_dataset)):
        item = dataset.hf_dataset[i]
        
        if "observation.state" in item:
            all_states.append(item["observation.state"].numpy())
        
        if "action" in item:
            all_actions.append(item["action"].numpy())
    
    results = {}
    
    # 计算observation.state统计信息
    if all_states:
        all_states = np.array(all_states)
        state_stats = {
            "min": np.min(all_states, axis=0).tolist(),
            "max": np.max(all_states, axis=0).tolist(),
            "mean": np.mean(all_states, axis=0).tolist(),
            "std": np.std(all_states, axis=0).tolist()
        }
        results["observation.state"] = state_stats
        
        print("observation.state manual statistics:")
        print(f"Min: {state_stats['min']}")
        print(f"Max: {state_stats['max']}")
        print(f"Shape: {all_states.shape}")
    
    # 计算action统计信息
    if all_actions:
        all_actions = np.array(all_actions)
        action_stats = {
            "min": np.min(all_actions, axis=0).tolist(),
            "max": np.max(all_actions, axis=0).tolist(),
            "mean": np.mean(all_actions, axis=0).tolist(),
            "std": np.std(all_actions, axis=0).tolist()
        }
        results["action"] = action_stats
        
        print("action manual statistics:")
        print(f"Min: {action_stats['min']}")
        print(f"Max: {action_stats['max']}")
        print(f"Shape: {all_actions.shape}")
    
    return results

def compute_all_stats(repo_id: str, use_manual=False):
    """计算所有统计信息的综合函数"""
    
    print(f"Computing statistics for dataset: {repo_id}")
    print("=" * 60)
    
    all_stats = {}
    
    if use_manual:
        print("Using manual computation method...")
        manual_stats = manually_compute_state_action_stats(repo_id)
        all_stats.update(manual_stats)
        print()
        
        # 对于图像，仍然使用已有的统计信息
        image_stats = compute_image_stats(repo_id)
        if image_stats:
            all_stats["observation.images.front"] = image_stats
    else:
        print("Using pre-computed statistics from dataset metadata...")
        
        # 计算图像统计信息
        image_stats = compute_image_stats(repo_id)
        if image_stats:
            all_stats["observation.images.front"] = image_stats
        
        print()
        
        # 计算observation.state统计信息
        state_stats = compute_observation_state_stats(repo_id)
        if state_stats:
            all_stats["observation.state"] = state_stats
        
        print()
        
        # 计算action统计信息
        action_stats = compute_action_stats(repo_id)
        if action_stats:
            all_stats["action"] = action_stats
    
    print("=" * 60)
    print("Summary of all statistics:")
    for key, stats in all_stats.items():
        print(f"{key}: {list(stats.keys())}")
    
    return all_stats

def format_stats_for_config(stats_dict):
    """将统计信息格式化为配置文件格式"""
    
    formatted_stats = {}
    
    for key, stats in stats_dict.items():
        if key == "observation.images.front":
            # 图像统计信息只需要mean和std
            formatted_stats[key] = {
                "mean": stats["mean"],
                "std": stats["std"]
            }
        else:
            # 状态和动作需要min和max
            formatted_stats[key] = {
                "min": stats["min"],
                "max": stats["max"]
            }
    
    print("Formatted statistics for configuration:")
    print("-" * 40)
    for key, stats in formatted_stats.items():
        print(f'"{key}": {{')
        for stat_name, values in stats.items():
            print(f'    "{stat_name}": {values},')
        print('},')
    
    return formatted_stats

if __name__ == "__main__":
    # 使用示例
    repo_id = "KANGKKANG/rl_kuavo_724_1615"
    
    # 方法1：使用预计算的统计信息
    print("方法1：使用预计算的统计信息")
    stats1 = compute_all_stats(repo_id, use_manual=False)
    
    print("\n" + "=" * 80 + "\n")
    
    # 方法2：手动计算统计信息（更准确但耗时）
    print("方法2：手动计算统计信息")
    stats2 = compute_all_stats(repo_id, use_manual=True)
    
    print("\n" + "=" * 80 + "\n")
    
    # 格式化为配置文件格式
    if stats1:
        print("配置文件格式的统计信息：")
        formatted = format_stats_for_config(stats1)
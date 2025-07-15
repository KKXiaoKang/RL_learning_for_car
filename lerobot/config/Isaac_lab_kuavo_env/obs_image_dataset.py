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

# 使用示例
repo_id = "KANGKKANG/rl_kuavo_714_1830"
stats = compute_image_stats(repo_id)
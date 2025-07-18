from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "KANGKKANG/rl_kuavo_718_1730",  # repo_id
    root="/home/lab/.cache/huggingface/lerobot/KANGKKANG/rl_kuavo_718_1730"
)
dataset.push_to_hub()
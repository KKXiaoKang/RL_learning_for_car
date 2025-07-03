# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Learner server runner for distributed HILSerl robot policy training.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.

Examples of usage:

- Start a learner server for training:
```bash
python lerobot/scripts/rl/learner.py --config_path lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: Start the learner server before launching the actor server. The learner opens a gRPC server
to communicate with actors.

**NOTE**: Training progress can be monitored through Weights & Biases if wandb.enable is set to true
in your configuration.

**WORKFLOW**:
1. Create training configuration with proper policy, dataset, and environment settings
2. Start this learner server with the configuration
3. Start an actor server with the same configuration
4. Monitor training progress through wandb dashboard

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.common.cameras import opencv  # noqa: F401
from lerobot.common.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.robots import so100_follower  # noqa: F401
from lerobot.common.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.common.transport import services_pb2_grpc
from lerobot.common.transport.utils import (
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.common.utils.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.common.utils.process import ProcessSignalHandler
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.train_utils import (
    load_training_state as utils_load_training_state,
)
from lerobot.common.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.scripts.rl import learner_service

LOG_PREFIX = "[LEARNER]"


#################################################
# MAIN ENTRY POINTS AND CORE ALGORITHM FUNCTIONS #
#################################################


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.common.utils.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed) # 设置随机种子

    """
        benchmark 会让 PyTorch 在第一次运行卷积操作时，自动测试所有可用的算法，选择最快的那个，然后缓存下来，测试结果会被缓存，后续使用相同输入尺寸时直接使用最优算法
        allow_tf32 允许在CUDA上使用TF32混合精度计算, 提高性能
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    """
        ┌─────────────────┐    gRPC通信    ┌─────────────────┐
        │   Actor Server  │ ◄────────────► │  Learner Server │
        │                 │                │                 │
        │  - 环境交互     │                │  - 策略训练     │
        │  - 数据收集     │                │  - 参数更新     │
        │  - 动作执行     │                │  - 模型保存     │
        └─────────────────┘                └─────────────────┘
                │                                   │
                │                                   │
                ▼                                   ▼
        ┌─────────────────┐                ┌─────────────────┐
        │  transition_queue│ ◄────────────► │ parameters_queue│
        │  (经验数据)      │                │  (策略参数)     │
        └─────────────────┘                └─────────────────┘
    """
    # Create multiprocessing queues 
    transition_queue = Queue() # # 用于接收来自Actor的转换数据（经验）的队列
    interaction_message_queue = Queue() # 用于接收来自Actor的交互消息的队列
    parameters_queue = Queue() # 用于向Actor发送策略参数的队列

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread # 使用多线程
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process # 多进程编程

    communication_process = concurrency_entity(
        target=start_learner,  # 目标函数
        args=(
            parameters_queue, # 策略参数队列
            transition_queue, # 转换数据队列
            interaction_message_queue, # 交互消息队列
            shutdown_event, # 关闭事件
            cfg, # 配置对象
        ),
        daemon=True, # 守护进程，主进程结束时自动结束
    )
    communication_process.start() # 启动通信进程

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


#################################################
# Core algorithm functions #
#################################################


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,  # 训练配置对象，包含所有超参数和设置
    wandb_logger: WandBLogger | None,  # Weights & Biases日志记录器，用于跟踪训练进度
    shutdown_event: any,  # Event,  # 用于信号关闭的事件对象
    transition_queue: Queue,  # 用于接收来自Actor的转换数据（经验）的队列
    interaction_message_queue: Queue,  # 用于接收来自Actor的交互消息的队列
    parameters_queue: Queue,  # 用于向Actor发送策略参数的队列
):
    """
    处理从Actor到Learner的数据传输，管理训练更新，并在在线强化学习设置中记录训练进度。

    此函数持续执行以下操作：
    - 将转换数据从Actor传输到回放缓冲区
    - 记录接收到的交互消息
    - 确保只有在回放缓冲区有足够数量的转换数据时才开始训练
    - 从回放缓冲区采样批次并执行多次评论家更新
    - 定期更新Actor、评论家和温度优化器
    - 记录训练统计信息，包括损失值和优化频率

    注意：此函数没有单一职责，将来应该拆分为多个函数。
    我们这样做的原因是Python的GIL（全局解释器锁）。性能会降低200倍，所以我们需要一个执行所有工作的单线程。

    参数：
        cfg (TrainRLServerPipelineConfig): 包含超参数的配置对象
        wandb_logger (WandBLogger | None): 用于跟踪训练进度的日志记录器
        shutdown_event (Event): 用于信号关闭的事件
        transition_queue (Queue): 用于接收来自Actor的转换数据的队列
        interaction_message_queue (Queue): 用于接收来自Actor的交互消息的队列
        parameters_queue (Queue): 用于向Actor发送策略参数的队列
    """
    # 在开始时提取所有配置变量，这可以提高7%的速度性能
    # 为PyTorch操作设置主要的计算设备（例如 'cuda' 或 'cpu'）
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True) 
    # 为回放缓冲区（Replay Buffer）设置存储设备，通常是 'cpu'，以节省宝贵的GPU显存
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    # 梯度裁剪的最大范数值，用于防止梯度爆炸，稳定训练
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    # 在正式开始学习前，需要从环境中收集的初始经验（transition）数量
    online_step_before_learning = cfg.policy.online_step_before_learning
    # 更新与数据比率（Update-to-Data Ratio），表示每从环境中获取一个新样本，就进行多少次梯度更新
    utd_ratio = cfg.policy.utd_ratio
    # 环境的帧率（Frames Per Second），主要用于保存数据集时的元数据
    fps = cfg.env.fps
    # 记录训练指标（如损失函数值）的频率，以优化步骤（optimization step）为单位
    log_freq = cfg.log_freq
    # 保存模型检查点（checkpoint）的频率，以优化步骤为单位
    save_freq = cfg.save_freq
    # 策略网络（Actor）和温度参数（Temperature）的更新频率，相对于评论家网络（Critic）的更新
    policy_update_freq = cfg.policy.policy_update_freq
    # 将更新后的策略参数推送到Actor的频率，以秒为单位
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    # 是否启用保存检查点功能的布尔标志
    saving_checkpoint = cfg.save_checkpoint
    # 在线训练的总步数
    online_steps = cfg.policy.online_steps
    # 是否为数据加载器启用异步预取功能，以提高数据加载效率
    async_prefetch = cfg.policy.async_prefetch

    # 为多进程初始化日志记录
    if not use_threads(cfg):  # 如果不是使用线程模式（即使用多进程模式）
        # 创建日志目录
        log_dir = os.path.join(cfg.output_dir, "logs")
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        # 为当前进程创建特定的日志文件
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        # 初始化日志记录，显示进程ID
        init_logging(log_file=log_file, display_pid=True)
        # 记录日志初始化完成的信息
        logging.info("Initialized logging for actor information and training process")

    # 记录开始初始化策略的信息
    logging.info("Initializing policy")

    # 创建策略网络（Policy），这是强化学习中的核心组件，负责根据环境状态选择动作
    policy: SACPolicy = make_policy(
        cfg=cfg.policy,  # 策略配置
        env_cfg=cfg.env,  # 环境配置
    )

    # 确保策略是一个神经网络模块
    assert isinstance(policy, nn.Module)

    # 将策略网络设置为训练模式，启用梯度计算
    policy.train()

    # 将策略网络的参数推送到Actor，以便Actor可以开始使用这个策略
    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    # 记录上次推送策略的时间，用于控制推送频率
    last_time_policy_pushed = time.time()

    # 创建优化器（例如 Adam）和学习率调度器（本实现中为 None）
    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # 如果需要恢复训练，则从上次的优化步骤和交互步骤开始
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    # 记录策略网络的初始信息，包括网络结构和参数数量
    log_training_info(cfg=cfg, policy=policy)

    # 初始化回放缓冲区（Replay Buffer），用于存储从环境中收集的经验数据 - 在线Buffer
    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    # 设置批量大小（Batch Size），用于控制每次训练的样本数量
    batch_size = cfg.batch_size
    # 初始化离线回放缓冲区（Offline Replay Buffer），用于存储从数据集加载的经验数据
    offline_replay_buffer = None

    # 如果配置中指定了数据集，则初始化离线回放缓冲区
    if cfg.dataset is not None:  # 如果配置中包含数据集信息
        # 初始化离线回放缓冲区
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,  # 配置对象
            device=device,  # 计算设备
            storage_device=storage_device,  # 存储设备
        )
        # 如果同时使用在线和离线数据，则将批量大小减半，因为我们将从两个缓冲区采样
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    # 记录开始学习者线程的信息
    logging.info("Starting learner thread")
    # 初始化人工干预信息变量
    interaction_message = None # 人工干预信息
    # 设置优化步骤计数器，如果恢复训练则从上次的步骤开始，否则从0开始
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    # 设置交互步骤偏移量，用于处理恢复训练时的步骤计数
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    # 初始化数据集仓库ID变量
    dataset_repo_id = None
    # 如果配置中包含数据集信息，则获取数据集仓库ID
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # 初始化迭代器变量
    online_iterator = None  # 在线数据迭代器
    offline_iterator = None  # 离线数据迭代器

    # 注意：这是学习者的主循环
    while True:
        # 如果请求关闭，则退出训练循环
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # 处理所有可用的转换数据到回放缓冲区，这些数据由Actor服务器发送
        process_transitions(
            transition_queue=transition_queue,  # 转换数据队列
            replay_buffer=replay_buffer,  # 在线回放缓冲区
            offline_replay_buffer=offline_replay_buffer,  # 离线回放缓冲区
            device=device,  # 计算设备
            dataset_repo_id=dataset_repo_id,  # 数据集仓库ID
            shutdown_event=shutdown_event,  # 关闭事件
        )

        # 处理所有可用的交互消息，这些消息由Actor服务器发送
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,  # 交互消息队列
            interaction_step_shift=interaction_step_shift,  # 交互步骤偏移量
            wandb_logger=wandb_logger,  # 日志记录器
            shutdown_event=shutdown_event,  # 关闭事件
        )

        # 等待直到回放缓冲区有足够的样本开始训练
        if len(replay_buffer) < online_step_before_learning:
            continue

        # 如果在线迭代器还没有初始化，则创建它
        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size,  # 批量大小
                async_prefetch=async_prefetch,  # 是否异步预取
                queue_size=2  # 队列大小
            )

        # 如果有离线回放缓冲区且离线迭代器还没有初始化，则创建它
        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size,  # 批量大小
                async_prefetch=async_prefetch,  # 是否异步预取
                queue_size=2  # 队列大小
            )

        # 记录一次优化步骤开始的时间
        time_for_one_optimization_step = time.time()
        # 执行UTD比率减1次数的优化步骤（除了最后一次）
        for _ in range(utd_ratio - 1):
            """
                在这里UTD过程当中只更新 critic 的梯度 包含如下critic
                * critic_ensemble 机械臂动作空间
                * discrete_critic 离散动作空间
            """
            # 从迭代器中采样数据
            batch = next(online_iterator) # 根据batch_size大小遍历online_iterator

            # 如果有离线数据集，则同时采样离线数据
            if dataset_repo_id is not None:
                batch_offline = next(offline_iterator) # 根据batch_size大小遍历offline_iterator
                # 将在线和离线批次数据连接起来 torch.cat
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch,  # 在线批次
                    right_batch_transition=batch_offline  # 离线批次
                )

            # 从批次中提取各个组件 - 基础四元组 (s, a, s', r) - 后面添加一个基础任务完成位 done
            actions = batch["action"]  # 动作
            rewards = batch["reward"]  # 奖励
            observations = batch["state"]  # 当前状态
            next_observations = batch["next_state"]  # 下一状态
            done = batch["done"]  # 完成标志
            # 检查转换数据中是否有NaN值
            check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            # 获取观测特征和下一观测特征
            observation_features, next_observation_features = get_observation_features(
                policy=policy,  # 策略网络
                observations=observations,  # 当前观测
                next_observations=next_observations  # 下一观测
            )

            # 创建包含前向传播所需所有元素的批次字典
            forward_batch = {
                "action": actions,  # 动作
                "reward": rewards,  # 奖励
                "state": observations,  # 当前状态
                "next_state": next_observations,  # 下一状态
                "done": done,  # 完成标志
                "observation_feature": observation_features,  # 观测特征
                "next_observation_feature": next_observation_features,  # 下一观测特征
                "complementary_info": batch["complementary_info"],  # 补充信息
            }

            # 使用前向传播方法计算评论家损失
            critic_output = policy.forward(forward_batch, model="critic")

            # 主要的评论家优化
            loss_critic = critic_output["loss_critic"]  # 获取评论家损失
            optimizers["critic"].zero_grad()  # 清零评论家优化器的梯度
            loss_critic.backward()  # 反向传播
            # 对评论家网络参数进行梯度裁剪
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(),  # 评论家集成网络的参数
                max_norm=clip_grad_norm_value  # 最大梯度范数
            )
            optimizers["critic"].step()  # 更新评论家网络参数

            # 离散评论家优化（如果可用）
            if policy.config.num_discrete_actions is not None:  # 如果有离散动作
                # 计算离散评论家输出
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]  # 获取离散评论家损失
                optimizers["discrete_critic"].zero_grad()  # 清零离散评论家优化器的梯度
                loss_discrete_critic.backward()  # 反向传播
                # 对离散评论家网络参数进行梯度裁剪
                discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.discrete_critic.parameters(),  # 离散评论家网络参数
                    max_norm=clip_grad_norm_value  # 最大梯度范数
                )
                optimizers["discrete_critic"].step()  # 更新离散评论家网络参数

            # 更新所有Q网络（包括当前Q网络集合、目标Q网络集合、离散空间Q网络）
            policy.update_target_networks()

        # 为UTD比率中的最后一次更新采样数据
        batch = next(online_iterator)

        # 如果有离线数据集，则同时采样离线数据
        if dataset_repo_id is not None:
            batch_offline = next(offline_iterator)
            # 将在线和离线批次数据连接起来
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch,  # 在线批次
                right_batch_transition=batch_offline  # 离线批次
            )

        # 从批次中提取各个组件
        actions = batch["action"]  # 动作
        rewards = batch["reward"]  # 奖励
        observations = batch["state"]  # 当前状态
        next_observations = batch["next_state"]  # 下一状态
        done = batch["done"]  # 完成标志

        # 检查转换数据中是否有NaN值
        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        # 获取观测特征和下一观测特征
        observation_features, next_observation_features = get_observation_features(
            policy=policy,  # 策略网络
            observations=observations,  # 当前观测
            next_observations=next_observations  # 下一观测
        )

        # 创建包含前向传播所需所有元素的批次字典
        forward_batch = {
            "action": actions,  # 动作
            "reward": rewards,  # 奖励
            "state": observations,  # 当前状态
            "next_state": next_observations,  # 下一状态
            "done": done,  # 完成标志
            "observation_feature": observation_features,  # 观测特征
            "next_observation_feature": next_observation_features,  # 下一观测特征
        }

        # 计算评论家输出
        critic_output = policy.forward(forward_batch, model="critic")

        # 获取评论家损失并进行优化
        loss_critic = critic_output["loss_critic"]  # 获取评论家损失
        optimizers["critic"].zero_grad()  # 清零评论家优化器的梯度
        loss_critic.backward()  # 反向传播
        # 对评论家网络参数进行梯度裁剪并获取梯度范数
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(),  # 评论家集成网络的参数
            max_norm=clip_grad_norm_value  # 最大梯度范数
        ).item()  # 转换为标量值
        optimizers["critic"].step()  # 更新评论家网络参数

        # 初始化训练信息字典
        training_infos = {
            "loss_critic": loss_critic.item(),  # 评论家损失
            "critic_grad_norm": critic_grad_norm,  # 评论家梯度范数
        }

        # 离散评论家优化（如果可用）
        if policy.config.num_discrete_actions is not None:  # 如果有离散动作
            # 计算离散评论家输出
            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]  # 获取离散评论家损失
            optimizers["discrete_critic"].zero_grad()  # 清零离散评论家优化器的梯度
            loss_discrete_critic.backward()  # 反向传播
            # 对离散评论家网络参数进行梯度裁剪并获取梯度范数
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.discrete_critic.parameters(),  # 离散评论家网络参数
                max_norm=clip_grad_norm_value  # 最大梯度范数
            ).item()  # 转换为标量值
            optimizers["discrete_critic"].step()  # 更新离散评论家网络参数

            # 将离散评论家信息添加到训练信息中
            training_infos["loss_discrete_critic"] = loss_discrete_critic.item()  # 离散评论家损失
            training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm  # 离散评论家梯度范数

        """
            在这个过程当中会把actor 策略网络 和 temperature 温度参数 的梯度进行更新
        """
        # Actor和温度优化（按指定频率进行）
        # 当前优化的步数（如果不是断点续训的话都是从0开始） % (policy_update_freq=1) == 0  | 代表每一步都要更新actor和温度
        if optimization_step % policy_update_freq == 0:  # 如果到了更新Actor和温度的时候
            for _ in range(policy_update_freq):  # 执行多次更新
                # Actor优化
                actor_output = policy.forward(forward_batch, model="actor")  # 计算Actor输出
                loss_actor = actor_output["loss_actor"]  # 获取Actor损失
                optimizers["actor"].zero_grad()  # 清零Actor优化器的梯度
                loss_actor.backward()  # 反向传播
                # 对Actor网络参数进行梯度裁剪并获取梯度范数
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor.parameters(),  # Actor网络参数
                    max_norm=clip_grad_norm_value  # 最大梯度范数
                ).item()  # 转换为标量值
                optimizers["actor"].step()  # 更新Actor网络参数

                # 将Actor信息添加到训练信息中
                training_infos["loss_actor"] = loss_actor.item()  # Actor损失
                training_infos["actor_grad_norm"] = actor_grad_norm  # Actor梯度范数

                # 温度优化
                temperature_output = policy.forward(forward_batch, model="temperature")  # 计算温度输出
                loss_temperature = temperature_output["loss_temperature"]  # 获取温度损失
                optimizers["temperature"].zero_grad()  # 清零温度优化器的梯度
                loss_temperature.backward()  # 反向传播
                # 对温度参数进行梯度裁剪并获取梯度范数
                temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=[policy.log_alpha],  # 温度参数（log_alpha）
                    max_norm=clip_grad_norm_value  # 最大梯度范数
                ).item()  # 转换为标量值
                optimizers["temperature"].step()  # 更新温度参数

                # 将温度信息添加到训练信息中
                training_infos["loss_temperature"] = loss_temperature.item()  # 温度损失
                training_infos["temperature_grad_norm"] = temp_grad_norm  # 温度梯度范数
                training_infos["temperature"] = policy.temperature  # 当前温度值

                # 更新温度
                policy.update_temperature()

        # 如果需要，将策略推送给Actor
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:  # 如果距离上次推送的时间超过了指定频率
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)  # 推送策略参数到队列
            last_time_policy_pushed = time.time()  # 更新上次推送时间

        # 更新所有Q网络（包括当前Q网络集合、目标Q网络集合、离散空间Q网络）
        policy.update_target_networks()

        # 按指定间隔记录训练指标
        # 当前步数 % (log_freq=1) == 0  | 代表每log_freq步记录一次日志
        if optimization_step % log_freq == 0:  # 如果到了记录日志的时候
            training_infos["replay_buffer_size"] = len(replay_buffer)  # 在线回放缓冲区大小
            if offline_replay_buffer is not None:  # 如果有离线回放缓冲区
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)  # 离线回放缓冲区大小
            training_infos["Optimization step"] = optimization_step  # 优化步骤

            # 记录训练指标
            if wandb_logger:  # 如果有WandB日志记录器
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")  # 记录训练信息

        # 计算并记录优化频率
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step  # 计算一次优化步骤所需的时间
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)  # 计算优化频率（Hz）

        # 记录优化频率到控制台
        logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

        # 记录优化频率到WandB
        if wandb_logger:  # 如果有WandB日志记录器
            wandb_logger.log_dict(
                {
                    "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,  # 优化频率
                    "Optimization step": optimization_step,  # 优化步骤
                },
                mode="train",  # 训练模式
                custom_step_key="Optimization step",  # 自定义步骤键
            )

        # 增加优化步骤计数器
        optimization_step += 1
        # 按指定间隔记录优化步骤数量
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # 按指定间隔保存检查点
        if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == online_steps):  # 如果需要保存检查点且到了保存时间或达到总步数
            save_training_checkpoint(
                cfg=cfg,  # 配置对象
                optimization_step=optimization_step,  # 当前优化步骤
                online_steps=online_steps,  # 总在线步数
                interaction_message=interaction_message,  # 交互消息
                policy=policy,  # 策略网络
                optimizers=optimizers,  # 优化器
                replay_buffer=replay_buffer,  # 在线回放缓冲区
                offline_replay_buffer=offline_replay_buffer,  # 离线回放缓冲区
                dataset_repo_id=dataset_repo_id,  # 数据集仓库ID
                fps=fps,  # 帧率
            )


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    Start the learner server for training.
    It will receive transitions and interaction messages from the actor server,
    and send policy parameters to the actor server.

    Args:
        parameters_queue: Queue for sending policy parameters to the actor
        transition_queue: Queue for receiving transitions from the actor
        interaction_message_queue: Queue for receiving interaction messages from the actor
        shutdown_event: Event to signal shutdown
        cfg: Training configuration
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        # TODO: Check if its useful
        _ = ProcessSignalHandler(False, display_pid=True)

    # 在 start_learner 函数中创建gRPC服务器
    service = learner_service.LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=learner_service.MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", learner_service.MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", learner_service.MAX_MESSAGE_SIZE),
        ],
    )

    # 注册服务
    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    # 启动服务器
    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(learner_service.SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
) -> None:
    """
    Save training checkpoint and associated data.

    This function performs the following steps:
    1. Creates a checkpoint directory with the current optimization step
    2. Saves the policy model, configuration, and optimizer states
    3. Saves the current interaction step for resuming training
    4. Updates the "last" checkpoint symlink to point to this checkpoint
    5. Saves the replay buffer as a dataset for later use
    6. If an offline replay buffer exists, saves it as a separate dataset

    Args:
        cfg: Training configuration
        optimization_step: Current optimization step
        online_steps: Total number of online steps
        interaction_message: Dictionary containing interaction information
        policy: Policy model to save
        optimizers: Dictionary of optimizers
        replay_buffer: Replay buffer to save as dataset
        offline_replay_buffer: Optional offline replay buffer to save
        dataset_repo_id: Repository ID for dataset
        fps: Frames per second for dataset
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    _num_digits = max(6, len(str(online_steps)))
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # Create checkpoint directory
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    # Save checkpoint
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # Save interaction step manually
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step, "interaction_step": interaction_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # Update the "last" symlink
    update_last_checkpoint(checkpoint_dir)

    # TODO : temporary save replay buffer here, remove later when on the robot
    # We want to control this with the keyboard inputs
    dataset_dir = os.path.join(cfg.output_dir, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Save dataset
    # NOTE: Handle the case where the dataset repo id is not specified in the config
    # eg. RL training without demonstrations data
    repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
    replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=dataset_dir)

    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        offline_replay_buffer.to_lerobot_dataset(
            cfg.dataset.repo_id,
            fps=fps,
            root=dataset_offline_dir,
        )

    logging.info("Resume training")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    NOTE:
    - If the encoder is shared, its parameters are excluded from the actor's optimization process.
    - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

    Args:
        cfg: Configuration object containing hyperparameters.
        policy (nn.Module): The policy model containing the actor, critic, and temperature components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names ("actor", "critic", "temperature") to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    if cfg.policy.num_discrete_actions is not None:
        optimizer_discrete_critic = torch.optim.Adam(
            params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
        )
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
    lr_scheduler = None
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    if cfg.policy.num_discrete_actions is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic
    return optimizers, lr_scheduler


#################################################
# Training setup functions #
#################################################


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    Handle the resume logic for training.

    If resume is True:
    - Verifies that a checkpoint exists
    - Loads the checkpoint configuration
    - Logs resumption details
    - Returns the checkpoint configuration

    If resume is False:
    - Checks if an output directory exists (to prevent accidental overwriting)
    - Returns the original configuration

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration

    Returns:
        TrainRLServerPipelineConfig: The updated configuration

    Raises:
        RuntimeError: If resume is True but no checkpoint found, or if resume is False but directory exists
    """
    out_dir = cfg.output_dir

    # Case 1: Not resuming, but need to check if directory exists to prevent overwrites
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # Case 2: Resuming training
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # Log that we found a valid checkpoint and are resuming
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # Load config using Draccus
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # Ensure resume flag is set in returned config
    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    Loads the training state (optimizers, step count, etc.) from a checkpoint.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        optimizers (Optimizer | dict): Optimizers to load state into

    Returns:
        tuple: (optimization_step, interaction_step) or (None, None) if not resuming
    """
    if not cfg.resume:
        return None, None

    # Construct path to the last checkpoint directory
    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # Use the utility function from train_utils which loads the optimizer state
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        # Load interaction step separately from training_state.pt
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)  # nosec B614: Safe usage of torch.load
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    Log information about the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        policy (nn.Module): Policy model
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized replay buffer
    """
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    logging.info("Resume training load the online dataset")
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    # NOTE: In RL is possible to not have a dataset.
    repo_id = None
    if cfg.dataset is not None:
        repo_id = cfg.dataset.repo_id
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True,
    )


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    Initialize an offline replay buffer from a dataset.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized offline replay buffer
    """
    # 如果cfg.resume为False，则创建离线数据集，不是恢复训练
    if not cfg.resume:
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg) # 创建离线数据集
    else:
        # 如果cfg.resume为True，则加载离线数据集，为恢复训练
        logging.info("load offline dataset")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )

    logging.info("Convert to a offline replay buffer")
    # 将离线数据集转换为离线回放缓冲区
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    return offline_replay_buffer


#################################################
# Utilities/Helpers functions #
#################################################


def get_observation_features(
    policy: SACPolicy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Get observation features from the policy encoder. It act as cache for the observation features.
    when the encoder is frozen, the observation features are not updated.
    We can save compute by caching the observation features.

    Args:
        policy: The policy model
        observations: The current observations
        next_observations: The next observations

    Returns:
        tuple: observation_features, next_observation_features
    """
    # 只有当策略配置了视觉编码器（vision_encoder_name 不为 None）
    # 且视觉编码器被冻结（freeze_vision_encoder = True）时
    # 才会启用特征缓存机制
    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad(): # 禁用梯度计算，因为编码器被冻结
        # 为当前观测提取特征
        observation_features = policy.actor.encoder.get_cached_image_features(observations, normalize=True)
        
        # 为下一个观测提取特征
        next_observation_features = policy.actor.encoder.get_cached_image_features(
            next_observations, normalize=True
        )

    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN values in transition data.

    Args:
        observations: Dictionary of observation tensors
        actions: Action tensor
        next_state: Dictionary of next state tensors
        raise_error: If True, raises ValueError when NaN is detected

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    nan_detected = False

    # Check observations
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # Check next state
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # Check actions
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")
    state_dict = move_state_dict_to_device(policy.actor.state_dict(), device="cpu")
    state_bytes = state_to_bytes(state_dict)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """Process a single interaction message with consistent handling."""
    message = bytes_to_python_object(message)
    # Shift interaction step for consistency with checkpointed state
    message["Interaction step"] += interaction_step_shift

    # Log if logger available
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
):
    """Process all available transitions from the queue.

    Args:
        transition_queue: Queue for receiving transitions from the actor
        replay_buffer: Replay buffer to add transitions to
        offline_replay_buffer: Offline replay buffer to add transitions to
        device: Device to move transitions to
        dataset_repo_id: Repository ID for dataset
        shutdown_event: Event to signal shutdown
    """
    # 当前policy transition队列不为空，并且shutdown事件没有被设置，则继续处理transition队列中的数据
    while not transition_queue.empty() and not shutdown_event.is_set():
        # 从transition队列中获取数据
        transition_list = transition_queue.get()
        # 将数据转换为transition列表
        transition_list = bytes_to_transitions(buffer=transition_list)

        # 遍历transition列表
        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device) # 将transition数据移动到cuda

            # Skip transitions with NaN values
            # 检查是否有NaN值，如果有则跳过
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition["action"],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            replay_buffer.add(**transition) # 将transition数据添加到在线replay_buffer中

            # Add to offline buffer if it's an intervention
            # 如果在线policy数据当中有干预数据，则将数据添加到离线回放缓冲区当中
            if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
                "is_intervention"
            ):
                offline_replay_buffer.add(**transition)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """Process all available interaction messages from the queue.

    Args:
        interaction_message_queue: Queue for receiving interaction messages
        interaction_step_shift: Amount to shift interaction step by
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown

    Returns:
        dict | None: The last interaction message processed, or None if none were processed
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")

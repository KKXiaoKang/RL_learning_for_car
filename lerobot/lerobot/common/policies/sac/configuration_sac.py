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

from dataclasses import dataclass, field

from lerobot.common.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.common.optim.optimizers import MultiAdamConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


def is_image_feature(key: str) -> bool:
    """Check if a feature key represents an image feature.

    Args:
        key: The feature key to check

    Returns:
        True if the key represents an image feature, False otherwise
    """
    return key.startswith(OBS_IMAGE)


@dataclass
class ConcurrencyConfig:
    """Configuration for the concurrency of the actor and learner.
    Possible values are:
    - "threads": Use threads for the actor and learner.
    - "processes": Use processes for the actor and learner.
    """

    actor: str = "threads"
    learner: str = "threads"


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@dataclass
class CriticNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    final_activation: str | None = None


@dataclass
class ActorNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True


@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    std_min: float = 1e-5
    std_max: float = 10.0
    init_final: float = 0.05


@PreTrainedConfig.register_subclass("sac")
@dataclass
class SACConfig(PreTrainedConfig):
    """Soft Actor-Critic (SAC) configuration.

    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework. It learns a policy and a Q-function simultaneously
    using experience collected from the environment.

    This configuration class contains all the parameters needed to define a SAC agent,
    including network architectures, optimization settings, and algorithm-specific
    hyperparameters.
    """

    # Mapping of feature types to normalization modes
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Statistics for normalizing different types of inputs
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            OBS_IMAGE: {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            OBS_STATE: {
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
            ACTION: {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
        }
    )

    # Architecture specifics
    # Device to run the model on (e.g., "cuda", "cpu")
    device: str = "cpu"
    # Device to store the model on
    storage_device: str = "cpu"
    # Name of the vision encoder model (Set to "helper2424/resnet10" for hil serl resnet10)
    vision_encoder_name: str | None = None
    # Whether to freeze the vision encoder during training
    freeze_vision_encoder: bool = True
    # Whether to disable vision features entirely (useful for debugging or ablation studies)
    disable_vision_features: bool = False
    # Whether to enable ResNet feature visualization (publishes features to ROS topic)
    enable_feature_visualization: bool = False
    # Hidden dimension size for the image encoder
    image_encoder_hidden_dim: int = 32
    # Whether to use a shared encoder for actor and critic
    shared_encoder: bool = True
    # Number of discrete actions, eg for gripper actions
    num_discrete_actions: int | None = None
    # Dimension of the image embedding pooling
    image_embedding_pooling_dim: int = 8

    # Training parameter
    # Number of steps for online training
    online_steps: int = 1000000
    # Seed for the online environment
    online_env_seed: int = 10000
    # Capacity of the online replay buffer
    online_buffer_capacity: int = 100000
    # Capacity of the offline replay buffer
    offline_buffer_capacity: int = 100000
    # Whether to use asynchronous prefetching for the buffers
    async_prefetch: bool = False
    # Number of steps before learning starts
    online_step_before_learning: int = 100
    # Frequency of policy updates
    policy_update_freq: int = 1

    # SAC algorithm parameters
    # Discount factor for the SAC algorithm
    discount: float = 0.99
    # Initial temperature value
    temperature_init: float = 1.0
    # Number of critics in the ensemble
    num_critics: int = 2
    # Number of subsampled critics for training
    num_subsample_critics: int | None = None
    # Learning rate for the critic network
    critic_lr: float = 3e-4
    # Learning rate for the actor network
    actor_lr: float = 3e-4
    # Learning rate for the temperature parameter
    temperature_lr: float = 3e-4
    # Weight for the critic target update
    critic_target_update_weight: float = 0.005
    # Update-to-data ratio for the UTD algorithm (If you want enable utd_ratio, you need to set it to >1)
    utd_ratio: int = 1
    # Hidden dimension size for the state encoder
    state_encoder_hidden_dim: int = 256
    # Dimension of the latent space
    latent_dim: int = 256
    # Target entropy for the SAC algorithm
    target_entropy: float | None = None
    # Whether to use backup entropy for the SAC algorithm
    use_backup_entropy: bool = True
    # Gradient clipping norm for the SAC algorithm
    grad_clip_norm: float = 40.0

    # Network configuration
    # Configuration for the critic network architecture
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for the actor network architecture
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # Configuration for the policy parameters
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    # Configuration for the discrete critic network
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for actor-learner architecture
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # Configuration for concurrency settings (you can use threads or processes for the actor and learner)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    
    # Warm-up configuration for loading pre-trained parameters
    # Path to warm-up model file (e.g., MLP BC model safetensors file)
    warmup_model_path: str | None = None
    # Whether to enable warm-up parameter loading
    enable_warmup: bool = False
    # Whether to freeze parameters loaded from warm-up model
    warmup_freeze_loaded_params: bool = False
    # Whether to require strict parameter matching during warm-up loading
    warmup_strict_loading: bool = False

    # Behavior Cloning (BC) hybrid loss configuration
    # Initial weight for BC loss in hybrid BC+SAC loss (0.0-1.0)
    bc_initial_weight: float = 0.5
    # Final weight for BC loss after decay (0.0-1.0)  
    bc_final_weight: float = 0.01
    # Number of training steps over which to decay BC weight
    bc_decay_steps: int = 50000

    # ACT Actor Configuration
    # Whether to use ACT Transformer Actor instead of MLP Actor
    use_act_actor: bool = False
    # Whether to use sequence version of ACT Actor (processes observation history)
    use_sequence_act_actor: bool = False
    # Observation history length for sequence ACT Actor
    obs_history_length: int = 5
    # Action sequence length (chunk size) for sequence ACT Actor
    act_chunk_size: int = 8
    
    # ACT Transformer architecture parameters
    # Hidden dimension of ACT Transformer
    act_dim_model: int = 512
    # Number of attention heads in ACT Transformer
    act_n_heads: int = 8
    # Feedforward dimension in ACT Transformer
    act_dim_feedforward: int = 3200
    # Number of encoder layers in ACT Transformer
    act_n_encoder_layers: int = 4
    # Number of decoder layers in ACT Transformer
    act_n_decoder_layers: int = 4  # 序列ACT需要更多decoder层
    # Dropout rate in ACT Transformer
    act_dropout: float = 0.1
    # Feedforward activation function in ACT Transformer
    act_feedforward_activation: str = "relu"
    # Whether to use pre-normalization in ACT Transformer
    act_pre_norm: bool = False
    # Maximum sequence length for positional encoding
    act_max_seq_length: int = 10

    # Q-chunking Configuration
    # Whether to enable Q-chunking algorithm
    enable_q_chunking: bool = True
    # Q-chunking strategy: 'standard', 'conservative', 'temporal_weighted'
    q_chunking_strategy: str = "standard"
    # Q-chunking horizon: number of actions to consider in non-standard strategies
    q_chunking_horizon: int = 3
    # Q-chunking temporal decay factor for weighted strategy
    q_chunking_decay: float = 0.9
    # Q-chunking entropy scaling strategy: 'linear', 'sqrt', 'log', 'none'
    q_chunking_entropy_scaling: str = "linear"

    # Optimizations
    use_torch_compile: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Any validation specific to SAC configuration
        self._validate_q_chunking_config()
    
    def _validate_q_chunking_config(self):
        """Validate Q-chunking configuration parameters"""
        # Q-chunking can only be enabled with sequence ACT actor
        if self.enable_q_chunking and not (self.use_act_actor and self.use_sequence_act_actor):
            raise ValueError(
                "Q-chunking (enable_q_chunking=True) requires sequence ACT actor "
                "(use_act_actor=True and use_sequence_act_actor=True)"
            )
        
        # Validate Q-chunking strategy
        valid_strategies = ['standard', 'conservative', 'temporal_weighted']
        if self.q_chunking_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid q_chunking_strategy '{self.q_chunking_strategy}'. "
                f"Must be one of: {valid_strategies}"
            )
        
        # Validate Q-chunking horizon
        if self.q_chunking_horizon < 1:
            raise ValueError("q_chunking_horizon must be >= 1")
        
        if self.q_chunking_horizon > self.act_chunk_size:
            raise ValueError(
                f"q_chunking_horizon ({self.q_chunking_horizon}) cannot be larger than "
                f"act_chunk_size ({self.act_chunk_size})"
            )
        
        # Validate temporal decay factor
        if not (0.0 < self.q_chunking_decay <= 1.0):
            raise ValueError("q_chunking_decay must be in range (0.0, 1.0]")
        
        # Validate entropy scaling strategy
        valid_entropy_strategies = ['linear', 'sqrt', 'log', 'none']
        if self.q_chunking_entropy_scaling not in valid_entropy_strategies:
            raise ValueError(
                f"Invalid q_chunking_entropy_scaling '{self.q_chunking_entropy_scaling}'. "
                f"Must be one of: {valid_entropy_strategies}"
            )

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
                "temperature": {"lr": self.temperature_lr},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        """ 检查config当中是否有任何输入特征键以observation.image开头"""
        has_image = any(is_image_feature(key) for key in self.input_features)
        """ 检查config当中是否有observation.state特征键 | 如果没有则报错 """
        has_state = OBS_STATE in self.input_features
        
        # If vision features are explicitly disabled, we only require state features
        if getattr(self, 'disable_vision_features', False):
            if not has_state:
                raise ValueError(
                    "When vision features are disabled (disable_vision_features=True), "
                    "you must provide 'observation.state' in the input features"
                )
        else:
            # Normal validation: require either state or image
            if not (has_state or has_image):
                raise ValueError(
                    "You must provide either 'observation.state' or an image observation "
                    "(key starting with 'observation.image') in the input features"
                )

        """ 检查config当中是否有action特征键 | 如果没有则报错 """
        if "action" not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return None  # SAC typically predicts one action at a time

    @property
    def reward_delta_indices(self) -> None:
        return None

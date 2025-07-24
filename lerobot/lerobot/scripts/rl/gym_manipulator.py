# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Robot Environment for LeRobot Manipulation Tasks

This module provides a comprehensive gym-compatible environment for robot manipulation
with support for:
- Multiple robot types (SO100, SO101, Koch and Moss)
- Human intervention via leader-follower control or gamepad

- End-effector and joint space control
- Image processing (cropping and resizing)

The environment is built using a composable wrapper pattern where each wrapper
adds specific functionality to the base RobotEnv.

Example:
    env = make_robot_env(cfg)
    obs, info = env.reset()
    action = policy.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
"""

import logging
import time
from collections import deque
from threading import Lock
from typing import Annotated, Any, Sequence, List

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
import matplotlib.pyplot as plt

from lerobot.common.cameras import opencv  # noqa: F401
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.model.kinematics import RobotKinematics
from lerobot.common.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.common.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
)
from lerobot.common.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
from lerobot.common.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import log_say
from lerobot.configs import parser

logging.basicConfig(level=logging.INFO)


def reset_follower_position(robot_arm, target_position):
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict], dtype=np.float32
    )
    trajectory = torch.from_numpy(
        np.linspace(current_position, target_position, 50)
    )  # NOTE: 30 is just an arbitrary number
    for pose in trajectory:
        action_dict = dict(zip(current_position_dict, pose, strict=False))
        robot_arm.bus.sync_write("Goal_Position", action_dict)
        busy_wait(0.015)


class TorchBox(gym.spaces.Box):
    """
    A version of gym.spaces.Box that handles PyTorch tensors.

    This class extends gym.spaces.Box to work with PyTorch tensors,
    providing compatibility between NumPy arrays and PyTorch tensors.
    """

    def __init__(
        self,
        low: float | Sequence[float] | np.ndarray,
        high: float | Sequence[float] | np.ndarray,
        shape: Sequence[int] | None = None,
        np_dtype: np.dtype | type = np.float32,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize the PyTorch-compatible Box space.

        Args:
            low: Lower bounds of the space.
            high: Upper bounds of the space.
            shape: Shape of the space. If None, inferred from low and high.
            np_dtype: NumPy data type for internal storage.
            torch_dtype: PyTorch data type for tensor conversion.
            device: PyTorch device for returned tensors.
            seed: Random seed for sampling.
        """
        super().__init__(low, high, shape=shape, dtype=np_dtype, seed=seed)
        self.torch_dtype = torch_dtype
        self.device = device

    def sample(self) -> torch.Tensor:
        """
        Sample a random point from the space.

        Returns:
            A PyTorch tensor within the space bounds.
        """
        arr = super().sample()
        return torch.as_tensor(arr, dtype=self.torch_dtype, device=self.device)

    def contains(self, x: torch.Tensor) -> bool:
        """
        Check if a tensor is within the space bounds.

        Args:
            x: The PyTorch tensor to check.

        Returns:
            Boolean indicating whether the tensor is within bounds.
        """
        # Move to CPU/numpy and cast to the internal dtype
        arr = x.detach().cpu().numpy().astype(self.dtype, copy=False)
        return super().contains(arr)

    def seed(self, seed: int | np.random.Generator | None = None):
        """
        Set the random seed for sampling.

        Args:
            seed: The random seed to use.

        Returns:
            List containing the seed.
        """
        super().seed(seed)
        return [seed]

    def __repr__(self) -> str:
        """
        Return a string representation of the space.

        Returns:
            Formatted string with space details.
        """
        return (
            f"TorchBox({self.low_repr}, {self.high_repr}, {self.shape}, "
            f"np={self.dtype.name}, torch={self.torch_dtype}, device={self.device})"
        )


class TorchActionWrapper(gym.Wrapper):
    """
    Wrapper that changes the action space to use PyTorch tensors.

    This wrapper modifies the action space to return PyTorch tensors when sampled
    and handles converting PyTorch actions to NumPy when stepping the environment.
    """

    def __init__(self, env: gym.Env, device: str):
        """
        Initialize the PyTorch action space wrapper.

        Args:
            env: The environment to wrap.
            device: The PyTorch device to use for tensor operations.
        """
        super().__init__(env)
        self.action_space = TorchBox(
            low=env.action_space.low,
            high=env.action_space.high,
            shape=env.action_space.shape,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def step(self, action: torch.Tensor):
        """
        Step the environment with a PyTorch tensor action.

        This method handles conversion from PyTorch tensors to NumPy arrays
        for compatibility with the underlying environment.

        Args:
            action: PyTorch tensor action to take.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if action.dim() == 2:
            action = action.squeeze(0)
        action = action.detach().cpu().numpy()
        return self.env.step(action)


class RobotEnv(gym.Env):
    """
    Gym-compatible environment for evaluating robotic control policies with integrated human intervention.

    This environment wraps a robot interface to provide a consistent API for policy evaluation. It supports both relative (delta)
    and absolute joint position commands and automatically configures its observation and action spaces based on the robot's
    sensors and configuration.
    """

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
    ):
        """
        Initialize the RobotEnv environment.

        The environment is set up with a robot interface, which is used to capture observations and send joint commands. The setup
        supports both relative (delta) adjustments and absolute joint positions for controlling the robot.

        Args:
            robot: The robot interface object used to connect and interact with the physical robot.
            display_cameras: If True, the robot's camera feeds will be displayed during execution.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        # Read initial joint positions using the bus
        self.current_joint_positions = self._get_observation()["agent_pos"]

        self.use_gripper = use_gripper

        self._setup_spaces()

    def _get_observation(self) -> np.ndarray:
        """Helper to convert a dictionary from bus.sync_read to an ordered numpy array."""
        obs_dict = self.robot.get_observation()
        joint_positions = np.array([obs_dict[name] for name in self._joint_names], dtype=np.float32)

        images = {key: obs_dict[key] for key in self._image_keys}
        return {"agent_pos": joint_positions, "pixels": images}

    def _setup_spaces(self):
        """
        Dynamically configure the observation and action spaces based on the robot's capabilities.

        Observation Space:
            - For keys with "image": A Box space with pixel values ranging from 0 to 255.
            - For non-image keys: A nested Dict space is created under 'observation.state' with a suitable range.

        Action Space:
            - The action space is defined as a Box space representing joint position commands. It is defined as relative (delta)
              or absolute, based on the configuration.
        """
        example_obs = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if "pixels" in example_obs:
            prefix = "observation.images" if len(example_obs["pixels"]) > 1 else "observation.image"
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=example_obs["pixels"][key].shape, dtype=np.uint8
                )
                for key in example_obs["pixels"]
            }

        observation_spaces["observation.state"] = gym.spaces.Box(
            low=0,
            high=10,
            shape=example_obs["agent_pos"].shape,
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to its initial state.
        This method resets the step counter and clears any episodic data.

        Args:
            seed: A seed for random number generation to ensure reproducibility.
            options: Additional options to influence the reset behavior.

        Returns:
            A tuple containing:
                - observation (dict): The initial sensor observation.
                - info (dict): A dictionary with supplementary information, including the key "is_intervention".
        """
        super().reset(seed=seed, options=options)

        self.robot.reset()

        # Capture the initial observation.
        observation = self._get_observation()

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None

        return observation, {"is_intervention": False}

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute a single step within the environment using the specified action.

        The provided action is processed and sent to the robot as joint position commands
        that may be either absolute values or deltas based on the environment configuration.

        Args:
            action: The commanded joint positions as a numpy array or torch tensor.

        Returns:
            A tuple containing:
                - observation (dict): The new sensor observation after taking the step.
                - reward (float): The step reward (default is 0.0 within this wrapper).
                - terminated (bool): True if the episode has reached a terminal state.
                - truncated (bool): True if the episode was truncated (e.g., time constraints).
                - info (dict): Additional debugging information including intervention status.
        """
        self.current_joint_positions = self._get_observation()["agent_pos"]

        action_dict = {"delta_x": action[0], "delta_y": action[1], "delta_z": action[2]}

        # 1.0 action corresponds to no-op action
        action_dict["gripper"] = action[3] if self.use_gripper else 1.0

        self.robot.send_action(action_dict)

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            {"is_intervention": False},
        )

    def render(self):
        """
        Render the current state of the environment by displaying the robot's camera feeds.
        """
        import cv2

        observation = self._get_observation()
        image_keys = [key for key in observation if "image" in key]

        for key in image_keys:
            cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self):
        """
        Close the environment and clean up resources by disconnecting the robot.

        If the robot is currently connected, this method properly terminates the connection to ensure that all
        associated resources are released.
        """
        if self.robot.is_connected:
            self.robot.disconnect()


class AddJointVelocityToObservation(gym.ObservationWrapper):
    """
    Wrapper that adds joint velocity information to the observation.

    This wrapper computes joint velocities by tracking changes in joint positions over time,
    and extends the observation space to include these velocities.
    """

    def __init__(self, env, joint_velocity_limits=100.0, fps=30, num_dof=6):
        """
        Initialize the joint velocity wrapper.

        Args:
            env: The environment to wrap.
            joint_velocity_limits: Maximum expected joint velocity for space bounds.
            fps: Frames per second used to calculate velocity (position delta / time).
            num_dof: Number of degrees of freedom (joints) in the robot.
        """
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["observation.state"].low
        old_high = self.observation_space["observation.state"].high
        old_shape = self.observation_space["observation.state"].shape

        self.last_joint_positions = np.zeros(num_dof)

        new_low = np.concatenate([old_low, np.ones(num_dof) * -joint_velocity_limits])
        new_high = np.concatenate([old_high, np.ones(num_dof) * joint_velocity_limits])

        new_shape = (old_shape[0] + num_dof,)

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

        self.dt = 1.0 / fps

    def observation(self, observation):
        """
        Add joint velocity information to the observation.

        Args:
            observation: The original observation from the environment.

        Returns:
            The modified observation with joint velocities.
        """
        joint_velocities = (observation["agent_pos"] - self.last_joint_positions) / self.dt
        self.last_joint_positions = observation["agent_pos"]
        observation["agent_pos"] = np.concatenate([observation["agent_pos"], joint_velocities], axis=-1)
        return observation


class AddCurrentToObservation(gym.ObservationWrapper):
    """
    Wrapper that adds motor current information to the observation.

    This wrapper extends the observation space to include the current values
    from each motor, providing information about the forces being applied.
    """

    def __init__(self, env, max_current=500, num_dof=6):
        """
        Initialize the current observation wrapper.

        Args:
            env: The environment to wrap.
            max_current: Maximum expected current for space bounds.
            num_dof: Number of degrees of freedom (joints) in the robot.
        """
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["observation.state"].low
        old_high = self.observation_space["observation.state"].high
        old_shape = self.observation_space["observation.state"].shape

        new_low = np.concatenate([old_low, np.zeros(num_dof)])
        new_high = np.concatenate([old_high, np.ones(num_dof) * max_current])

        new_shape = (old_shape[0] + num_dof,)

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        """
        Add current information to the observation.

        Args:
            observation: The original observation from the environment.

        Returns:
            The modified observation with current values.
        """
        present_current_observation = self.unwrapped._get_observation()["agent_pos"]
        observation["agent_pos"] = np.concatenate(
            [observation["agent_pos"], present_current_observation], axis=-1
        )
        return observation


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier, device="cuda"):
        """
        Wrapper to add reward prediction to the environment using a trained classifier.

        Args:
            env: The environment to wrap.
            reward_classifier: The reward classifier model.
            device: The device to run the model on.
        """
        self.env = env

        self.device = device

        self.reward_classifier = torch.compile(reward_classifier)
        self.reward_classifier.to(self.device)

    def step(self, action):
        """
        Execute a step and compute the reward using the classifier.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        observation, _, terminated, truncated, info = self.env.step(action)

        images = {}
        for key in observation:
            if "image" in key:
                images[key] = observation[key].to(self.device, non_blocking=(self.device == "cuda"))
                if images[key].dim() == 3:
                    images[key] = images[key].unsqueeze(0)

        start_time = time.perf_counter()
        with torch.inference_mode():
            success = (
                self.reward_classifier.predict_reward(images, threshold=0.7)
                if self.reward_classifier is not None
                else 0.0
            )
        info["Reward classifier frequency"] = 1 / (time.perf_counter() - start_time)

        reward = 0.0
        if success == 1.0:
            terminated = True
            reward = 1.0

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            The initial observation and info from the wrapped environment.
        """
        return self.env.reset(seed=seed, options=options)


class TimeLimitWrapper(gym.Wrapper):
    """
    Wrapper that adds a time limit to episodes and tracks execution time.

    This wrapper terminates episodes after a specified time has elapsed, providing
    better control over episode length.
    """

    def __init__(self, env, control_time_s, fps):
        """
        Initialize the time limit wrapper.

        Args:
            env: The environment to wrap.
            control_time_s: Maximum episode duration in seconds.
            fps: Frames per second for calculating the maximum number of steps.
        """
        self.env = env
        self.control_time_s = control_time_s
        self.fps = fps

        self.last_timestamp = 0.0
        self.episode_time_in_s = 0.0

        self.max_episode_steps = int(self.control_time_s * self.fps)

        self.current_step = 0

    def step(self, action):
        """
        Step the environment and track time elapsed.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        time_since_last_step = time.perf_counter() - self.last_timestamp
        self.episode_time_in_s += time_since_last_step
        self.last_timestamp = time.perf_counter()
        self.current_step += 1
        # check if last timestep took more time than the expected fps
        if 1.0 / time_since_last_step < self.fps:
            logging.debug(f"Current timestep exceeded expected fps {self.fps}")

        if self.current_step >= self.max_episode_steps:
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment and time tracking.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            The initial observation and info from the wrapped environment.
        """
        self.episode_time_in_s = 0.0
        self.last_timestamp = time.perf_counter()
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)


class ImageCropResizeWrapper(gym.Wrapper):
    """
    Wrapper that crops and resizes image observations.

    This wrapper processes image observations to focus on relevant regions by
    cropping and then resizing to a standard size.
    """

    def __init__(
        self,
        env,
        crop_params_dict: dict[str, Annotated[tuple[int], 4]],
        resize_size=None,
    ):
        """
        Initialize the image crop and resize wrapper.

        Args:
            env: The environment to wrap.
            crop_params_dict: Dictionary mapping image observation keys to crop parameters
                             (top, left, height, width).
            resize_size: Target size for resized images (height, width). Defaults to (128, 128).
        """
        super().__init__(env)
        self.env = env
        self.crop_params_dict = crop_params_dict
        print(f"obs_keys , {self.env.observation_space}")
        print(f"crop params dict {crop_params_dict.keys()}")
        for key_crop in crop_params_dict:
            if key_crop not in self.env.observation_space.keys():  # noqa: SIM118
                raise ValueError(f"Key {key_crop} not in observation space")
        for key in crop_params_dict:
            new_shape = (3, resize_size[0], resize_size[1])
            self.observation_space[key] = gym.spaces.Box(low=0, high=255, shape=new_shape)

        self.resize_size = resize_size
        if self.resize_size is None:
            self.resize_size = (128, 128)

    def step(self, action):
        """
        Step the environment and process image observations.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) with processed images.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self.crop_params_dict:
            device = obs[k].device
            if obs[k].dim() >= 3:
                # Reshape to combine height and width dimensions for easier calculation
                batch_size = obs[k].size(0)
                channels = obs[k].size(1)
                flattened_spatial_dims = obs[k].view(batch_size, channels, -1)

                # Calculate standard deviation across spatial dimensions (H, W)
                # If any channel has std=0, all pixels in that channel have the same value
                # This is helpful if one camera mistakenly covered or the image is black
                std_per_channel = torch.std(flattened_spatial_dims, dim=2)
                if (std_per_channel <= 0.02).any():
                    logging.warning(
                        f"Potential hardware issue detected: All pixels have the same value in observation {k}"
                    )

            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()

            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            # TODO (michel-aractingi): Bug in resize, it returns values outside [0, 1]
            obs[k] = obs[k].clamp(0.0, 1.0)
            obs[k] = obs[k].to(device)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment and process image observations.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) with processed images.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        for k in self.crop_params_dict:
            device = obs[k].device
            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()
            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            obs[k] = obs[k].clamp(0.0, 1.0)
            obs[k] = obs[k].to(device)
        return obs, info


class ConvertToLeRobotObservation(gym.ObservationWrapper):
    """
    Wrapper that converts standard observations to LeRobot format.

    This wrapper processes observations to match the expected format for LeRobot,
    including normalizing image values and moving tensors to the specified device.
    """

    def __init__(self, env, device: str = "cpu"):
        """
        Initialize the LeRobot observation converter.

        Args:
            env: The environment to wrap.
            device: Target device for the observation tensors.
        """
        super().__init__(env)

        self.device = torch.device(device)

    def observation(self, observation):
        """
        Convert observations to LeRobot format.

        Args:
            observation: The original observation from the environment.

        Returns:
            The processed observation with normalized images and proper tensor formats.
        """
        observation = preprocess_observation(observation)
        observation = {
            key: observation[key].to(self.device, non_blocking=self.device.type == "cuda")
            for key in observation
        }
        return observation


class ResetWrapper(gym.Wrapper):
    """
    Wrapper that handles environment reset procedures.

    This wrapper provides additional functionality during environment reset,
    including the option to reset to a fixed pose or allow manual reset.
    """

    def __init__(
        self,
        env: RobotEnv,
        reset_pose: np.ndarray | None = None,
        reset_time_s: float = 5,
    ):
        """
        Initialize the reset wrapper.

        Args:
            env: The environment to wrap.
            reset_pose: Fixed joint positions to reset to. If None, manual reset is used.
            reset_time_s: Time in seconds to wait after reset or allowed for manual reset.
        """
        super().__init__(env)
        self.reset_time_s = reset_time_s
        self.reset_pose = reset_pose
        self.robot = self.unwrapped.robot

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment with either fixed or manual reset procedure.

        If reset_pose is provided, the robot will move to that position.
        Otherwise, manual teleoperation control is allowed for reset_time_s seconds.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            The initial observation and info from the wrapped environment.
        """
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.unwrapped.robot, self.reset_pose)
            log_say("Reset the environment done.", play_sounds=True)

            if hasattr(self.env, "robot_leader"):
                self.env.robot_leader.bus.sync_write("Torque_Enable", 1)
                log_say("Reset the leader robot.", play_sounds=True)
                reset_follower_position(self.env.robot_leader, self.reset_pose)
                log_say("Reset the leader robot done.", play_sounds=True)
        else:
            log_say(
                f"Manually reset the environment for {self.reset_time_s} seconds.",
                play_sounds=True,
            )
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.reset_time_s:
                action = self.env.robot_leader.get_action()
                self.unwrapped.robot.send_action(action)

            log_say("Manual reset of the environment done.", play_sounds=True)

        busy_wait(self.reset_time_s - (time.perf_counter() - start_time))

        return super().reset(seed=seed, options=options)


def plot_episode_rewards(rewards: List[float], save_path: str = "episode_rewards_analysis.png", show_plot: bool = False, success_count: int = None, total_episodes: int = None) -> str:
    """
    绘制回合奖励统计图表，包含趋势、分布、成功率和统计信息
    
    Args:
        rewards: 每个回合的奖励列表
        save_path: 保存图像的路径
        show_plot: 是否显示图表窗口
        success_count: 真正成功的episode数量（如果为None则使用旧的阈值计算）
        total_episodes: 总episode数量
        
    Returns:
        保存的图像文件路径
    """
    if not rewards:
        print("WARNING: 没有奖励数据可供分析")
        return save_path
    
    try:
        # 设置matplotlib不显示GUI（避免在无GUI环境中卡住）
        import matplotlib
        if not show_plot:
            matplotlib.use('Agg')
        
        # 创建2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Episode Rewards Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. 回合奖励趋势图
        episodes = range(1, len(rewards) + 1)
        ax1.plot(episodes, rewards, 'b-', marker='o', linewidth=2, markersize=4, alpha=0.7)
        ax1.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode Number')
        ax1.set_ylabel('Reward Value')
        ax1.grid(True, alpha=0.3)
        
        # 添加移动平均线（如果有足够的数据点）
        if len(rewards) > 5:
            window_size = min(10, len(rewards) // 3)
            moving_avg = []
            for i in range(len(rewards)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(rewards[start_idx:i+1]))
            ax1.plot(episodes, moving_avg, 'r--', linewidth=3, 
                    label=f'Moving Average (window={window_size})', alpha=0.8)
            ax1.legend()
        
        # 添加零线作为参考
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax1.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Success Threshold')
        
        # 2. 奖励分布直方图
        unique_rewards = len(set(rewards))
        bins = min(20, max(5, unique_rewards))
        n, bins_edges, patches = ax2.hist(rewards, bins=bins, alpha=0.7, color='skyblue', 
                                         edgecolor='black', linewidth=1)
        ax2.set_title('Reward Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Reward Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计线
        mean_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        ax2.axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_reward:.3f}')
        ax2.axvline(median_reward, color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {median_reward:.3f}')
        ax2.legend()
        
        # 3. 成功率饼图
        if success_count is not None and total_episodes is not None:
            # 使用传入的准确成功计数
            failure_count = total_episodes - success_count
        else:
            # 回退到旧的阈值计算方式（向后兼容）
            success_threshold = 0.9  # 考虑浮点数精度，将接近1.0的值视为成功
            success_count = sum(1 for r in rewards if r >= success_threshold)
            failure_count = len(rewards) - success_count
        
        if success_count > 0 or failure_count > 0:
            labels = ['Success', 'Failure']
            sizes = [success_count, failure_count]
            colors = ['lightgreen', 'lightcoral']
            explode = (0.1, 0)  # 突出成功部分
            
            wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, 
                                              colors=colors, autopct='%1.1f%%',
                                              shadow=True, startangle=90)
            
            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
                
            ax3.set_title(f'Success Rate\n({success_count}/{len(rewards)} episodes)', 
                          fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Success Rate', fontsize=14, fontweight='bold')
        
        # 4. 统计信息表格
        ax4.axis('tight')
        ax4.axis('off')
        
        # 计算详细统计信息
        if success_count is not None and total_episodes is not None:
            success_rate = (success_count / total_episodes * 100) if total_episodes > 0 else 0
        else:
            success_rate = (success_count / len(rewards) * 100) if rewards else 0
        std_reward = np.std(rewards) if len(rewards) > 1 else 0
        
        stats_data = [
            ['Total Episodes', f'{len(rewards)}'],
            ['Success Rate', f'{success_rate:.1f}%'],
            ['Mean Reward', f'{mean_reward:.4f}'],
            ['Std Deviation', f'{std_reward:.4f}'],
            ['Min Reward', f'{np.min(rewards):.4f}'],
            ['Max Reward', f'{np.max(rewards):.4f}'],
            ['Median Reward', f'{median_reward:.4f}'],
            ['Q1 (25%)', f'{np.percentile(rewards, 25):.4f}'],
            ['Q3 (75%)', f'{np.percentile(rewards, 75):.4f}']
        ]
        
        table = ax4.table(cellText=stats_data,
                          colLabels=['Metric', 'Value'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        # 设置表格样式
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 标题行
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j == 0:  # 指标列
                        cell.set_facecolor('#f8f9fa')
                    else:  # 数值列
                        cell.set_facecolor('#ffffff')
                    cell.set_text_props(weight='normal')
        
        ax4.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # 关闭图形，释放内存
        
        print(f"SUCCESS: 图表已成功保存至 {save_path}")
        return save_path
        
    except Exception as e:
        print(f"ERROR: 生成图表时出错: {e}")
        print(f"WARNING: 图表保存可能失败，但程序将继续运行")
        return save_path


def print_rewards_summary(rewards: List[float], success_count: int = None, total_episodes: int = None):
    """
    打印奖励统计摘要
    
    Args:
        rewards: 每个回合的奖励列表
        success_count: 真正成功的episode数量（如果为None则使用旧的阈值计算）
        total_episodes: 总episode数量
    """
    if not rewards:
        print("WARNING: 没有奖励数据可供分析")
        return
    
    print("\n" + "="*60)
    print("Episode Rewards Analysis Summary")
    print("="*60)
    
    if success_count is not None and total_episodes is not None:
        # 使用传入的准确成功计数
        success_rate = (success_count / total_episodes * 100) if total_episodes > 0 else 0
    else:
        # 回退到旧的阈值计算方式（向后兼容）
        success_count = sum(1 for r in rewards if r >= 0.9)
        success_rate = (success_count / len(rewards) * 100) if rewards else 0
    
    print(f"Basic Statistics:")
    print(f"  • Total Episodes: {len(rewards)}")
    print(f"  • Success Episodes: {success_count}")
    print(f"  • Success Rate: {success_rate:.1f}%")
    print(f"  • Mean Reward: {np.mean(rewards):.4f}")
    print(f"  • Std Deviation: {np.std(rewards):.4f}")
    print(f"  • Min Reward: {np.min(rewards):.4f}")
    print(f"  • Max Reward: {np.max(rewards):.4f}")
    print(f"  • Median Reward: {np.median(rewards):.4f}")
    
    # 分析奖励趋势（如果有足够数据）
    if len(rewards) >= 10:
        first_half = rewards[:len(rewards)//2]
        second_half = rewards[len(rewards)//2:]
        first_half_avg = np.mean(first_half)
        second_half_avg = np.mean(second_half)
        improvement = second_half_avg - first_half_avg
        
        print(f"\nLearning Trend Analysis:")
        print(f"  • First Half Average: {first_half_avg:.4f}")
        print(f"  • Second Half Average: {second_half_avg:.4f}")
        print(f"  • Improvement: {improvement:+.4f}")
        
        if improvement > 0.05:
            print(f"  STATUS: Learning performance shows upward trend")
        elif improvement < -0.05:
            print(f"  WARNING: Learning performance shows downward trend")
        else:
            print(f"  STATUS: Learning performance is relatively stable")
    
    print("="*60)


class BatchCompatibleWrapper(gym.ObservationWrapper):
    """
    Wrapper that ensures observations are compatible with batch processing.

    This wrapper adds a batch dimension to observations that don't already have one,
    making them compatible with models that expect batched inputs.
    """

    def __init__(self, env):
        """
        Initialize the batch compatibility wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Add batch dimensions to observations if needed.

        Args:
            observation: Dictionary of observation tensors.

        Returns:
            Dictionary of observation tensors with batch dimensions.
        """
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
            if "velocity" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


class GripperPenaltyWrapper(gym.RewardWrapper):
    """
    Wrapper that adds penalties for inefficient gripper commands.

    This wrapper modifies rewards to discourage excessive gripper movement
    or commands that attempt to move the gripper beyond its physical limits.
    """

    def __init__(self, env, penalty: float = -0.1):
        """
        Initialize the gripper penalty wrapper.

        Args:
            env: The environment to wrap.
            penalty: Negative reward value to apply for inefficient gripper actions.
        """
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_state = None

    def reward(self, reward, action):
        """
        Apply penalties to reward based on gripper actions.

        Args:
            reward: The original reward from the environment.
            action: The action that was taken.

        Returns:
            Modified reward with penalty applied if necessary.
        """
        gripper_state_normalized = self.last_gripper_state / self.unwrapped.robot.config.max_gripper_pos

        action_normalized = action - 1.0  # action / MAX_GRIPPER_COMMAND

        gripper_penalty_bool = (gripper_state_normalized < 0.5 and action_normalized > 0.5) or (
            gripper_state_normalized > 0.75 and action_normalized < -0.5
        )

        return reward + self.penalty * int(gripper_penalty_bool)

    def step(self, action):
        """
        Step the environment and apply gripper penalties.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) with penalty applied.
        """
        self.last_gripper_state = self.unwrapped.robot.bus.sync_read("Present_Position")["gripper"]

        gripper_action = action[-1]
        obs, reward, terminated, truncated, info = self.env.step(action)
        gripper_penalty = self.reward(reward, gripper_action)

        info["discrete_penalty"] = gripper_penalty

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment and penalty tracking.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset.

        Returns:
            The initial observation and info with gripper penalty initialized.
        """
        self.last_gripper_state = None
        obs, info = super().reset(**kwargs)
        info["gripper_penalty"] = 0.0
        return obs, info


class GripperActionWrapper(gym.ActionWrapper):
    """
    Wrapper that processes gripper control commands.

    This wrapper quantizes and processes gripper commands, adding a sleep time between
    consecutive gripper actions to prevent rapid toggling.
    """

    def __init__(self, env, quantization_threshold: float = 0.2, gripper_sleep: float = 0.0):
        """
        Initialize the gripper action wrapper.

        Args:
            env: The environment to wrap.
            quantization_threshold: Threshold below which gripper commands are quantized to zero.
            gripper_sleep: Minimum time in seconds between consecutive gripper commands.
        """
        super().__init__(env)
        self.quantization_threshold = quantization_threshold
        self.gripper_sleep = gripper_sleep
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None

    def action(self, action):
        """
        Process gripper commands in the action.

        Args:
            action: The original action from the agent.

        Returns:
            Modified action with processed gripper command.
        """
        if self.gripper_sleep > 0.0:
            if (
                self.last_gripper_action is not None
                and time.perf_counter() - self.last_gripper_action_time < self.gripper_sleep
            ):
                action[-1] = self.last_gripper_action
            else:
                self.last_gripper_action_time = time.perf_counter()
                self.last_gripper_action = action[-1]

        gripper_command = action[-1]
        # Gripper actions are between 0, 2
        # we want to quantize them to -1, 0 or 1
        gripper_command = gripper_command - 1.0

        if self.quantization_threshold is not None:
            # Quantize gripper command to -1, 0 or 1
            gripper_command = (
                np.sign(gripper_command) if abs(gripper_command) > self.quantization_threshold else 0.0
            )
        gripper_command = gripper_command * self.unwrapped.robot.config.max_gripper_pos

        gripper_state = self.unwrapped.robot.bus.sync_read("Present_Position")["gripper"]

        gripper_action_value = np.clip(
            gripper_state + gripper_command, 0, self.unwrapped.robot.config.max_gripper_pos
        )
        action[-1] = gripper_action_value.item()
        return action

    def reset(self, **kwargs):
        """
        Reset the gripper action tracking.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset.

        Returns:
            The initial observation and info.
        """
        obs, info = super().reset(**kwargs)
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None
        return obs, info


class EEObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that adds end-effector pose information to observations.

    This wrapper computes the end-effector pose using forward kinematics
    and adds it to the observation space.
    """

    def __init__(self, env, ee_pose_limits):
        """
        Initialize the end-effector observation wrapper.

        Args:
            env: The environment to wrap.
            ee_pose_limits: Dictionary with 'min' and 'max' keys containing limits for EE pose.
        """
        super().__init__(env)

        # Extend observation space to include end effector pose
        prev_space = self.observation_space["observation.state"]

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=np.concatenate([prev_space.low, ee_pose_limits["min"]]),
            high=np.concatenate([prev_space.high, ee_pose_limits["max"]]),
            shape=(prev_space.shape[0] + 3,),
            dtype=np.float32,
        )

        # Initialize kinematics instance for the appropriate robot type
        robot_type = getattr(env.unwrapped.robot.config, "robot_type", "so101")
        if "so100" in robot_type or "so101" in robot_type:
            # Note to be compatible with the rest of the codebase,
            # we are using the new calibration method for so101 and so100
            robot_type = "so_new_calibration"
        self.kinematics = RobotKinematics(robot_type)

    def observation(self, observation):
        """
        Add end-effector pose to the observation.

        Args:
            observation: Original observation from the environment.

        Returns:
            Enhanced observation with end-effector pose information.
        """
        current_joint_pos = self.unwrapped._get_observation()["agent_pos"]

        current_ee_pos = self.kinematics.forward_kinematics(current_joint_pos, frame="gripper_tip")[:3, 3]
        observation["agent_pos"] = np.concatenate([observation["agent_pos"], current_ee_pos], -1)
        return observation


###########################################################
# Wrappers related to human intervention and input devices
###########################################################


class BaseLeaderControlWrapper(gym.Wrapper):
    """
    Base class for leader-follower robot control wrappers.

    This wrapper enables human intervention through a leader-follower robot setup,
    where the human can control a leader robot to guide the follower robot's movements.
    """

    def __init__(
        self,
        env,
        teleop_device,
        end_effector_step_sizes,
        use_geared_leader_arm: bool = False,
        use_gripper=False,
    ):
        """
        Initialize the base leader control wrapper.

        Args:
            env: The environment to wrap.
            teleop_device: The teleoperation device.
            use_geared_leader_arm: Whether to use a geared leader arm setup.
            use_gripper: Whether to include gripper control.
        """
        super().__init__(env)
        self.robot_leader = teleop_device
        self.robot_follower = env.unwrapped.robot
        self.use_geared_leader_arm = use_geared_leader_arm
        self.use_gripper: bool = use_gripper
        self.end_effector_step_sizes = np.array(list(end_effector_step_sizes.values()))

        # Set up keyboard event tracking
        self._init_keyboard_events()
        self.event_lock = Lock()  # Thread-safe access to events

        # Initialize robot control
        robot_type = getattr(env.unwrapped.robot.config, "robot_type", "so101")
        if "so100" in robot_type or "so101" in robot_type:
            # Note to be compatible with the rest of the codebase,
            # we are using the new calibration method for so101 and so100
            robot_type = "so_new_calibration"
        self.kinematics = RobotKinematics(robot_type)
        self.leader_torque_enabled = True
        self.prev_leader_gripper = None

        # Configure leader arm
        # NOTE: Lower the gains of leader arm for automatic take-over
        # With lower gains we can manually move the leader arm without risk of injury to ourselves or the robot
        # With higher gains, it would be dangerous and difficult to modify the leader's pose while torque is enabled
        # Default value for P_coeff is 32
        self.robot_leader.bus.sync_write("Torque_Enable", 1)
        for motor in self.robot_leader.bus.motors:
            self.robot_leader.bus.write("P_Coefficient", motor, 16)
            self.robot_leader.bus.write("I_Coefficient", motor, 0)
            self.robot_leader.bus.write("D_Coefficient", motor, 16)

        self.leader_tracking_error_queue = deque(maxlen=4)
        self._init_keyboard_listener()

    def _init_keyboard_events(self):
        """
        Initialize the keyboard events dictionary.

        This method sets up tracking for keyboard events used for intervention control.
        It should be overridden in subclasses to add additional events.
        """
        self.keyboard_events = {
            "episode_success": False,
            "episode_end": False,
            "rerecord_episode": False,
        }

    def _handle_key_press(self, key, keyboard_device):
        """
        Handle key press events.

        Args:
            key: The key that was pressed.
            keyboard: The keyboard module with key definitions.

        This method should be overridden in subclasses for additional key handling.
        """
        try:
            if key == keyboard_device.Key.esc:
                self.keyboard_events["episode_end"] = True
                return
            if key == keyboard_device.Key.left:
                self.keyboard_events["rerecord_episode"] = True
                return
            if hasattr(key, "char") and key.char == "s":
                logging.info("Key 's' pressed. Episode success triggered.")
                self.keyboard_events["episode_success"] = True
                return
        except Exception as e:
            logging.error(f"Error handling key press: {e}")

    def _init_keyboard_listener(self):
        """
        Initialize the keyboard listener for intervention control.

        This method sets up keyboard event handling if not in headless mode.
        """
        from pynput import keyboard as keyboard_device

        def on_press(key):
            with self.event_lock:
                self._handle_key_press(key, keyboard_device)

        self.listener = keyboard_device.Listener(on_press=on_press)
        self.listener.start()

    def _check_intervention(self):
        """
        Check if human intervention is needed.

        Returns:
            Boolean indicating whether intervention is needed.

        This method should be overridden in subclasses with specific intervention logic.
        """
        return False

    def _handle_intervention(self, action):
        """
        Process actions during intervention mode.

        Args:
            action: The original action from the agent.

        Returns:
            Tuple of (modified_action, intervention_action).
        """
        if self.leader_torque_enabled:
            self.robot_leader.bus.sync_write("Torque_Enable", 0)
            self.leader_torque_enabled = False

        leader_pos_dict = self.robot_leader.bus.sync_read("Present_Position")
        follower_pos_dict = self.robot_follower.bus.sync_read("Present_Position")

        leader_pos = np.array([leader_pos_dict[name] for name in leader_pos_dict], dtype=np.float32)
        follower_pos = np.array([follower_pos_dict[name] for name in follower_pos_dict], dtype=np.float32)

        self.leader_tracking_error_queue.append(np.linalg.norm(follower_pos[:-1] - leader_pos[:-1]))

        # [:3, 3] Last column of the transformation matrix corresponds to the xyz translation
        leader_ee = self.kinematics.forward_kinematics(leader_pos, frame="gripper_tip")[:3, 3]
        follower_ee = self.kinematics.forward_kinematics(follower_pos, frame="gripper_tip")[:3, 3]

        action = np.clip(leader_ee - follower_ee, -self.end_effector_step_sizes, self.end_effector_step_sizes)
        # Normalize the action to the range [-1, 1]
        action = action / self.end_effector_step_sizes

        if self.use_gripper:
            if self.prev_leader_gripper is None:
                self.prev_leader_gripper = np.clip(
                    leader_pos[-1], 0, self.robot_follower.config.max_gripper_pos
                )

            # Get gripper action delta based on leader pose
            leader_gripper = leader_pos[-1]
            gripper_delta = leader_gripper - self.prev_leader_gripper

            # Normalize by max angle and quantize to {0,1,2}
            normalized_delta = gripper_delta / self.robot_follower.config.max_gripper_pos
            if normalized_delta >= 0.3:
                gripper_action = 2
            elif normalized_delta <= 0.1:
                gripper_action = 0
            else:
                gripper_action = 1

            action = np.append(action, gripper_action)

        return action

    def _handle_leader_teleoperation(self):
        """
        Handle leader teleoperation in non-intervention mode.

        This method synchronizes the leader robot position with the follower.
        """

        prev_leader_pos_dict = self.robot_leader.bus.sync_read("Present_Position")
        prev_leader_pos = np.array(
            [prev_leader_pos_dict[name] for name in prev_leader_pos_dict], dtype=np.float32
        )

        if not self.leader_torque_enabled:
            self.robot_leader.bus.sync_write("Torque_Enable", 1)
            self.leader_torque_enabled = True

        follower_pos_dict = self.robot_follower.bus.sync_read("Present_Position")
        follower_pos = np.array([follower_pos_dict[name] for name in follower_pos_dict], dtype=np.float32)

        goal_pos = {f"{motor}": follower_pos[i] for i, motor in enumerate(self.robot_leader.bus.motors)}
        self.robot_leader.bus.sync_write("Goal_Position", goal_pos)

        self.leader_tracking_error_queue.append(np.linalg.norm(follower_pos[:-1] - prev_leader_pos[:-1]))

    def step(self, action):
        """
        Execute a step with possible human intervention.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        is_intervention = self._check_intervention()

        # NOTE:
        if is_intervention:
            action = self._handle_intervention(action)
        else:
            self._handle_leader_teleoperation()

        # NOTE:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add intervention info
        info["is_intervention"] = is_intervention
        info["action_intervention"] = action

        self.prev_leader_gripper = np.clip(
            self.robot_leader.bus.sync_read("Present_Position")["gripper"],
            0,
            self.robot_follower.config.max_gripper_pos,
        )

        # Check for success or manual termination
        success = self.keyboard_events["episode_success"]
        terminated = terminated or self.keyboard_events["episode_end"] or success

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment and intervention state.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset.

        Returns:
            The initial observation and info.
        """
        self.keyboard_events = dict.fromkeys(self.keyboard_events, False)
        self.leader_tracking_error_queue.clear()
        return super().reset(**kwargs)

    def close(self):
        """
        Clean up resources, including stopping keyboard listener.

        Returns:
            Result of closing the wrapped environment.
        """
        if hasattr(self, "listener") and self.listener is not None:
            self.listener.stop()
        return self.env.close()


class GearedLeaderControlWrapper(BaseLeaderControlWrapper):
    """
    Wrapper that enables manual intervention via keyboard.

    This wrapper extends the BaseLeaderControlWrapper to allow explicit toggling
    of human intervention mode with keyboard controls.
    """

    def _init_keyboard_events(self):
        """
        Initialize keyboard events including human intervention flag.

        Extends the base class dictionary with an additional flag for tracking
        intervention state toggled by keyboard.
        """
        super()._init_keyboard_events()
        self.keyboard_events["human_intervention_step"] = False

    def _handle_key_press(self, key, keyboard_device):
        """
        Handle key presses including space for intervention toggle.

        Args:
            key: The key that was pressed.
            keyboard: The keyboard module with key definitions.

        Extends the base handler to respond to space key for toggling intervention.
        """
        super()._handle_key_press(key, keyboard_device)
        if key == keyboard_device.Key.space:
            if not self.keyboard_events["human_intervention_step"]:
                logging.info(
                    "Space key pressed. Human intervention required.\n"
                    "Place the leader in similar pose to the follower and press space again."
                )
                self.keyboard_events["human_intervention_step"] = True
                log_say("Human intervention step.", play_sounds=True)
            else:
                self.keyboard_events["human_intervention_step"] = False
                logging.info("Space key pressed for a second time.\nContinuing with policy actions.")
                log_say("Continuing with policy actions.", play_sounds=True)

    def _check_intervention(self):
        """
        Check if human intervention is active based on keyboard toggle.

        Returns:
            Boolean indicating whether intervention mode is active.
        """
        return self.keyboard_events["human_intervention_step"]


class GearedLeaderAutomaticControlWrapper(BaseLeaderControlWrapper):
    """
    Wrapper with automatic intervention based on error thresholds.

    This wrapper monitors the error between leader and follower positions
    and automatically triggers intervention when error exceeds thresholds.
    """

    def __init__(
        self,
        env,
        teleop_device,
        end_effector_step_sizes,
        use_gripper=False,
        intervention_threshold=10.0,
        release_threshold=1e-2,
    ):
        """
        Initialize the automatic intervention wrapper.

        Args:
            env: The environment to wrap.
            teleop_device: The teleoperation device.
            use_gripper: Whether to include gripper control.
            intervention_threshold: Error threshold to trigger intervention.
            release_threshold: Error threshold to release intervention.
            queue_size: Number of error measurements to track for smoothing.
        """
        super().__init__(env, teleop_device, end_effector_step_sizes, use_gripper=use_gripper)

        # Error tracking parameters
        self.intervention_threshold = intervention_threshold  # Threshold to trigger intervention
        self.release_threshold = release_threshold  # Threshold to release intervention
        self.is_intervention_active = False
        self.start_time = time.perf_counter()

    def _check_intervention(self):
        """
        Determine if intervention should occur based on the rate of change of leader-follower error in end_effector space.

        This method monitors the rate of change of leader-follower error in end_effector space
        and automatically triggers intervention when the rate of change exceeds
        the intervention threshold, releasing when it falls below the release threshold.

        Returns:
            Boolean indicating whether intervention should be active.
        """

        # Condition for starting the intervention
        # If the error in teleoperation is too high, that means the a user has grasped the leader robot and he wants to take over
        if (
            not self.is_intervention_active
            and len(self.leader_tracking_error_queue) == self.leader_tracking_error_queue.maxlen
            and np.var(list(self.leader_tracking_error_queue)[-2:]) > self.intervention_threshold
        ):
            self.is_intervention_active = True
            self.leader_tracking_error_queue.clear()
            log_say("Intervention started", play_sounds=True)
            return True

        # Track the error over time in leader_tracking_error_queue
        # If the variance of the tracking error is too low, that means the user has let go of the leader robot and the intervention is over
        if (
            self.is_intervention_active
            and len(self.leader_tracking_error_queue) == self.leader_tracking_error_queue.maxlen
            and np.var(self.leader_tracking_error_queue) < self.release_threshold
        ):
            self.is_intervention_active = False
            self.leader_tracking_error_queue.clear()
            log_say("Intervention ended", play_sounds=True)
            return False

        # If not change has happened that merits a change in the intervention state, return the current state
        return self.is_intervention_active

    def reset(self, **kwargs):
        """
        Reset error tracking on environment reset.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset.

        Returns:
            The initial observation and info.
        """
        self.is_intervention_active = False
        return super().reset(**kwargs)


class GamepadControlWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling a gym environment with a gamepad.

    This wrapper intercepts the step method and allows human input via gamepad
    to override the agent's actions when desired.
    """

    def __init__(
        self,
        env,
        teleop_device,  # Accepts an instantiated teleoperator
        use_gripper=False,  # This should align with teleop_device's config
        auto_reset=False,
    ):
        """
        Initialize the gamepad controller wrapper.

        Args:
            env: The environment to wrap.
            teleop_device: The instantiated teleoperation device (e.g., GamepadTeleop).
            use_gripper: Whether to include gripper control (should match teleop_device.config.use_gripper).
            auto_reset: Whether to auto reset the environment when episode ends.
        """
        super().__init__(env)

        self.teleop_device = teleop_device
        # Ensure the teleop_device is connected if it has a connect method
        if hasattr(self.teleop_device, "connect") and not self.teleop_device.is_connected:
            self.teleop_device.connect()

        # self.controller attribute is removed

        self.auto_reset = auto_reset
        # use_gripper from args should ideally match teleop_device.config.use_gripper
        # For now, we use the one passed, but it can lead to inconsistency if not set correctly from config
        self.use_gripper = use_gripper

        logging.info("Gamepad control wrapper initialized with provided teleop_device.")
        print(
            "Gamepad controls (managed by the provided teleop_device - specific button mappings might vary):"
        )
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick: Move in Z axis (up/down)")
        print("  X/Square button: End episode (FAILURE)")
        print("  Y/Triangle button: End episode (SUCCESS)")
        print("  B/Circle button: Exit program")

    def get_teleop_commands(
        self,
    ) -> tuple[bool, np.ndarray, bool, bool, bool]:
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple containing:
            - is_active: Whether gamepad input is active (from teleop_device.gamepad.should_intervene())
            - action: The action derived from gamepad input (from teleop_device.get_action())
            - terminate_episode: Whether episode termination was requested
            - success: Whether episode success was signaled
            - rerecord_episode: Whether episode rerecording was requested
        """
        if not hasattr(self.teleop_device, "gamepad") or self.teleop_device.gamepad is None:
            raise AttributeError(
                "teleop_device does not have a 'gamepad' attribute or it is None. Expected for GamepadControlWrapper."
            )

        # Get status flags from the underlying gamepad controller within the teleop_device
        self.teleop_device.gamepad.update()  # Ensure gamepad state is fresh
        intervention_is_active = self.teleop_device.gamepad.should_intervene()
        episode_end_status = self.teleop_device.gamepad.get_episode_end_status()

        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        # Get the action dictionary from the teleop_device
        action_dict = self.teleop_device.get_action()

        # Convert action_dict to numpy array based on expected structure
        # Order: delta_x, delta_y, delta_z, gripper (if use_gripper)
        action_list = [action_dict["delta_x"], action_dict["delta_y"], action_dict["delta_z"]]
        if self.use_gripper:
            # GamepadTeleop returns gripper action as 0 (close), 1 (stay), 2 (open)
            # This needs to be consistent with what EEActionWrapper expects if it's used downstream
            # EEActionWrapper for gripper typically expects 0.0 (closed) to 2.0 (open)
            # For now, we pass the direct value from GamepadTeleop, ensure downstream compatibility.
            gripper_val = action_dict.get("gripper", 1.0)  # Default to 1.0 (stay) if not present
            action_list.append(float(gripper_val))

        gamepad_action_np = np.array(action_list, dtype=np.float32)

        return (
            intervention_is_active,
            gamepad_action_np,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using gamepad input to override actions when active.

        Args:
            action: Original action from agent.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Get gamepad state and action
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_teleop_commands()

        # Update episode ending state if requested
        if terminate_episode:
            logging.info(f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}")

        # Only override the action if gamepad is active
        action = gamepad_action if is_intervention else action

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        info["is_intervention"] = is_intervention
        # The original `BaseLeaderControlWrapper` puts `action_intervention` in info.
        # For Gamepad, if intervention, `gamepad_action` is the intervention.
        # If not intervention, policy's action is `action`.
        # For consistency, let's store the *human's* action if intervention occurred.
        info["action_intervention"] = torch.from_numpy(gamepad_action).to(action.device)

        info["rerecord_episode"] = rerecord_episode

        # If episode ended, reset the state
        if terminated or truncated:
            # Add success/failure information to info dict
            info["next.success"] = success

            # Auto reset if configured
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def close(self):
        """
        Clean up resources when environment closes.

        Returns:
            Result of closing the wrapped environment.
        """
        if hasattr(self.teleop_device, "disconnect"):
            self.teleop_device.disconnect()

        # Call the parent close method
        return self.env.close()


class KeyboardControlWrapper(GamepadControlWrapper):
    """
    Wrapper that allows controlling a gym environment with a keyboard.

    This wrapper intercepts the step method and allows human input via keyboard
    to override the agent's actions when desired.

    Inherits from GamepadControlWrapper to avoid code duplication.
    """

    def __init__(
        self,
        env,
        teleop_device,  # Accepts an instantiated teleoperator
        use_gripper=False,  # This should align with teleop_device's config
        auto_reset=False,
    ):
        """
        Initialize the gamepad controller wrapper.

        Args:
            env: The environment to wrap.
            teleop_device: The instantiated teleoperation device (e.g., GamepadTeleop).
            use_gripper: Whether to include gripper control (should match teleop_device.config.use_gripper).
            auto_reset: Whether to auto reset the environment when episode ends.
        """
        super().__init__(env, teleop_device, use_gripper, auto_reset)

        self.is_intervention_active = False

        logging.info("Keyboard control wrapper initialized with provided teleop_device.")
        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Right Ctrl and Left Ctrl: Open and close gripper")
        print("  f: End episode with FAILURE")
        print("  s: End episode with SUCCESS")
        print("  r: End episode with RERECORD")
        print("  i: Start/Stop Intervention")

    def get_teleop_commands(
        self,
    ) -> tuple[bool, np.ndarray, bool, bool, bool]:
        action_dict = self.teleop_device.get_action()
        episode_end_status = None

        # Unroll the misc_keys_queue to check for events related to intervention, episode success, etc.
        while not self.teleop_device.misc_keys_queue.empty():
            key = self.teleop_device.misc_keys_queue.get()
            if key == "i":
                self.is_intervention_active = not self.is_intervention_active
            elif key == "f":
                episode_end_status = "failure"
            elif key == "s":
                episode_end_status = "success"
            elif key == "r":
                episode_end_status = "rerecord_episode"

        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        # Convert action_dict to numpy array based on expected structure
        # Order: delta_x, delta_y, delta_z, gripper (if use_gripper)
        action_list = [action_dict["delta_x"], action_dict["delta_y"], action_dict["delta_z"]]
        if self.use_gripper:
            # GamepadTeleop returns gripper action as 0 (close), 1 (stay), 2 (open)
            # This needs to be consistent with what EEActionWrapper expects if it's used downstream
            # EEActionWrapper for gripper typically expects 0.0 (closed) to 2.0 (open)
            # For now, we pass the direct value from GamepadTeleop, ensure downstream compatibility.
            gripper_val = action_dict.get("gripper", 1.0)  # Default to 1.0 (stay) if not present
            action_list.append(float(gripper_val))

        gamepad_action_np = np.array(action_list, dtype=np.float32)

        return (
            self.is_intervention_active,
            gamepad_action_np,
            terminate_episode,
            success,
            rerecord_episode,
        )


class GymHilDeviceWrapper(gym.Wrapper):
    def __init__(self, env, device="cpu"):
        super().__init__(env)
        self.device = device

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in obs:
            obs[k] = obs[k].to(self.device)
        if "action_intervention" in info:
            # NOTE: This is a hack to ensure the action intervention is a float32 tensor and supported on MPS device
            info["action_intervention"] = info["action_intervention"].astype(np.float32)
            info["action_intervention"] = torch.from_numpy(info["action_intervention"]).to(self.device)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        for k in obs:
            obs[k] = obs[k].to(self.device)
        if "action_intervention" in info:
            # NOTE: This is a hack to ensure the action intervention is a float32 tensor and supported on MPS device
            info["action_intervention"] = info["action_intervention"].astype(np.float32)
            info["action_intervention"] = torch.from_numpy(info["action_intervention"]).to(self.device)
        return obs, info


class GymHilObservationProcessorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        prev_space = self.observation_space
        new_space = {}

        if isinstance(prev_space, gym.spaces.Dict):
            # Case for environments with dictionary-based observations (e.g., with cameras)
            if "pixels" in prev_space.spaces:
                for k in prev_space.spaces["pixels"].spaces:
                    # Use the actual image shape from the environment instead of assuming fixed size
                    original_shape = prev_space.spaces["pixels"].spaces[k].shape
                    # Convert from HWC to CHW format for LeRobot
                    if len(original_shape) == 3 and original_shape[-1] == 3:
                        # Assume HWC format (height, width, channels)
                        h, w, c = original_shape
                        lerobot_shape = (c, h, w)  # CHW format
                    else:
                        # Already in CHW format or different format
                        lerobot_shape = original_shape
                    
                    new_space[f"observation.images.{k}"] = gym.spaces.Box(
                        0.0, 255.0, shape=lerobot_shape, dtype=np.uint8
                    )
            if "agent_pos" in prev_space.spaces:
                new_space["observation.state"] = prev_space.spaces["agent_pos"]
            # Handle environment_state by combining it with agent_pos for state observation
            if "environment_state" in prev_space.spaces:
                # Combine agent_pos and environment_state into a single observation.state
                agent_pos_shape = prev_space.spaces["agent_pos"].shape[0]
                env_state_shape = prev_space.spaces["environment_state"].shape[0] 
                combined_shape = (agent_pos_shape + env_state_shape,)
                new_space["observation.state"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=combined_shape, dtype=np.float32
                )
        elif isinstance(prev_space, gym.spaces.Box):
            # Case for environments with box-based observations (e.g., state-only like RLCar)
            new_space["observation.state"] = prev_space
        else:
            raise TypeError(f"Unsupported observation space type: {type(prev_space)}")

        self.observation_space = gym.spaces.Dict(new_space)

    def observation(self, observation: Any) -> dict[str, Any]:
        # If the original observation is a numpy array (from a Box space),
        # wrap it in a dictionary to match what `preprocess_observation` expects.
        if isinstance(observation, np.ndarray):
            observation = {"agent_pos": observation}
        
        # If environment_state exists, combine it with agent_pos
        if "environment_state" in observation and "agent_pos" in observation:
            combined_state = np.concatenate([observation["agent_pos"], observation["environment_state"]], axis=-1)
            observation["agent_pos"] = combined_state
            # Remove environment_state since it's now included in agent_pos
            del observation["environment_state"]
            
        return preprocess_observation(observation)


###########################################################
# Factory functions
###########################################################


def make_robot_env(cfg: EnvConfig) -> gym.Env:
    """
    Factory function to create a robot environment.

    This function builds a robot environment with all necessary wrappers
    based on the provided configuration.

    Args:
        cfg: Configuration object containing environment parameters.

    Returns:
        A gym environment with all necessary wrappers applied.
    """
    if cfg.type == "hil":
        import gym_hil  # noqa: F401
    
        # Handle RLCar environment specifically, as it has different needs.
        if "RLCar" in cfg.task:
            env = gym.make(f"gym_hil/{cfg.task}")
            # Wrappers to make the environment compatible with the rest of the script.
            env = GymHilObservationProcessorWrapper(env=env)
            env = GymHilDeviceWrapper(env=env, device=cfg.device)
            env = BatchCompatibleWrapper(env=env)
            env = TorchActionWrapper(env=env, device=cfg.device)
            return env
    
        if "RLKuavo" in cfg.task:
            # Extract smoothing parameters from config if available
            vel_smoothing_factor = 0.00001  # default
            arm_smoothing_factor = 0.00001 # default    
            wbc_observation_enabled = True  # default
            action_dim = None  # default
            if cfg.wrapper:
                vel_smoothing_factor = getattr(cfg.wrapper, 'vel_smoothing_factor', 0.3)
                arm_smoothing_factor = getattr(cfg.wrapper, 'arm_smoothing_factor', 0.4)
                wbc_observation_enabled = getattr(cfg.wrapper, 'wbc_observation_enabled', True)
            # Extract action_dim from features if available
            if hasattr(cfg, 'features') and 'action' in cfg.features:
                action_dim = cfg.features['action'].shape[0]
            print(" ----- RLKuavo Parameters -----")
            print(f"vel_smoothing_factor: {vel_smoothing_factor}, arm_smoothing_factor: {arm_smoothing_factor}")
            print(f"wbc_observation_enabled: {wbc_observation_enabled}")
            print(f"action_dim: {action_dim}")
            print(" ----- RLKuavo Parameters -----")
            env = gym.make(f"gym_hil/{cfg.task}", 
                          vel_smoothing_factor=vel_smoothing_factor,
                          arm_smoothing_factor=arm_smoothing_factor,
                          wbc_observation_enabled=wbc_observation_enabled,
                          action_dim=action_dim)
            # First process observations to LeRobot format
            env = GymHilObservationProcessorWrapper(env=env)
            
            # Apply image cropping and resizing if configured
            if cfg.wrapper and (cfg.wrapper.crop_params_dict is not None or cfg.wrapper.resize_size is not None):
                # If only resize_size is provided but no crop_params_dict, create default crop params for all image keys
                crop_params_dict = cfg.wrapper.crop_params_dict
                if crop_params_dict is None and cfg.wrapper.resize_size is not None:
                    # Get all image observation keys and create default crop params (full image)
                    crop_params_dict = {}
                    for key in env.observation_space:
                        if "image" in key:
                            # Get the original image shape to create full-image crop params
                            img_shape = env.observation_space[key].shape
                            if len(img_shape) == 3:
                                h, w = img_shape[1], img_shape[2]  # CHW format: shape[0]=channels, shape[1]=height, shape[2]=width
                                crop_params_dict[key] = [0, 0, h, w]  # [top, left, height, width]
                
                if crop_params_dict:
                    env = ImageCropResizeWrapper(
                        env=env,
                        crop_params_dict=crop_params_dict,
                        resize_size=cfg.wrapper.resize_size,
                    )
            
            # Convert to device and make compatible with the rest of the script
            env = GymHilDeviceWrapper(env=env, device=cfg.device)
            env = BatchCompatibleWrapper(env=env)
            env = TorchActionWrapper(env=env, device=cfg.device)
            return env
    
        # TODO (azouitine)
        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=cfg.wrapper.use_gripper,
            gripper_penalty=cfg.wrapper.gripper_penalty,
        )
        env = GymHilObservationProcessorWrapper(env=env)
        env = GymHilDeviceWrapper(env=env, device=cfg.device)
        env = BatchCompatibleWrapper(env=env)
        env = TorchActionWrapper(env=env, device=cfg.device)
        return env

    if not hasattr(cfg, "robot") or not hasattr(cfg, "teleop"):
        raise ValueError(
            "Configuration for 'gym_manipulator' must be HILSerlRobotEnvConfig with robot and teleop."
        )

    if cfg.robot is None:
        raise ValueError("RobotConfig (cfg.robot) must be provided for gym_manipulator environment.")
    robot = make_robot_from_config(cfg.robot)

    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment
    env = RobotEnv(
        robot=robot,
        use_gripper=cfg.wrapper.use_gripper,
        display_cameras=cfg.wrapper.display_cameras if cfg.wrapper else False,
    )

    # Add observation and image processing
    if cfg.wrapper:
        if cfg.wrapper.add_joint_velocity_to_observation:
            env = AddJointVelocityToObservation(env=env, fps=cfg.fps)
        if cfg.wrapper.add_current_to_observation:
            env = AddCurrentToObservation(env=env)
        if cfg.wrapper.add_ee_pose_to_observation:
            env = EEObservationWrapper(env=env, ee_pose_limits=robot.end_effector_bounds)

    env = ConvertToLeRobotObservation(env=env, device=cfg.device)

    if cfg.wrapper and cfg.wrapper.crop_params_dict is not None:
        env = ImageCropResizeWrapper(
            env=env,
            crop_params_dict=cfg.wrapper.crop_params_dict,
            resize_size=cfg.wrapper.resize_size,
        )

    # Add reward computation and control wrappers
    reward_classifier = init_reward_classifier(cfg)
    if reward_classifier is not None:
        env = RewardWrapper(env=env, reward_classifier=reward_classifier, device=cfg.device)

    env = TimeLimitWrapper(env=env, control_time_s=cfg.wrapper.control_time_s, fps=cfg.fps)
    if cfg.wrapper.use_gripper and cfg.wrapper.gripper_penalty is not None:
        env = GripperPenaltyWrapper(
            env=env,
            penalty=cfg.wrapper.gripper_penalty,
        )

    # Control mode specific wrappers
    control_mode = cfg.wrapper.control_mode
    if control_mode == "gamepad":
        assert isinstance(teleop_device, GamepadTeleop), (
            "teleop_device must be an instance of GamepadTeleop for gamepad control mode"
        )
        env = GamepadControlWrapper(
            env=env,
            teleop_device=teleop_device,
            use_gripper=cfg.wrapper.use_gripper,
        )
    elif control_mode == "keyboard_ee":
        assert isinstance(teleop_device, KeyboardEndEffectorTeleop), (
            "teleop_device must be an instance of KeyboardEndEffectorTeleop for keyboard control mode"
        )
        env = KeyboardControlWrapper(
            env=env,
            teleop_device=teleop_device,
            use_gripper=cfg.wrapper.use_gripper,
        )
    elif control_mode == "leader":
        env = GearedLeaderControlWrapper(
            env=env,
            teleop_device=teleop_device,
            end_effector_step_sizes=cfg.robot.end_effector_step_sizes,
            use_gripper=cfg.wrapper.use_gripper,
        )
    elif control_mode == "leader_automatic":
        env = GearedLeaderAutomaticControlWrapper(
            env=env,
            teleop_device=teleop_device,
            end_effector_step_sizes=cfg.robot.end_effector_step_sizes,
            use_gripper=cfg.wrapper.use_gripper,
        )
    else:
        raise ValueError(f"Invalid control mode: {control_mode}")

    env = ResetWrapper(
        env=env,
        reset_pose=cfg.wrapper.fixed_reset_joint_positions,
        reset_time_s=cfg.wrapper.reset_time_s,
    )

    env = BatchCompatibleWrapper(env=env)
    env = TorchActionWrapper(env=env, device=cfg.device)

    return env


def init_reward_classifier(cfg):
    """
    Load a reward classifier policy from a pretrained path if configured.

    Args:
        cfg: The environment configuration containing classifier paths.

    Returns:
        The loaded classifier model or None if not configured.
    """
    if cfg.reward_classifier_pretrained_path is None:
        return None

    from lerobot.common.policies.sac.reward_model.modeling_classifier import Classifier

    # Get device from config or default to CUDA
    device = getattr(cfg, "device", "cpu")

    # Load the classifier directly using from_pretrained
    classifier = Classifier.from_pretrained(
        pretrained_name_or_path=cfg.reward_classifier_pretrained_path,
    )

    # Ensure model is on the correct device
    classifier.to(device)
    classifier.eval()  # Set to evaluation mode

    return classifier


###########################################################
# Record and replay functions
###########################################################


def record_dataset(env, policy, cfg):
    """
    Record a dataset of robot interactions using either a policy or teleop.

    This function runs episodes in the environment and records the observations,
    actions, and results for dataset creation.

    Args:
        env: The environment to record from.
        policy: Optional policy to generate actions (if None, uses teleop).
        cfg: Configuration object containing recording parameters like:
            - repo_id: Repository ID for dataset storage
            - dataset_root: Local root directory for dataset
            - num_episodes: Number of episodes to record
            - fps: Frames per second for recording
            - push_to_hub: Whether to push dataset to Hugging Face Hub
            - task: Name/description of the task being recorded
            - number_of_steps_after_success: Number of additional steps to continue recording after
                                  a success (reward=1) is detected. This helps collect
                                  more positive examples for reward classifier training.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Setup initial action (zero action if using teleop)
    action = env.action_space.sample() * 0.0

    if "RLCar" in cfg.task:
        action_names = ["left_wheel_velocity", "right_wheel_velocity"]
    else:
        action_names = ["delta_x_ee", "delta_y_ee", "delta_z_ee"]
        if cfg.wrapper.use_gripper:
            action_names.append("gripper_delta")

    # Configure dataset features based on environment spaces
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": env.observation_space["observation.state"].shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": env.action_space.shape,
            "names": action_names,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    if "RLCar" not in cfg.task:
        features["complementary_info.discrete_penalty"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        }

    # Add image features
    for key in env.observation_space:
        if "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": env.observation_space[key].shape,
                "names": ["channels", "height", "width"],
            }

    # Create dataset
    dataset = LeRobotDataset.create(
        cfg.repo_id,
        cfg.fps,
        root=cfg.dataset_root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    # Record episodes
    episode_index = 0
    recorded_action = None
    episode_rewards = []  # 记录每个回合的奖励
    episode_success_count = 0  # 记录真正成功的回合数
    while episode_index < cfg.num_episodes:
        obs, _ = env.reset()
        start_episode_t = time.perf_counter()
        log_say(f"Recording episode {episode_index}", play_sounds=True)

        # Track success state collection
        success_detected = False
        success_steps_collected = 0
        episode_total_reward = 0.0  # 追踪当前回合的总奖励

        # Run episode steps
        while time.perf_counter() - start_episode_t < cfg.wrapper.control_time_s:
            start_loop_t = time.perf_counter()

            # Get action from policy if available
            if cfg.pretrained_policy_name_or_path is not None:
                action = policy.select_action(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 累积当前回合的奖励
            episode_total_reward += reward

            # Check if episode needs to be rerecorded
            if info.get("rerecord_episode", False):
                break

            # For teleop, get action from intervention
            recorded_action = {
                "action": info["action_intervention"].cpu().squeeze(0).float() if policy is None else action
            }

            # Process observation for dataset
            obs_processed = {k: v.cpu().squeeze(0).float() for k, v in obs.items()}

            # Check if we've just detected success
            if reward == 1.0 and not success_detected:
                success_detected = True
                logging.info("Success detected! Collecting additional success states.")

            # Add frame to dataset - continue marking as success even during extra collection steps
            frame = {**obs_processed, **recorded_action}

            # If we're in the success collection phase, keep marking rewards as 1.0
            if success_detected:
                frame["next.reward"] = np.array([1.0], dtype=np.float32)
            else:
                frame["next.reward"] = np.array([reward], dtype=np.float32)

            # Only mark as done if we're truly done (reached end or collected enough success states)
            really_done = terminated or truncated
            if success_detected:
                success_steps_collected += 1
                really_done = success_steps_collected >= cfg.number_of_steps_after_success

            frame["next.done"] = np.array([really_done], dtype=bool)
            if "RLCar" not in cfg.task:
                frame["complementary_info.discrete_penalty"] = torch.tensor(
                    [info.get("discrete_penalty", 0.0)], dtype=torch.float32
                )
            dataset.add_frame(frame, task=cfg.task)

            # Maintain consistent timing
            if cfg.fps:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

            # Check if we should end the episode
            if (terminated or truncated) and not success_detected:
                # Regular termination without success
                break
            elif success_detected and success_steps_collected >= cfg.number_of_steps_after_success:
                # We've collected enough success states
                logging.info(f"Collected {success_steps_collected} additional success states")
                break

        # Handle episode recording
        if info.get("rerecord_episode", False):
            dataset.clear_episode_buffer()
            logging.info(f"Re-recording episode {episode_index}")
            continue

        dataset.save_episode()
        episode_rewards.append(episode_total_reward)  # 记录当前回合的总奖励
        
        # 判断该episode是否真正成功（基于合理的成功标准）
        # 对于RLKuavo任务，成功通常意味着较高的奖励（比如>200）
        if "RLKuavo" in cfg.task and episode_total_reward > 200:
            episode_success_count += 1
        elif reward == 1.0 or episode_total_reward > 50:  # 其他环境的成功标准
            episode_success_count += 1
            
        episode_index += 1

    # Finalize dataset
    # dataset.consolidate(run_compute_stats=True)
    if cfg.push_to_hub:
        dataset.push_to_hub()
    
    # 生成数据录制过程中的奖励统计分析报告
    if episode_rewards:
        print("\nGenerating recording process reward statistics analysis report...")
        print(f"Recording completed: {episode_success_count}/{len(episode_rewards)} episodes were truly successful")
        
        # 打印文字统计摘要
        print_rewards_summary(episode_rewards, success_count=episode_success_count, total_episodes=len(episode_rewards))
        
        # 生成可视化图表
        plot_save_path = f"recording_rewards_analysis_{cfg.task}_{len(episode_rewards)}eps.png"
        saved_plot_path = plot_episode_rewards(episode_rewards, save_path=plot_save_path, show_plot=False, 
                                             success_count=episode_success_count, total_episodes=len(episode_rewards))
        print(f"\nRecording process reward analysis chart saved to: {saved_plot_path}")
        print("Recording process reward statistics analysis completed!")
        print(f"CORRECT Recording Success Rate: {(episode_success_count/len(episode_rewards)*100):.1f}%")
    else:
        print("\nWARNING: No reward data collected during recording, skipping analysis")


def replay_episode(env, cfg):
    """
    Replay a recorded episode in the environment.

    This function loads actions from a previously recorded episode
    and executes them in the environment.

    Args:
        env: The environment to replay in.
        cfg: Configuration object containing replay parameters:
            - repo_id: Repository ID for dataset
            - dataset_root: Local root directory for dataset
            - episode: Episode ID to replay
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.dataset_root, episodes=[cfg.episode])
    env.reset()

    actions = dataset.hf_dataset.select_columns("action")
    
    # 记录回放过程中的奖励
    replay_rewards = []
    episode_total_reward = 0.0
    replay_success_count = 0

    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 累积奖励
        episode_total_reward += reward
        
        # 如果回合结束，记录总奖励并重置
        if terminated or truncated:
            replay_rewards.append(episode_total_reward)
            
            # 判断是否成功（使用与录制相同的标准）
            if "RLKuavo" in cfg.task and episode_total_reward > 200:
                replay_success_count += 1
            elif reward == 1.0 or episode_total_reward > 50:
                replay_success_count += 1
                
            episode_total_reward = 0.0
            env.reset()

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / 10 - dt_s)
    
    # 如果最后一个回合没有正常结束，也记录其奖励
    if episode_total_reward != 0.0:
        replay_rewards.append(episode_total_reward)
        # 判断最后一个episode是否成功
        if "RLKuavo" in cfg.task and episode_total_reward > 200:
            replay_success_count += 1
        elif episode_total_reward > 50:
            replay_success_count += 1
    
    # 生成回放过程的奖励统计分析报告
    if replay_rewards:
        print("\nGenerating replay process reward statistics analysis report...")
        print(f"Replay completed: {replay_success_count}/{len(replay_rewards)} episodes were truly successful")
        
        # 打印文字统计摘要
        print_rewards_summary(replay_rewards, success_count=replay_success_count, total_episodes=len(replay_rewards))
        
        # 生成可视化图表
        plot_save_path = f"replay_rewards_analysis_{cfg.task}_ep{cfg.episode}.png"
        saved_plot_path = plot_episode_rewards(replay_rewards, save_path=plot_save_path, show_plot=False,
                                             success_count=replay_success_count, total_episodes=len(replay_rewards))
        print(f"\nReplay process reward analysis chart saved to: {saved_plot_path}")
        print("Replay process reward statistics analysis completed!")
        print(f"CORRECT Replay Success Rate: {(replay_success_count/len(replay_rewards)*100):.1f}%")
    else:
        print("\nWARNING: No reward data collected during replay, skipping analysis")


@parser.wrap()
def main(cfg: EnvConfig):
    """Main entry point for the robot environment script.

    This function runs the robot environment in one of several modes
    based on the provided configuration.

    Args:
        cfg: Configuration object defining the run parameters,
             including mode (record, replay, random) and other settings.
    """
    env = make_robot_env(cfg)
    policy = None

    if cfg.mode == "record":
        policy = None
        if cfg.pretrained_policy_name_or_path is not None:
            from lerobot.common.policies.sac.modeling_sac import SACPolicy

            policy = SACPolicy.from_pretrained(cfg.pretrained_policy_name_or_path)
            policy.to(cfg.device)
            policy.eval()

        record_dataset(
            env,
            policy=policy,
            cfg=cfg,
        )
        exit()

    if cfg.mode == "replay":
        replay_episode(
            env,
            cfg=cfg,
        )
        exit()

    if cfg.mode == "eval":
        policy = None
        if cfg.pretrained_policy_name_or_path is not None:
            from lerobot.common.policies.sac.modeling_sac import SACPolicy
            print(f"cfg.pretrained_policy_name_or_path: {cfg.pretrained_policy_name_or_path}")
            policy = SACPolicy.from_pretrained(cfg.pretrained_policy_name_or_path)
            policy.to(cfg.device)
            policy.eval()

    obs, _ = env.reset()

    # Initialize the smoothed action as a random sample.
    action = env.action_space.sample() * 0.0

    # Smoothing coefficient (alpha) defines how much of the new random sample to mix in.
    # A value close to 0 makes the trajectory very smooth (slow to change), while a value close to 1 is less smooth.
    alpha = 1.0

    num_episode = 0
    successes = []
    episode_outcomes = []  # Track the reason for episode termination
    episode_total_reward = 0.0  # Track cumulative reward for current episode
    
    while num_episode < cfg.num_episodes:
        start_loop_s = time.perf_counter()
        if policy is not None:
            action = policy.select_action(obs)
        else:
            # Sample a new random action from the robot's action space.
            new_random_action = env.action_space.sample()
            # Update the smoothed action using an exponential moving average.
            action = alpha * new_random_action + (1 - alpha) * action

        # Execute the step: wrap the NumPy action in a torch tensor.
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate reward for current episode
        episode_total_reward += float(reward)
        
        if terminated or truncated:
            # Determine the episode outcome based on termination reason
            episode_outcome = "unknown"
            
            # Check for box fallen (failure) - highest priority failure
            if info.get("box_fallen", False):
                episode_outcome = "box_fallen"
                episode_reward = episode_total_reward  # Keep exploration value (already includes -50 penalty)
            # Check for timeout without success - failure
            elif truncated and not terminated:
                episode_outcome = "timeout"
                episode_reward = episode_total_reward  # Keep accumulated reward from exploration
            # Check for true success - only if no failures occurred
            elif terminated and not info.get("box_fallen", False):
                # Use reward threshold to determine success since environments may not set succeed flag correctly
                if episode_total_reward > 50:  # High reward indicates successful task completion
                    episode_outcome = "success"
                    episode_reward = episode_total_reward
                else:
                    episode_outcome = "low_reward_termination"  # Terminated but low reward = failure
                    episode_reward = episode_total_reward
            # Explicit success flag check as backup
            elif info.get("succeed", False) and not info.get("box_fallen", False):
                episode_outcome = "success"
                episode_reward = episode_total_reward
            # Other termination cases (failures)
            else:
                episode_outcome = "other_failure"
                episode_reward = episode_total_reward
            
            # Record the outcome-adjusted reward for this episode
            successes.append(episode_reward)
            episode_outcomes.append(episode_outcome)  # Track episode termination reason
            
            if cfg.num_episodes <= 20:  # Only print details for smaller test runs
                print(f"Episode {num_episode + 1} ended: {episode_outcome} (reward: {episode_reward:.2f}, steps: {info.get('episode_steps', 'unknown')})")
            
            obs, _ = env.reset()
            num_episode += 1
            episode_total_reward = 0.0  # Reset for next episode

        dt_s = time.perf_counter() - start_loop_s
        busy_wait(1 / cfg.fps - dt_s)

    print(f"Episode results over {cfg.num_episodes} episodes: {successes}")
    
    if len(successes) > 0:
        # Count different outcomes using episode_outcomes tracking
        success_count = episode_outcomes.count("success")
        box_fall_count = episode_outcomes.count("box_fallen")
        timeout_count = episode_outcomes.count("timeout")
        low_reward_count = episode_outcomes.count("low_reward_termination")
        other_failure_count = episode_outcomes.count("other_failure")
        unknown_count = episode_outcomes.count("unknown")
        
        # Calculate success rate (only true successes count)
        success_rate = success_count / len(successes)
        total_failures = box_fall_count + timeout_count + low_reward_count + other_failure_count + unknown_count
        
        print(f"\n=== CORRECTED Episode Analysis ===")
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   • Total Episodes: {cfg.num_episodes}")
        print(f"   • True Successes: {success_count} ({success_rate:.1%})")
        print(f"   • Total Failures: {total_failures} ({total_failures/cfg.num_episodes:.1%})")
        print(f"")
        print(f"📋 FAILURE BREAKDOWN:")
        print(f"   • Box Dropped: {box_fall_count} ({box_fall_count/cfg.num_episodes:.1%})")
        print(f"   • Timeout: {timeout_count} ({timeout_count/cfg.num_episodes:.1%})")
        print(f"   • Low Reward Termination: {low_reward_count} ({low_reward_count/cfg.num_episodes:.1%})")
        print(f"   • Other Failures: {other_failure_count} ({other_failure_count/cfg.num_episodes:.1%})")
        if unknown_count > 0:
            print(f"   • Unknown Outcomes: {unknown_count} ({unknown_count/cfg.num_episodes:.1%})")
        
        print(f"")
        print(f"💰 REWARD STATISTICS:")
        
        # Calculate rewards by outcome type
        success_rewards = [successes[i] for i, outcome in enumerate(episode_outcomes) if outcome == "success"]
        box_fall_rewards = [successes[i] for i, outcome in enumerate(episode_outcomes) if outcome == "box_fallen"]
        timeout_rewards = [successes[i] for i, outcome in enumerate(episode_outcomes) if outcome == "timeout"]
        low_reward_rewards = [successes[i] for i, outcome in enumerate(episode_outcomes) if outcome == "low_reward_termination"]
        
        overall_average = sum(successes) / len(successes)
        print(f"   • Overall Average Reward: {overall_average:.2f}")
        print(f"   • Max Reward: {max(successes):.2f}")
        print(f"   • Min Reward: {min(successes):.2f}")
        
        # Detailed breakdown by outcome
        if success_count > 0:
            avg_success_reward = sum(success_rewards) / len(success_rewards)
            print(f"   • Success Episodes Avg: {avg_success_reward:.1f}")
        
        if box_fall_count > 0:
            avg_box_fall_reward = sum(box_fall_rewards) / len(box_fall_rewards)
            print(f"   • Box Fall Episodes Avg: {avg_box_fall_reward:.1f} (exploration value - 50 penalty)")
        
        if timeout_count > 0:
            avg_timeout_reward = sum(timeout_rewards) / len(timeout_rewards)
            print(f"   • Timeout Episodes Avg: {avg_timeout_reward:.1f} (exploration value preserved)")
        
        if low_reward_count > 0:
            avg_low_reward = sum(low_reward_rewards) / len(low_reward_rewards)
            print(f"   • Low Reward Episodes Avg: {avg_low_reward:.1f}")
        
        print(f"")
        print(f"📝 NOTE: Box_fallen and timeout are failures, but rewards include exploration value for learning.")
        
        # Debug: Show episode outcomes for verification
        if cfg.num_episodes <= 20:  # Only for small test runs
            print(f"\n📋 Episode Outcomes Debug: {episode_outcomes}")
        
        success_rate_display = success_rate
    else:
        print("No episodes completed.")
        success_rate_display = 0.0
    
    # 生成奖励统计分析报告
    if successes:
        print("\nGenerating reward statistics analysis report...")
        print(f"CORRECTED Success Rate: {success_rate_display:.1%} (True successes only)")
        
        # 打印文字统计摘要
        print_rewards_summary(successes, success_count=success_count, total_episodes=cfg.num_episodes)
        
        # 生成可视化图表  
        plot_save_path = f"episode_rewards_analysis_{cfg.task}_{cfg.num_episodes}eps.png"
        saved_plot_path = plot_episode_rewards(successes, save_path=plot_save_path, show_plot=False, success_count=success_count, total_episodes=cfg.num_episodes)
        print(f"\nReward analysis chart saved to: {saved_plot_path}")
        print("Reward statistics analysis completed!")
        print(f"FINAL CORRECTED Success Rate: {success_rate_display:.1%}")
    else:
        print("\nWARNING: No reward data collected, skipping analysis")


if __name__ == "__main__":
    main()

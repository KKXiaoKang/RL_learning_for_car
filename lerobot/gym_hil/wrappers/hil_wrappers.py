#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import sys
import time
import threading

import gymnasium as gym
import numpy as np

from gym_hil.mujoco_gym_env import MAX_GRIPPER_COMMAND

DEFAULT_EE_STEP_SIZE = {"x": 0.025, "y": 0.025, "z": 0.025}


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        info["discrete_penalty"] = 0.0
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.1
        ):
            info["discrete_penalty"] = self.penalty

        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return observation, reward, terminated, truncated, info


class EEActionWrapper(gym.ActionWrapper):
    def __init__(self, env, ee_action_step_size, use_gripper=False):
        super().__init__(env)
        self.ee_action_step_size = ee_action_step_size
        self.use_gripper = use_gripper

        self._ee_step_size = np.array(
            [
                ee_action_step_size["x"],
                ee_action_step_size["y"],
                ee_action_step_size["z"],
            ]
        )
        num_actions = 3

        # Initialize action space bounds for the non-gripper case
        action_space_bounds_min = -np.ones(num_actions)
        action_space_bounds_max = np.ones(num_actions)

        if self.use_gripper:
            action_space_bounds_min = np.concatenate([action_space_bounds_min, [0.0]])
            action_space_bounds_max = np.concatenate([action_space_bounds_max, [2.0]])
            num_actions += 1

        ee_action_space = gym.spaces.Box(
            low=action_space_bounds_min,
            high=action_space_bounds_max,
            shape=(num_actions,),
            dtype=np.float32,
        )
        self.action_space = ee_action_space

    def action(self, action):
        """
        Mujoco env is expecting a 7D action space
        [x, y, z, rx, ry, rz, gripper_open]
        For the moment we only control the x, y, z, gripper
        """

        # action between -1 and 1, scale to step_size
        action_xyz = action[:3] * self._ee_step_size
        # TODO: Extend to enable orientation control
        actions_orn = np.zeros(3)

        gripper_open_command = [0.0]
        if self.use_gripper:
            # NOTE: Normalize gripper action from [0, 2] -> [-1, 1]
            gripper_open_command = [action[-1] - 1.0]

        action = np.concatenate([action_xyz, actions_orn, gripper_open_command])
        return action


class InputsControlWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling a gym environment with a gamepad.

    This wrapper intercepts the step method and allows human input via gamepad
    to override the agent's actions when desired.
    """

    def __init__(
        self,
        env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=False,
        auto_reset=False,
        input_threshold=0.001,
        use_gamepad=True,
        controller_config_path=None,
    ):
        """
        Initialize the inputs controller wrapper.

        Args:
            env: The environment to wrap
            x_step_size: Base movement step size for X axis in meters
            y_step_size: Base movement step size for Y axis in meters
            z_step_size: Base movement step size for Z axis in meters
            use_gripper: Whether to use gripper control
            auto_reset: Whether to auto reset the environment when episode ends
            input_threshold: Minimum movement delta to consider as active input
            use_gamepad: Whether to use gamepad or keyboard control
            controller_config_path: Path to the controller configuration JSON file
        """
        super().__init__(env)
        from gym_hil.wrappers.intervention_utils import (
            GamepadController,
            GamepadControllerHID,
            KeyboardController,
        )

        # use HidApi for macos
        if use_gamepad:
            if sys.platform == "darwin":
                self.controller = GamepadControllerHID(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                )
            else:
                self.controller = GamepadController(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                    config_path=controller_config_path,
                )
        else:
            self.controller = KeyboardController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )

        self.auto_reset = auto_reset
        self.use_gripper = use_gripper
        self.input_threshold = input_threshold
        self.controller.start()

    def get_gamepad_action(self):
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.controller.get_deltas()

        intervention_is_active = self.controller.should_intervene()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        if self.use_gripper:
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                gamepad_action = np.concatenate([gamepad_action, [2.0]])
            elif gripper_command == "close":
                gamepad_action = np.concatenate([gamepad_action, [0.0]])
            else:
                gamepad_action = np.concatenate([gamepad_action, [1.0]])

        # Check episode ending buttons
        # We'll rely on controller.get_episode_end_status() which returns "success", "failure", or None
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using gamepad input to override actions when active.

        cfg.
            action: Original action from agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get gamepad state and action
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Update episode ending state if requested
        if terminate_episode:
            logging.info(f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}")

        if is_intervention:
            action = gamepad_action

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        info["is_intervention"] = is_intervention
        action_intervention = action

        info["action_intervention"] = action_intervention
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

    def reset(self, **kwargs):
        """Reset the environment."""
        self.controller.reset()
        return self.env.reset(**kwargs)

    def close(self):
        """Clean up resources when environment closes."""
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # Call the parent close method
        return self.env.close()


class ResetDelayWrapper(gym.Wrapper):
    """
    Wrapper that adds a time delay when resetting the environment.

    This can be useful for adding a pause between episodes to allow for human observation.
    """

    def __init__(self, env, delay_seconds=1.0):
        """
        Initialize the time delay reset wrapper.

        Args:
            env: The environment to wrap
            delay_seconds: The number of seconds to delay during reset
        """
        super().__init__(env)
        self.delay_seconds = delay_seconds

    def reset(self, **kwargs):
        """Reset the environment with a time delay."""
        # Add the time delay
        logging.info(f"Reset delay of {self.delay_seconds} seconds")
        time.sleep(self.delay_seconds)

        # Call the parent reset method
        return self.env.reset(**kwargs)


class RLCarGamepadWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling the RLCar environment with a gamepad.
    When intervention is initiated, it first sends a zero action to stop the car,
    then gives control to the human.
    """

    def __init__(
        self,
        env,
        linear_vel_scale=1.0,
        angular_vel_scale=0.2,
        auto_reset=False,
        controller_config_path=None,
    ):
        """
        Initialize the RLCar gamepad controller wrapper.
        """
        super().__init__(env)
        from gym_hil.wrappers.intervention_utils import RLCarGamepadController

        self.controller = RLCarGamepadController(
            linear_vel_scale=linear_vel_scale,
            angular_vel_scale=angular_vel_scale,
            config_path=controller_config_path,
        )

        self.auto_reset = auto_reset
        self.controller.start()
        # State tracking for intervention
        self.was_intervening = False

    def get_gamepad_action(self):
        """
        Get the current action from the gamepad.
        """
        self.controller.update()

        intervention_is_active = self.controller.should_intervene()
        gamepad_action = self.controller.get_action()
        episode_end_status = self.controller.get_episode_end_status()

        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return intervention_is_active, gamepad_action, terminate_episode, success, rerecord_episode

    def step(self, action):
        """
        Step the environment, using gamepad input to override the policy's action when intervention is active.
        """
        (
            is_intervening_now,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Intervention logic with state tracking
        if is_intervening_now:
            if not self.was_intervening:
                # This is the first frame of intervention. Stop the car.
                print("Intervention started: Sending stop command.")
                action = np.array([0.0, 0.0], dtype=np.float32)
            else:
                # Already intervening, use gamepad joystick action.
                action = gamepad_action
        # If not intervening, the original `action` from the policy is used.

        # Update the state for the next step.
        self.was_intervening = is_intervening_now

        # Step the environment with the determined action
        obs, reward, terminated, truncated, info = self.env.step(action)

        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0

        info["is_intervention"] = is_intervening_now
        info["action_intervention"] = action
        info["rerecord_episode"] = rerecord_episode

        if terminated or truncated:
            info["next.success"] = success
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment."""
        self.controller.reset()
        self.was_intervening = False
        return self.env.reset(**kwargs)

    def close(self):
        """Clean up resources when environment closes."""
        if hasattr(self, "controller"):
            self.controller.stop()
        return self.env.close()

class RLKuavoGamepadWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling the RLKuavo environment with a gamepad.
    When intervention is initiated, it first sends a zero action to stop the robot,
    then gives control to the human.
    """

    def __init__(
        self,
        env,
        auto_reset=False,
        controller_config_path=None,
    ):
        """
        Initialize the RLKuavo gamepad controller wrapper.
        """
        super().__init__(env)
        from gym_hil.wrappers.intervention_utils import RLKuavoGamepadController

        self.controller = RLKuavoGamepadController(
            config_path=controller_config_path,
        )

        self.auto_reset = auto_reset
        self.controller.start()
        # State tracking for intervention
        self.was_intervening = False
        # The zero action for Kuavo.
        self.zero_action = np.zeros(self.env.action_space.shape, dtype=np.float32)

    def get_gamepad_action(self):
        """
        Get the current action from the gamepad.
        """
        self.controller.update()

        intervention_is_active = self.controller.should_intervene()
        gamepad_action = self.controller.get_action()
        episode_end_status = self.controller.get_episode_end_status()

        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return intervention_is_active, gamepad_action, terminate_episode, success, rerecord_episode

    def step(self, action):
        """
        Step the environment, using gamepad input to override the policy's action when intervention is active.
        """
        (
            is_intervening_now,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Intervention logic with state tracking
        if is_intervening_now:
            if not self.was_intervening:
                # This is the first frame of intervention. Stop the robot.
                print("Intervention started: Sending stop command.")
                action = self.zero_action
            else:
                # Already intervening, use gamepad joystick action.
                action = gamepad_action
        # If not intervening, the original `action` from the policy is used.

        # Update the state for the next step.
        self.was_intervening = is_intervening_now

        # Step the environment with the determined action
        obs, reward, terminated, truncated, info = self.env.step(action)

        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0

        info["is_intervention"] = is_intervening_now
        info["action_intervention"] = action
        info["rerecord_episode"] = rerecord_episode

        if terminated or truncated:
            info["next.success"] = success
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment."""
        self.controller.reset()
        self.was_intervening = False
        return self.env.reset(**kwargs)

    def close(self):
        """Clean up resources when environment closes."""
        if hasattr(self, "controller"):
            self.controller.stop()
        return self.env.close()

class RLKuavoMetaVRWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling the RLKuavo environment with Meta VR (Quest3) device.
    When intervention is initiated, it listens to VR-generated control commands from
    /mm_kuavo_arm_traj and /cmd_vel topics instead of letting the environment publish actions.
    """

    def __init__(
        self,
        env,
        auto_reset=False,
        intervention_threshold=1.0,
        rerecord_threshold=1.0,
        wbc_observation_enabled=True,
    ):
        """
        Initialize the RLKuavo Meta VR controller wrapper.
        
        Args:
            env: The environment to wrap
            auto_reset: Whether to auto reset the environment when episode ends
            intervention_threshold: Threshold for right_grip to trigger intervention
            rerecord_threshold: Threshold for left_grip to trigger rerecord
        """
        super().__init__(env)
        from gym_hil.wrappers.intervention_utils import Quest3Controller
        
        # Import ROS dependencies
        try:
            import rospy
            from geometry_msgs.msg import Twist
            from sensor_msgs.msg import JointState
            from kuavo_msgs.msg import twoArmHandPoseCmd
            from std_msgs.msg import Float64MultiArray, Bool
            self.rospy = rospy
            self.Twist = Twist
            self.JointState = JointState
            self.twoArmHandPoseCmd = twoArmHandPoseCmd
            self.Float64MultiArray = Float64MultiArray
            self.Bool = Bool
            self.ros_available = True
        except ImportError:
            print("Warning: ROS dependencies not available for VR intervention listening")
            self.ros_available = False

        self.controller = Quest3Controller(
            intervention_threshold=intervention_threshold,
            rerecord_threshold=rerecord_threshold,
        )

        self.auto_reset = auto_reset
        self.wbc_observation_enabled = wbc_observation_enabled
        self.controller.start()
        
        # State tracking for intervention
        self.was_intervening = False
        
        # The zero action for Kuavo
        self.zero_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        
        # VR intervention data
        self.latest_cmd_vel = None
        self.latest_arm_traj = None
        self.latest_ee_pose_cmd = None
        self.latest_left_action = None
        self.latest_right_action = None
        self.vr_action_lock = threading.Lock()
        
        # Episode tracking to prevent stale data usage
        self.current_episode_id = 0
        self.left_action_episode_id = -1
        self.right_action_episode_id = -1
        
        # Publisher to request trajectory restart
        self.trajectory_restart_pub = None
        
        # Setup ROS subscribers for VR-generated commands
        if self.ros_available:
            self._setup_vr_listeners()

    def _setup_vr_listeners(self):
        """Setup ROS subscribers to listen to VR-generated control commands."""
        try:
            # Subscribe to VR-generated control topics
            self.cmd_vel_sub = self.rospy.Subscriber(
                '/cmd_vel', self.Twist, self._cmd_vel_callback, queue_size=1
            )
            
            # Subscribe to Bézier trajectory action topics
            self.left_action_sub = self.rospy.Subscriber(
                '/sac/kuavo_eef_action_scale_left', self.Float64MultiArray, 
                self._left_action_callback, queue_size=1
            )
            self.right_action_sub = self.rospy.Subscriber(
                '/sac/kuavo_eef_action_scale_right', self.Float64MultiArray, 
                self._right_action_callback, queue_size=1
            )
            
            # Publisher to request trajectory restart
            self.trajectory_restart_pub = self.rospy.Publisher(
                '/sac/trajectory_restart_request', self.Bool, queue_size=1
            )
            
            print("VR intervention listeners setup successfully")
            print("Listening to topics:")
            print("  - /cmd_vel")
            print("  - /sac/kuavo_eef_action_scale_left")
            print("  - /sac/kuavo_eef_action_scale_right")
            print("Publishing to topics:")
            print("  - /sac/trajectory_restart_request")
        except Exception as e:
            print(f"Failed to setup VR listeners: {e}")
            self.ros_available = False

    def _cmd_vel_callback(self, msg):
        """Callback for VR-generated cmd_vel messages."""
        with self.vr_action_lock:
            self.latest_cmd_vel = msg

    def _left_action_callback(self, msg):
        """Callback for left hand action scale messages."""
        with self.vr_action_lock:
            # Convert Float64MultiArray to numpy array
            if len(msg.data) >= 3:  # Ensure we have at least x, y, z components
                self.latest_left_action = np.array(msg.data[:3], dtype=np.float32)
                self.left_action_episode_id = self.current_episode_id
                print(f"[VR DEBUG] Received left action for episode {self.current_episode_id}: {self.latest_left_action}")
            else:
                print(f"Warning: Left action data has insufficient components: {len(msg.data)}")

    def _right_action_callback(self, msg):
        """Callback for right hand action scale messages."""
        with self.vr_action_lock:
            # Convert Float64MultiArray to numpy array
            if len(msg.data) >= 3:  # Ensure we have at least x, y, z components
                self.latest_right_action = np.array(msg.data[:3], dtype=np.float32)
                self.right_action_episode_id = self.current_episode_id
                print(f"[VR DEBUG] Received right action for episode {self.current_episode_id}: {self.latest_right_action}")
            else:
                print(f"Warning: Right action data has insufficient components: {len(msg.data)}")

    def get_vr_action(self):
        """
        Get the current action from the VR controller if any input is active.

        Returns:
            Tuple containing:
            - is_active: Whether VR input is active
            - action: The action derived from VR input (numpy array)
            - terminate_episode: Whether episode termination was requested
            - success: Whether episode success was signaled
            - rerecord_episode: Whether episode rerecording was requested
        """
        # Update controller state
        self.controller.update()
        
        # Get intervention status
        is_intervention = self.controller.should_intervene()
        episode_end_status = self.controller.get_episode_end_status()

        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        # Get VR action from ROS topics
        vr_action = None
        with self.vr_action_lock:
            # Check if we have action data from Bézier trajectory topics for current episode
            has_left_action = (self.latest_left_action is not None and 
                             self.left_action_episode_id == self.current_episode_id)
            has_right_action = (self.latest_right_action is not None and 
                              self.right_action_episode_id == self.current_episode_id)
            
            # Only proceed if we're actually in intervention mode AND have valid CURRENT episode data
            if is_intervention and (has_left_action or has_right_action):
                # If we have both left and right actions, combine them into a single action
                if has_left_action and has_right_action:
                    # Combine left and right actions according to environment's action space
                    # Assuming the action space expects [left_x, left_y, left_z, right_x, right_y, right_z]
                    vr_action = np.concatenate([self.latest_left_action, self.latest_right_action])
                    
                elif has_left_action:
                    # Only left action available, pad with zeros for right hand
                    vr_action = np.concatenate([self.latest_left_action, np.zeros(3, dtype=np.float32)])
                    
                elif has_right_action:
                    # Only right action available, pad with zeros for left hand
                    vr_action = np.concatenate([np.zeros(3, dtype=np.float32), self.latest_right_action])
                
                # Ensure action matches the expected action space dimension
                if vr_action is not None:
                    expected_dim = self.env.action_space.shape[0]
                    if len(vr_action) != expected_dim:
                        # Adjust action size to match environment's action space
                        if len(vr_action) < expected_dim:
                            # Pad with zeros if action is too short
                            vr_action = np.pad(vr_action, (0, expected_dim - len(vr_action)), mode='constant')
                        else:
                            # Truncate if action is too long
                            vr_action = vr_action[:expected_dim]
                        
                    # Ensure the action is within the expected range [-1, 1]
                    vr_action = np.clip(vr_action, -1.0, 1.0)
                
                # Debug information
                if vr_action is not None:
                    print(f"VR Action - Left: {self.latest_left_action}, Right: {self.latest_right_action}")
                    print(f"Combined VR Action: {vr_action}")
            else:
                # Not in intervention mode or no valid action data available
                if is_intervention and not (has_left_action or has_right_action):
                    print(f"[VR DEBUG] Intervention active but no current episode action data available")
                    print(f"  Left: valid={self.latest_left_action is not None}, episode_id={self.left_action_episode_id}, current={self.current_episode_id}")
                    print(f"  Right: valid={self.latest_right_action is not None}, episode_id={self.right_action_episode_id}, current={self.current_episode_id}")

        return (
            is_intervention,
            vr_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using VR input to override the policy's action when intervention is active.
        """
        
        (
            is_intervening_now,
            vr_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_vr_action()

        # Intervention logic with state tracking
        if is_intervening_now:
            if not self.was_intervening:
                # This is the first frame of intervention. Set intervention mode in environment.
                # Set the intervention mode in the underlying environment
                if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'set_vr_intervention_mode'):
                    self.env.unwrapped.set_vr_intervention_mode(True)
                
                # Use the VR action directly
                if vr_action is not None:
                    action = vr_action
                    print("------------- Kuavo Meta VR is_intervening_now ------------ ") # 第一帧的时候打印
                else:
                    action = self.zero_action
            else:
                # Already intervening, use VR-generated action from the environment
                if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'get_vr_action'):
                    if vr_action is not None:
                        action = vr_action
                        print(" ------------ use wrapper vr action ---------------")
                    else:
                        action = self.zero_action
                else:
                    if vr_action is not None:
                        action = vr_action
                    else:
                        action = self.zero_action
        else:
            # Not intervening
            if self.was_intervening:
                # Clear the intervention mode in the underlying environment
                if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'set_vr_intervention_mode'):
                    self.env.unwrapped.set_vr_intervention_mode(False)
                
                # Clear VR action data when intervention stops to prevent stale data usage
                with self.vr_action_lock:
                    # Reset episode IDs to prevent using stale data
                    self.left_action_episode_id = -1
                    self.right_action_episode_id = -1
                print("[VR DEBUG] Intervention stopped - Invalidated VR action data")

        # Update the state for the next step.
        self.was_intervening = is_intervening_now

        # # 打印action
        # print(f"action: {action}")
    
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Set info for recording and monitoring
        info["is_intervention"] = is_intervening_now
        info["action_intervention"] = action
        info["rerecord_episode"] = rerecord_episode
        info["vr_grip_values"] = self.controller.get_grip_values()
        
        # Add VR data availability and action source info for debugging
        if is_intervening_now:
            with self.vr_action_lock:
                info["wbc_observation_enabled"] = self.wbc_observation_enabled
                info["action_source"] = "vr_intervention"
        else:
            info["action_source"] = "policy"

        # Handle episode termination
        if terminate_episode:
            terminated = True
            if success:
                reward = 1.0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment."""
        self.controller.reset()
        self.was_intervening = False
        
        # Increment episode ID to invalidate old action data
        self.current_episode_id += 1
        
        # Request trajectory restart from Bézier tool
        if self.ros_available and self.trajectory_restart_pub:
            restart_msg = self.Bool()
            restart_msg.data = True
            self.trajectory_restart_pub.publish(restart_msg)
            print(f"[VR DEBUG] Sent trajectory restart request for episode {self.current_episode_id}")
        
        # Clear VR intervention data
        with self.vr_action_lock:
            self.latest_cmd_vel = None
            self.latest_arm_traj = None
            self.latest_ee_pose_cmd = None
            # Keep the action data but mark it as invalid for current episode
            # This prevents using stale data from previous episodes
            
        # Reset intervention mode in the underlying environment
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'set_vr_intervention_mode'):
            self.env.unwrapped.set_vr_intervention_mode(False)
        
        print(f"[VR DEBUG] Reset - Episode ID incremented to {self.current_episode_id}")
        print(f"[VR DEBUG] Previous action episode IDs - Left: {self.left_action_episode_id}, Right: {self.right_action_episode_id}")
            
        return self.env.reset(**kwargs)

    def close(self):
        """Clean up resources when environment closes."""
        if hasattr(self, "controller"):
            self.controller.stop()
            
        # Clean up ROS subscribers
        if self.ros_available:
            if hasattr(self, 'cmd_vel_sub'):
                self.cmd_vel_sub.unregister()
            if hasattr(self, 'arm_traj_sub'):
                self.arm_traj_sub.unregister()
            if hasattr(self, 'ee_pose_cmd_sub'):
                self.ee_pose_cmd_sub.unregister()
            if hasattr(self, 'left_action_sub'):
                self.left_action_sub.unregister()
            if hasattr(self, 'right_action_sub'):
                self.right_action_sub.unregister()
            if hasattr(self, 'trajectory_restart_pub'):
                # Publishers don't need explicit unregistration, but we can set to None
                self.trajectory_restart_pub = None
                
        return self.env.close()

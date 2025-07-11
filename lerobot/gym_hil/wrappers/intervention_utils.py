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

import json
from pathlib import Path
import numpy as np


def load_controller_config(controller_name: str, config_path: str | None = None) -> dict:
    """
    Load controller configuration from a JSON file.

    Args:
        controller_name: Name of the controller to load.
        config_path: Path to the config file. If None, uses the package's default config.

    Returns:
        Dictionary containing the selected controller's configuration.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "controller_config.json"

    with open(config_path) as f:
        config = json.load(f)

    controller_config = config[controller_name] if controller_name in config else config["default"]

    if controller_name not in config:
        print(f"Controller {controller_name} not found in config. Using default configuration.")

    return controller_config


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def reset(self):
        """Reset the controller."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "no-op"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "success": False,
            "failure": False,
            "intervention": False,
            "rerecord": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = True
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = True
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = "success"
                elif key == keyboard.Key.esc:
                    self.key_states["failure"] = True
                    self.episode_end_status = "failure"
                elif key == keyboard.Key.space:
                    self.key_states["intervention"] = not self.key_states["intervention"]
                elif key == keyboard.Key.r:
                    self.key_states["rerecord"] = True
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = False
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Right Ctrl and Left Ctrl: Open and close gripper")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  Space: Start/Stop Intervention")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z

    def should_save(self):
        """Return True if Enter was pressed (save episode)."""
        return self.key_states["success"] or self.key_states["failure"]

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.key_states["intervention"]

    def reset(self):
        """Reset the controller."""
        for key in self.key_states:
            self.key_states[key] = False


class GamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, deadzone=0.1, config_path=None):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False
        self.config_path = config_path
        self.controller_config = None
        self.controller_type = None  # "xbox" or "bt2pro"

    def _print_controls(self):
        """Prints the control mapping for the user."""
        buttons = self.controller_config.get("buttons", {})
        if self.controller_type == "bt2pro":
            print("bt2pro Gamepad controls:")
            print("  Button 0: End episode with FAILURE")
            print("  Button 1: Exit")
            print("  Button 3: Rerecord episode")
            print("  Button 4: No function")
            print("  Button 6: End episode with SUCCESS")
            print("  Button 7: Intervention")
            print("  Button 8: Open gripper")
            print("  Button 9: Close gripper")
            print("  Left analog stick: Move in X-Y plane")
            print("  Right analog stick (vertical): Move in Z axis")
        else:
            print("Gamepad controls:")
            print(f"  {buttons.get('rb', 'RB')} button: Intervention")
            print("  Left analog stick: Move in X-Y plane")
            print("  Right analog stick (vertical): Move in Z axis")
            print(f"  {buttons.get('lt', 'LT')} button: Close gripper")
            print(f"  {buttons.get('rt', 'RT')} button: Open gripper")
            print(f"  {buttons.get('b', 'B')}/Circle button: Exit")
            print(f"  {buttons.get('y', 'Y')}/Triangle button: End episode with SUCCESS")
            print(f"  {buttons.get('a', 'A')}/Cross button: End episode with FAILURE")
            print(f"  {buttons.get('x', 'X')}/Square button: Rerecord episode")

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        joystick_name = self.joystick.get_name()
        print(f"Initialized gamepad: {joystick_name}")

        # Determine controller type based on joystick name
        if "bt2pro" in joystick_name.lower() or "beitong" in joystick_name.lower():
            self.controller_type = "bt2pro"
            print("Detected bt2pro controller - using custom button mappings")
        else:
            self.controller_type = "xbox"
            print("Detected Xbox controller - using default button mappings")

        # Load controller configuration based on joystick name
        self.controller_config = load_controller_config(joystick_name, self.config_path)

        # Print controls
        self._print_controls()

    def stop(self):
        """Clean up pygame resources."""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        import pygame

        if self.controller_type == "bt2pro":
            # bt2pro controller button mappings
            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 0:
                        self.episode_end_status = "failure"
                    elif event.button == 1:
                        self.episode_end_status = "exit"
                    elif event.button == 3:
                        self.episode_end_status = "rerecord_episode"
                    elif event.button == 6:
                        self.episode_end_status = "success"
                    elif event.button == 7:
                        self.intervention_flag = True
                    elif event.button == 8:
                        self.open_gripper_command = True
                    elif event.button == 9:
                        self.close_gripper_command = True

                # Reset episode status on button release
                elif event.type == pygame.JOYBUTTONUP:
                    if event.button in [0, 1, 3, 6]:
                        self.episode_end_status = None
                    elif event.button == 7:
                        self.intervention_flag = False
                    elif event.button == 8:
                        self.open_gripper_command = False
                    elif event.button == 9:
                        self.close_gripper_command = False
        else:
            # Original Xbox controller logic
            # Get button mappings from config
            buttons = self.controller_config.get("buttons", {})
            y_button = buttons.get("y", 3)  # Default to 3 if not found
            a_button = buttons.get("a", 0)  # Default to 0 if not found (Logitech F310)
            x_button = buttons.get("x", 2)  # Default to 2 if not found (Logitech F310)
            lt_button = buttons.get("lt", 6)  # Default to 6 if not found
            rt_button = buttons.get("rt", 7)  # Default to 7 if not found
            rb_button = buttons.get("rb", 5)  # Default to 5 if not found

            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == y_button:
                        self.episode_end_status = "success"
                    elif event.button == a_button:
                        self.episode_end_status = "failure"
                    elif event.button == x_button:
                        self.episode_end_status = "rerecord_episode"
                    elif event.button == lt_button:
                        self.close_gripper_command = True
                    elif event.button == rt_button:
                        self.open_gripper_command = True

                # Reset episode status on button release
                elif event.type == pygame.JOYBUTTONUP:
                    if event.button in [x_button, a_button, y_button]:
                        self.episode_end_status = None
                    elif event.button == lt_button:
                        self.close_gripper_command = False
                    elif event.button == rt_button:
                        self.open_gripper_command = False

                # Check for RB button for intervention flag
                if self.joystick.get_button(rb_button):
                    self.intervention_flag = True
                else:
                    self.intervention_flag = False

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        try:
            if self.controller_type == "bt2pro":
                # bt2pro controller axis handling - all axes need inversion
                # Get axis indices (bt2pro typically uses standard axis mapping)
                left_x_axis = 0  # Left/Right
                left_y_axis = 1  # Up/Down
                right_y_axis = 3  # Up/Down for Z

                # Read joystick axes
                x_input = self.joystick.get_axis(left_x_axis)  # Left/Right
                y_input = self.joystick.get_axis(left_y_axis)  # Up/Down
                z_input = self.joystick.get_axis(right_y_axis)  # Up/Down for Z

                # Apply deadzone to avoid drift
                x_input = 0 if abs(x_input) < self.deadzone else x_input
                y_input = 0 if abs(y_input) < self.deadzone else y_input
                z_input = 0 if abs(z_input) < self.deadzone else z_input

                # For bt2pro, all axes need inversion to match expected behavior
                x_input = -x_input  # Invert left/right
                y_input = -y_input  # Invert forward/backward
                z_input = -z_input  # Invert up/down

                # Calculate deltas
                delta_x = y_input * self.y_step_size  # Forward/backward
                delta_y = x_input * self.x_step_size  # Left/right
                delta_z = z_input * self.z_step_size  # Up/down

                return delta_x, delta_y, delta_z
            else:
                # Original Xbox controller logic
                # Get axis mappings from config
                axes = self.controller_config.get("axes", {})
                axis_inversion = self.controller_config.get("axis_inversion", {})

                # Get axis indices from config (with defaults if not found)
                left_x_axis = axes.get("left_x", 0)
                left_y_axis = axes.get("left_y", 1)
                right_y_axis = axes.get("right_y", 3)

                # Get axis inversion settings (with defaults if not found)
                invert_left_x = axis_inversion.get("left_x", False)
                invert_left_y = axis_inversion.get("left_y", True)
                invert_right_y = axis_inversion.get("right_y", True)

                # Read joystick axes
                x_input = self.joystick.get_axis(left_x_axis)  # Left/Right
                y_input = self.joystick.get_axis(left_y_axis)  # Up/Down
                z_input = self.joystick.get_axis(right_y_axis)  # Up/Down for Z

                # Apply deadzone to avoid drift
                x_input = 0 if abs(x_input) < self.deadzone else x_input
                y_input = 0 if abs(y_input) < self.deadzone else y_input
                z_input = 0 if abs(z_input) < self.deadzone else z_input

                # Apply inversion if configured
                if invert_left_x:
                    x_input = -x_input
                if invert_left_y:
                    y_input = -y_input
                if invert_right_y:
                    z_input = -z_input

                # Calculate deltas
                delta_x = y_input * self.y_step_size  # Forward/backward
                delta_y = x_input * self.x_step_size  # Left/right
                delta_z = z_input * self.z_step_size  # Up/down

                return delta_x, delta_y, delta_z

        except pygame.error:
            print("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states
        self.buttons = {}
        self.quit_requested = False
        self.save_requested = False

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        print("No gamepad found, check the connection and the product string in HID to add your gamepad")
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        import hid

        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            print(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            print(f"Connected to {manufacturer} {product}")

            print("Gamepad controls (HID mode):")
            print("  Left analog stick: Move in X-Y plane")
            print("  Right analog stick: Move in Z axis (vertical)")
            print("  Button 1/B/Circle: Exit")
            print("  Button 2/A/Cross: End episode with SUCCESS")
            print("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            print(f"Error opening gamepad: {e}")
            print("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            # Interpret gamepad data - this will vary by controller model
            # These offsets are for the Logitech RumblePad 2
            if data and len(data) >= 8:
                # Normalize joystick values from 0-255 to -1.0-1.0
                self.left_x = (data[1] - 128) / 128.0
                self.left_y = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # Apply deadzone
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                # Parse button states (byte 5 in the Logitech RumblePad 2)
                buttons = data[5]

                # Check if RB is pressed then the intervention flag should be set
                self.intervention_flag = data[6] in [2, 6, 10, 14]

                # Check if RT is pressed
                self.open_gripper_command = data[6] in [8, 10, 12]

                # Check if LT is pressed
                self.close_gripper_command = data[6] in [4, 6, 12]

                # Check if Y/Triangle button (bit 7) is pressed for saving
                # Check if X/Square button (bit 5) is pressed for failure
                # Check if A/Cross button (bit 4) is pressed for rerecording
                if buttons & 1 << 7:
                    self.episode_end_status = "success"
                elif buttons & 1 << 5:
                    self.episode_end_status = "failure"
                elif buttons & 1 << 4:
                    self.episode_end_status = "rerecord_episode"
                else:
                    self.episode_end_status = None

        except OSError as e:
            print(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_y * self.x_step_size  # Forward/backward
        delta_y = -self.left_x * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z

    def should_quit(self):
        """Return True if quit button was pressed."""
        return self.quit_requested

    def should_save(self):
        """Return True if save button was pressed."""
        return self.save_requested


class RLCarGamepadController(GamepadController):
    """Generate differential drive wheel velocities from gamepad input for RLCar."""

    def __init__(self, linear_vel_scale=1.0, angular_vel_scale=1.0, deadzone=0.1, config_path=None):
        # For a differential drive car:
        # y_step_size corresponds to linear velocity scale.
        # x_step_size corresponds to angular velocity scale.
        super().__init__(
            x_step_size=angular_vel_scale, y_step_size=linear_vel_scale, deadzone=deadzone, config_path=config_path
        )

    def _print_controls(self):
        """Override to print car-specific controls."""
        if self.controller_type == "bt2pro":
            print("RLCar Gamepad controls (bt2pro):")
            print("  Button 7 (RB): Press and hold for manual intervention")
            print("  Left stick (Up/Down): Forward/Backward")
            print("  Right stick (Left/Right): Turn Left/Right")
            print("  Button 6 (Y): End episode with SUCCESS")
            print("  Button 0 (A): End episode with FAILURE")
            print("  Button 3 (X): Rerecord episode")
        else:
            buttons = self.controller_config.get("buttons", {})
            print("RLCar Gamepad controls (Xbox-like):")
            print(f"  {buttons.get('rb', 'RB')} button: Press and hold for manual intervention")
            print("  Left stick (Up/Down): Forward/Backward")
            print("  Right stick (Left/Right): Turn Left/Right")
            print(f"  {buttons.get('y', 'Y')}/Triangle button: End episode with SUCCESS")
            print(f"  {buttons.get('a', 'A')}/Cross button: End episode with FAILURE")
            print(f"  {buttons.get('x', 'X')}/Square button: Rerecord episode")

    def get_action(self) -> np.ndarray:
        """Get the current wheel velocities from gamepad state using differential drive mixing."""
        import pygame

        try:
            steering_input, throttle_input = 0.0, 0.0
            # The axis mapping is adapted to be consistent with the user's `joy.py` example.
            # Throttle (forward/backward) is mapped to a vertical stick axis.
            # Steering (turn) is mapped to a horizontal stick axis.

            if self.controller_type == "bt2pro":
                # As per joy.py: throttle from axes[1], steering from axes[2]
                # For bt2pro on Linux, this typically corresponds to:
                # axes[1]: Left Stick Y
                # axes[2]: Right Stick X
                throttle_axis = 1
                steering_axis = 2

                throttle_input = self.joystick.get_axis(throttle_axis)
                steering_input = self.joystick.get_axis(steering_axis)

                # Invert throttle axis for intuitive control (up on stick = forward)
                throttle_input = -throttle_input

            else:
                # Default/Xbox controller logic from config
                axes = self.controller_config.get("axes", {})
                axis_inversion = self.controller_config.get("axis_inversion", {})

                # We will use Left Stick Y for throttle and Right Stick X for steering.
                # This is a common configuration and consistent with the logic requested.
                throttle_axis_name = "left_y"
                steering_axis_name = "right_x"

                # Get axis indices from config (with defaults if not found)
                throttle_axis_idx = axes.get(throttle_axis_name, 1)  # default to axis 1
                steering_axis_idx = axes.get(steering_axis_name, 2)  # default to axis 2

                throttle_input = self.joystick.get_axis(throttle_axis_idx)
                steering_input = self.joystick.get_axis(steering_axis_idx)

                # Get inversion from config, default to inverting Y axis
                invert_throttle = axis_inversion.get(throttle_axis_name, True)
                invert_steering = axis_inversion.get(steering_axis_name, False)

                if invert_throttle:
                    throttle_input = -throttle_input
                if invert_steering:
                    steering_input = -steering_input

            # Apply deadzone
            steering_input = 0 if abs(steering_input) < self.deadzone else steering_input
            throttle_input = 0 if abs(throttle_input) < self.deadzone else throttle_input

            # Calculate forward and turn components
            # self.y_step_size is linear_vel_scale
            # self.x_step_size is angular_vel_scale
            forward = throttle_input * self.y_step_size
            turn = steering_input * self.x_step_size

            # Differential drive mixing formula
            left_wheel_vel = forward + turn
            right_wheel_vel = forward - turn

            # Normalize to stay within a max velocity, similar to joy.py
            max_vel = self.y_step_size  # Assumes max linear velocity is the scale
            max_abs_speed = max(abs(left_wheel_vel), abs(right_wheel_vel))
            if max_abs_speed > max_vel:
                scale = max_vel / max_abs_speed
                left_wheel_vel *= scale
                right_wheel_vel *= scale

            return np.array([left_wheel_vel, right_wheel_vel], dtype=np.float32)

        except pygame.error:
            print("Error reading gamepad. Is it still connected?")
            return np.array([0.0, 0.0], dtype=np.float32)

    def get_deltas(self):
        # This controller returns a 2D action, not 3D deltas.
        # This method is here for compatibility with the base class.
        raise NotImplementedError("RLCarGamepadController uses get_action(), not get_deltas().")

class RLKuavoGamepadController(GamepadController):
    """
    Generate velocity commands from gamepad input for the Kuavo robot.
    This controller only affects the `cmd_vel` portion of the action space,
    leaving the arm joints at a neutral (zero) command during intervention.
    """

    def __init__(self, config_path=None, deadzone=0.1, enable_roll_pitch_control=False):
        """
        Initialize the Kuavo gamepad controller.

        Args:
            config_path: Path to the controller configuration JSON file.
            deadzone: Joystick deadzone to prevent drift.
            enable_roll_pitch_control: If True, enables control over angular x and y velocities.
        """
        # We don't use the step sizes from the base class, but it's good practice to call super.
        super().__init__(config_path=config_path, deadzone=deadzone)

        self.enable_roll_pitch_control = enable_roll_pitch_control

        if self.enable_roll_pitch_control:
            # Matches RLKuavoGymEnv vel_action_scale for 6D control
            self.vel_action_scale = np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.25])
            self.vel_dim = 6
        else:
            # Matches RLKuavoGymEnv vel_action_scale for 4D control
            self.vel_action_scale = np.array([0.5, 0.5, 0.5, 0.25])
            self.vel_dim = 4

        self.arm_dim = 14
        self.action_dim = self.vel_dim + self.arm_dim

    def _print_controls(self):
        """Override to print Kuavo-specific controls."""
        if self.controller_type == "bt2pro":
            print("\n--- RLKuavo Gamepad Controls (bt2pro) ---")
            print("  Button 7: Press and hold for manual intervention")
            print("  Left Stick (Up/Down): Forward/Backward (linear x)")
            print("  Left Stick (Left/Right): Strafe Left/Right (linear y)")
            print("  Right Stick (Up/Down): Move Up/Down (linear z)")
            print("  Right Stick (Left/Right): Turn Left/Right (angular z)")
            print("\n  Button 6: End episode with SUCCESS")
            print("  Button 0: End episode with FAILURE")
            print("  Button 3: Rerecord episode")
            print("---------------------------------\n")
        else:  # For xbox-like controllers
            buttons = self.controller_config.get("buttons", {})
            print("\n--- RLKuavo Gamepad Controls (Xbox-like) ---")
            print(f"  {buttons.get('rb', 'RB')} button: Press and hold for manual intervention")
            print("  Left Stick (Up/Down): Forward/Backward (linear x)")
            print("  Left Stick (Left/Right): Strafe Left/Right (linear y)")
            print("  Right Stick (Up/Down): Move Up/Down (linear z)")
            print("  Right Stick (Left/Right): Turn Left/Right (angular z)")
            if self.enable_roll_pitch_control:
                print(f"  {buttons.get('lt', 'LT')}/{buttons.get('rt', 'RT')} triggers: Roll (angular x)")
            print(f"\n  {buttons.get('y', 'Y')}/Triangle: End episode with SUCCESS")
            print(f"  {buttons.get('a', 'A')}/Cross: End episode with FAILURE")
            print(f"  {buttons.get('x', 'X')}/Square: Rerecord episode")
            print("---------------------------------\n")

    def get_action(self) -> np.ndarray:
        """Get the full action vector from the gamepad state."""
        import pygame

        try:
            axes_input = np.zeros(self.vel_dim)

            if self.controller_type == "bt2pro":
                # Using bt2pro specific mappings
                # [lx, ly, rx, ry, lt, rt]
                lx, ly = self.joystick.get_axis(0), -self.joystick.get_axis(1)
                rx, ry = self.joystick.get_axis(2), -self.joystick.get_axis(3)
                
                axes_input[0] = ly  # Linear X
                axes_input[1] = lx  # Linear Y
                axes_input[2] = ry  # Linear Z
                axes_input[3] = rx  # Angular Z
                if self.enable_roll_pitch_control:
                    # Placeholder for roll/pitch on bt2pro
                    pass

            else: # Xbox-like controller
                axes = self.controller_config.get("axes", {})
                axis_inversion = self.controller_config.get("axis_inversion", {})

                # Axis names for clarity
                ax_lu, ax_ld, ax_ru, ax_rd = "left_y", "left_x", "right_y", "right_x"
                
                # Get axis indices
                axis_indices = {name: axes.get(name, default_idx) for name, default_idx in 
                                [(ax_lu, 1), (ax_ld, 0), (ax_ru, 4), (ax_rd, 3)]} # Note: Xbox controller right stick axes are 3 and 4 in pygame
                
                # Read raw axis values
                raw_axes = {name: self.joystick.get_axis(idx) for name, idx in axis_indices.items()}

                # Apply inversion
                inversions = {name: axis_inversion.get(name, default) for name, default in
                              [(ax_lu, True), (ax_ld, False), (ax_ru, True), (ax_rd, False)]}

                processed_axes = {name: -val if inversions[name] else val for name, val in raw_axes.items()}

                # Map to velocity commands
                axes_input[0] = processed_axes[ax_lu] # Linear X
                axes_input[1] = processed_axes[ax_ld] # Linear Y
                axes_input[2] = processed_axes[ax_ru] # Linear Z
                axes_input[3] = processed_axes[ax_rd] # Angular Z
                
                if self.enable_roll_pitch_control:
                    # Using triggers for roll (angular x)
                    lt_axis, rt_axis = axes.get("lt", 2), axes.get("rt", 5)
                    lt_val = (self.joystick.get_axis(lt_axis) + 1) / 2 # Normalize to 0-1
                    rt_val = (self.joystick.get_axis(rt_axis) + 1) / 2 # Normalize to 0-1
                    axes_input[4] = rt_val - lt_val # angular.x (roll)
                    # axes_input[5] would be angular.y (pitch) - not mapped yet

            # Apply deadzone to all axes
            for i in range(len(axes_input)):
                if abs(axes_input[i]) < self.deadzone:
                    axes_input[i] = 0.0
            
            # Scale the velocity commands
            vel_action = axes_input * self.vel_action_scale

            # Create the full action vector with zero for arm joints
            full_action = np.zeros(self.action_dim, dtype=np.float32)
            full_action[:self.vel_dim] = vel_action
            
            return full_action

        except pygame.error:
            print("Error reading gamepad. Is it still connected?")
            return np.zeros(self.action_dim, dtype=np.float32)

    def get_deltas(self):
        """This controller returns a full action, not 3D deltas."""
        raise NotImplementedError("RLKuavoGamepadController uses get_action(), not get_deltas().")

class Quest3Controller(InputController):
    """Generate intervention signals from Quest3 VR device input."""

    def __init__(self, intervention_threshold=1.0, rerecord_threshold=1.0):
        """
        Initialize the Quest3 controller.

        Args:
            intervention_threshold: Threshold for right_grip to trigger intervention (default: 1.0)
            rerecord_threshold: Threshold for left_grip to trigger rerecord (default: 1.0)
        """
        super().__init__()
        self.intervention_threshold = intervention_threshold
        self.rerecord_threshold = rerecord_threshold
        self.subscriber = None
        self.latest_joystick_data = None
        self.ros_available = False
        
        # Try to import ROS dependencies
        try:
            import rospy
            from noitom_hi5_hand_udp_python.msg import JoySticks
            self.rospy = rospy
            self.JoySticks = JoySticks
            self.ros_available = True
        except ImportError as e:
            print(f"Warning: ROS dependencies not available for Quest3Controller: {e}")
            print("Please ensure ROS and noitom_hi5_hand_udp_python package are installed")

    def start(self):
        """Start the ROS subscriber for Quest3 joystick data."""
        if not self.ros_available:
            print("Cannot start Quest3Controller: ROS dependencies not available")
            self.running = False
            return

        def joystick_callback(msg):
            """Callback function for joystick data."""
            self.latest_joystick_data = msg
            
            # Check intervention condition based on right_grip
            if msg.right_grip >= self.intervention_threshold: # - 右摇杆下扳机键 - 干预
                self.intervention_flag = True
            else:
                self.intervention_flag = False
            
            # Check for rerecord condition based on left_grip - 左摇杆下扳机键 - 重录回合
            if msg.left_grip >= self.rerecord_threshold:
                self.episode_end_status = "rerecord_episode"
            
            # Optional: Check for episode end conditions using other buttons
            # You can extend this based on Quest3 button mappings
            elif hasattr(msg, 'left_second_button_pressed') and msg.left_second_button_pressed: # 左摇杆-Y键-成功
                self.episode_end_status = "success"
            elif hasattr(msg, 'left_first_button_pressed') and msg.left_first_button_pressed: # 左摇杆-X键-失败
                self.episode_end_status = "failure"

        try:
            self.subscriber = self.rospy.Subscriber(
                '/quest_joystick_data', 
                self.JoySticks, 
                joystick_callback,
                queue_size=1
            )
            print("Quest3 Controller started - monitoring /quest_joystick_data")
            print(f"  Right grip >= {self.intervention_threshold}: Enable intervention")
            print(f"  Left grip >= {self.rerecord_threshold}: Rerecord episode")
            print("  Left second button: End episode with SUCCESS")
            print("  Left first button: End episode with FAILURE")
        except Exception as e:
            print(f"Failed to start Quest3 subscriber: {e}")
            self.running = False

    def stop(self):
        """Stop the ROS subscriber."""
        if self.subscriber:
            self.subscriber.unregister()
            self.subscriber = None
        print("Quest3 Controller stopped")

    def should_intervene(self):
        """Return True if right grip exceeds threshold."""
        return self.intervention_flag

    def update(self):
        """Update is handled by ROS callback, no action needed."""
        # Check if we haven't received data recently (optional timeout check)
        if self.latest_joystick_data is None:
            # No data received yet, intervention_flag should remain False
            self.intervention_flag = False

    def get_deltas(self):
        """
        Quest3Controller doesn't provide motion deltas directly.
        This method returns zero deltas as it's primarily for intervention detection.
        """
        return 0.0, 0.0, 0.0

    def get_joystick_data(self):
        """
        Get the latest joystick data from Quest3.
        
        Returns:
            Latest JoySticks message or None if no data available
        """
        return self.latest_joystick_data

    def get_grip_values(self):
        """
        Get current grip values from both controllers.
        
        Returns:
            Tuple of (left_grip, right_grip) or (0.0, 0.0) if no data
        """
        if self.latest_joystick_data:
            return (self.latest_joystick_data.left_grip, self.latest_joystick_data.right_grip)
        return (0.0, 0.0)

    def reset(self):
        """Reset the controller state."""
        self.intervention_flag = False
        self.episode_end_status = None
        # Note: We don't reset latest_joystick_data as it represents current hardware state

import gymnasium as gym
import rospy
import threading
import numpy as np
from typing import Any


class IsaacLabGymEnv(gym.Env):
    """
    Abstract base Gymnasium environment for Isaac Lab robots, communicating via ROS.
    This class handles the low-level ROS communication setup and provides a
    thread-safe mechanism for handling observations. Subclasses must implement
    the specifics of ROS topic/service communication and task logic.
    """

    def __init__(self):
        super().__init__()

        # The script using this class should handle the main node initialization.
        if not rospy.core.is_initialized():
            rospy.init_node('isaac_lab_gym_env', anonymous=True)
            rospy.loginfo("gym.Env: ROS node initialized.")

        # Thread-safe observation buffer to be populated by subclass callbacks
        self.latest_obs = None
        self.obs_lock = threading.Lock()
        self.new_obs_event = threading.Event()

        # Subclasses must set up their own ROS publishers, subscribers, and services
        self._setup_ros_communication()

    def _setup_ros_communication(self):
        """
        Abstract method for subclasses to set up their specific ROS publishers,
        subscribers, and service clients.
        """
        raise NotImplementedError

    def _get_observation(self, timeout=2.0):
        """Waits for and retrieves the latest observation."""
        self.new_obs_event.clear()
        if self.new_obs_event.wait(timeout):
            with self.obs_lock:
                return self.latest_obs.copy()
        else:
            rospy.logerr("Timeout waiting for new observation from ROS topics.")
            raise TimeoutError("Did not receive a new observation.")

    def _send_action(self, action: np.ndarray):
        """
        Abstract method for subclasses to publish an action to the robot.
        """
        raise NotImplementedError

    def _reset_simulation(self):
        """
        Abstract method for subclasses to call the appropriate ROS service
        to reset the Isaac Lab simulation.
        """
        raise NotImplementedError

    def step(self, action):
        """
        The step method must be implemented by the subclass,
        which defines the specific task logic.
        """
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Resets the environment's random number generator.
        Subclasses should call this method via `super().reset(seed=seed)`
        to ensure proper seeding. The subclass is responsible for resetting
        the simulation and returning the initial observation.
        """
        super().reset(seed=seed)

    def render(self):
        """Rendering is handled by the Isaac Sim process, not the gym env."""
        pass

    def close(self):
        """Clean up ROS resources if necessary."""
        rospy.loginfo("Closing IsaacLabGymEnv.") 
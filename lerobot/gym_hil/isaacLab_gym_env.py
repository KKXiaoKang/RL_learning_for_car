import gymnasium as gym
import rospy
import message_filters
import threading
import time
import numpy as np
from typing import Any

from kuavo_msgs.msg import jointCmd
from geometry_msgs.msg import PoseStamped
from kuavo_msgs.srv import resetIsaaclab

class IsaacLabGymEnv(gym.Env):
    """
    Base Gymnasium environment for Isaac Lab robots, communicating via ROS.
    This class handles the low-level ROS communication, assuming the Isaac Sim
    simulation is running in a separate process.
    """

    def __init__(self):
        super().__init__()

        # The script using this class should handle the main node initialization.
        if not rospy.core.is_initialized():
            rospy.init_node('isaac_lab_gym_env', anonymous=True)
            rospy.loginfo("gym.Env: ROS node initialized.")

        # ROS Publishers and Service Clients
        self.cmd_pub = rospy.Publisher('/joint_cmd', jointCmd, queue_size=1)
        self.reset_client = rospy.ServiceProxy('/isaac_lab_reset_scene', resetIsaaclab)
        
        # ROS Subscribers with message_filters for synchronization
        robot_pose_sub = message_filters.Subscriber('/robot_pose', PoseStamped)
        goal_pose_sub = message_filters.Subscriber('/goal_pose', PoseStamped)
        goal_torso_pose_sub = message_filters.Subscriber('/goal_torso_pose', PoseStamped)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [robot_pose_sub, goal_pose_sub, goal_torso_pose_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self._obs_callback)
        
        # Thread-safe observation buffer
        self.latest_obs = None
        self.obs_lock = threading.Lock()
        self.new_obs_event = threading.Event()

    def _obs_callback(self, robot_pose, goal_pose, goal_torso_pose):
        """Synchronously handles incoming observation messages."""
        with self.obs_lock:
            robot_state = np.array([
                robot_pose.pose.position.x, robot_pose.pose.position.y, robot_pose.pose.position.z,
                robot_pose.pose.orientation.x, robot_pose.pose.orientation.y, 
                robot_pose.pose.orientation.z, robot_pose.pose.orientation.w
            ])
            
            goal_state = np.array([
                goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z,
                goal_pose.pose.orientation.x, goal_pose.pose.orientation.y, 
                goal_pose.pose.orientation.z, goal_pose.pose.orientation.w
            ])

            goal_torso_state = np.array([
                goal_torso_pose.pose.position.x, goal_torso_pose.pose.position.y, goal_torso_pose.pose.position.z,
                goal_torso_pose.pose.orientation.x, goal_torso_pose.pose.orientation.y, 
                goal_torso_pose.pose.orientation.z, goal_torso_pose.pose.orientation.w
            ])
            
            self.latest_obs = np.concatenate([robot_state, goal_state, goal_torso_state]).astype(np.float32)
            self.new_obs_event.set()

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
        """Publishes an action to the robot."""
        cmd = jointCmd()
        cmd.tau = action.tolist()
        self.cmd_pub.publish(cmd)

    def _reset_simulation(self):
        """Calls the ROS service to reset the Isaac Lab simulation."""
        try:
            rospy.wait_for_service('/isaac_lab_reset_scene', timeout=5.0)
            # The '0' indicates to use a new random seed within the simulation
            resp = self.reset_client(0) 
            if not resp.success:
                raise RuntimeError(f"Failed to reset simulation: {resp.message}")
            rospy.loginfo("Simulation reset successfully via ROS service.")
        except rospy.ServiceException as e:
            raise RuntimeError(f"Service call to reset simulation failed: {str(e)}")
        except rospy.ROSException as e:
             raise RuntimeError(f"Failed to connect to reset service: {str(e)}")

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
        to ensure proper seeding.
        """
        # This is the correct implementation. It calls the parent gym.Env's reset
        # to handle the random number generator and that's all it needs to do.
        super().reset(seed=seed)

    def render(self):
        """Rendering is handled by the Isaac Sim process, not the gym env."""
        pass

    def close(self):
        """Clean up ROS resources if necessary."""
        rospy.loginfo("Closing IsaacLabGymEnv.") 
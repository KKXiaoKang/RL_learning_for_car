from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces
import rospy
import message_filters
from kuavo_msgs.msg import jointCmd
from geometry_msgs.msg import PoseStamped
from kuavo_msgs.srv import resetIsaaclab


from gym_hil.isaacLab_gym_env import IsaacLabGymEnv


class RLCarGymEnv(IsaacLabGymEnv):
    """
    A gymnasium environment for the RL car task in Isaac Lab.
    This class defines the task-specific logic, including reward calculation,
    termination conditions, and observation/action spaces. It implements the
    ROS communication details for this specific task.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, debug: bool = False):
        # The base class __init__ will initialize the node and call 
        # _setup_ros_communication, so it must be called first.
        super().__init__()

        # Define observation and action spaces based on sac.py
        # The action space is normalized to [-1, 1] for the agent.
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(21,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Action scaling factor to map policy output to robot's physical command range
        self.action_scale = 20.0 
        
        # Task-specific state
        self.last_distance = float('inf')
        self.debug = debug
        
        # Environment boundaries from sac.py
        self.x_range = (-24.62, 4.5)
        self.y_range = (-17.32, 26.35)
        self.safety_margin = 0.5
        self.x_safe_range = (self.x_range[0] + self.safety_margin, self.x_range[1] - self.safety_margin)
        self.y_safe_range = (self.y_range[0] + self.safety_margin, self.y_range[1] - self.safety_margin)
        
        # Danger zone
        self.x_danger_range = (-10.46, 3.37)
        self.y_danger_range = (8.9, 24.9)
        
        # Goal radius
        self.reach_agent_radius = 1.0

        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _setup_ros_communication(self):
        """Sets up the ROS publishers, subscribers, and service clients for the RL Car task."""
        # Publishers
        self.cmd_pub = rospy.Publisher('/joint_cmd', jointCmd, queue_size=1)
        
        # Service Clients
        self.reset_client = rospy.ServiceProxy('/isaac_lab_reset_scene', resetIsaaclab)

        # Subscribers with message_filters for synchronization
        robot_pose_sub = message_filters.Subscriber('/robot_pose', PoseStamped)
        goal_pose_sub = message_filters.Subscriber('/goal_pose', PoseStamped)
        goal_torso_pose_sub = message_filters.Subscriber('/goal_torso_pose', PoseStamped)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [robot_pose_sub, goal_pose_sub, goal_torso_pose_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self._obs_callback)

    def _obs_callback(self, robot_pose, goal_pose, goal_torso_pose):
        """Synchronously handles incoming observation messages and populates the observation buffer."""
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        # Scale the normalized action from [-1, 1] to the robot's command range
        scaled_action = action * self.action_scale
        
        # Send the scaled action to the robot
        self._send_action(scaled_action)
        
        # Get the next observation
        obs = self._get_observation()
        
        # Compute reward and termination condition
        reward, done, info = self._compute_reward_and_done(obs, action)

        # The 'truncated' flag is False as we don't have a time limit in this implementation
        return obs, reward, done, False, info

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the state of the environment and returns an initial observation."""
        super().reset(seed=seed)
        
        # Reset the simulation via ROS service
        self._reset_simulation()
        
        # Get the initial observation after reset
        obs = self._get_observation()
        
        # Initialize the last distance to goal
        robot_pos = obs[0:2]
        goal_pos = obs[7:9]
        self.last_distance = np.linalg.norm(robot_pos - goal_pos)
        
        self.last_action.fill(0.0)
        
        if self.debug:
            print(f"Environment reset. Initial distance to goal: {self.last_distance:.3f}")
        
        return obs, {}

    def _compute_reward_and_done(self, obs: np.ndarray, action: np.ndarray) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculates the reward, done condition, and info dict for the current step.
        """
        reward = 0.0
        info = {}

        robot_pos = obs[0:2]
        goal_pos = obs[7:9]
        distance = np.linalg.norm(robot_pos - goal_pos)
        
        # Boundary collision check
        collided_boundary = not (self.x_safe_range[0] < robot_pos[0] < self.x_safe_range[1] and \
                                 self.y_safe_range[0] < robot_pos[1] < self.y_safe_range[1])

        # Danger zone check
        in_danger_zone = (self.x_danger_range[0] < robot_pos[0] < self.x_danger_range[1] and \
                          self.y_danger_range[0] < robot_pos[1] < self.y_danger_range[1])

        collided = collided_boundary or in_danger_zone
        if collided:
            reward -= 10.0  # Collision penalty

        # Distance-based reward (reward for getting closer)
        if self.last_distance != float('inf'):
            distance_change = self.last_distance - distance
            reward += 10.0 * distance_change

        # Goal reached reward
        reached_goal = distance < self.reach_agent_radius
        if reached_goal:
            reward += 300.0

        # Update last distance for the next step
        self.last_distance = distance
        
        # Action smoothness penalty
        action_smoothness_penalty = np.sum(np.square(action - self.last_action))
        reward -= 0.1 * action_smoothness_penalty  # The coefficient 0.1 can be tuned

        # Update last action
        self.last_action = action
        
        # Determine if the episode is done
        done = reached_goal or collided
        
        info["collided"] = collided
        info["distance_to_goal"] = distance
        info["reached_goal"] = reached_goal

        if self.debug:
            print(
                f"Dist: {distance:.2f} | "
                f"Rew: {reward:.2f} | "
                f"Done: {done} | "
                f"Info: {info}"
            )

        return reward, done, info


if __name__ == "__main__":
    import traceback
    
    print("Starting RLCarGymEnv test script...")

    # The environment itself handles ROS node initialization,
    # but it's good practice to have it here for a standalone script.
    if not rospy.core.is_initialized():
        rospy.init_node('rl_car_env_test', anonymous=True)

    # Instantiate the environment with debugging enabled
    env = RLCarGymEnv(debug=True)

    try:
        num_episodes = 5
        for i in range(num_episodes):
            print(f"\n--- Starting Episode {i + 1}/{num_episodes} ---")
            
            # Reset the environment
            obs, info = env.reset()
            print(f"Initial observation received. Shape: {obs.shape}")

            episode_reward = 0
            terminated = False
            truncated = False
            step_count = 0

            # Run the episode for a maximum of 100 steps
            while not (terminated or truncated) and step_count < 100:
                # Sample a random action from the normalized space [-1, 1]
                action = env.action_space.sample()
                print(f"\nStep {step_count + 1}: Sampled normalized action: {action}")
                
                # The env's step function will handle scaling.
                # Let's check the scaled action in the test for verification.
                scaled_action_for_print = action * env.action_scale
                print(f"Corresponding scaled action sent to robot: {scaled_action_for_print}")

                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Received observation. Shape: {obs.shape}")
                print(f"Reward: {reward:.4f}, Terminated: {terminated}, Info: {info}")
                
                episode_reward += reward
                step_count += 1
                
                # A small delay to allow ROS messages to be processed
                rospy.sleep(0.1)

            print(f"--- Episode {i + 1} Finished ---")
            print(f"Total steps: {step_count}, Total reward: {episode_reward:.4f}")

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        traceback.print_exc()
    finally:
        # Cleanly close the environment
        env.close()
        print("\nTest script finished.") 
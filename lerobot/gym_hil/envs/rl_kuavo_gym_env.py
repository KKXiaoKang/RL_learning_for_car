from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces
import rospy
import message_filters
import threading
import os
import xml.etree.ElementTree as ET
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from kuavo_msgs.msg import sensorsData
from ocs2_msgs.msg import mpc_observation
from kuavo_msgs.srv import resetIsaaclab

from gym_hil.isaacLab_gym_env import IsaacLabGymEnv


class RLKuavoGymEnv(IsaacLabGymEnv):
    """
    A gymnasium environment for the RL Kuavo robot task in Isaac Lab.
    This class will define the task-specific logic, including reward calculation,
    termination conditions, and observation/action spaces.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, debug: bool = True, image_size=(224, 224), enable_roll_pitch_control: bool = False):
        # Separate storage for headerless topics that will be initialized in callbacks.
        # This needs to be done BEFORE super().__init__() which sets up subscribers.
        self.latest_ang_vel = None
        self.latest_lin_accel = None
        self.latest_wbc = None
        self.ang_vel_lock = threading.Lock()
        self.lin_accel_lock = threading.Lock()
        self.wbc_lock = threading.Lock()
        
        self.enable_roll_pitch_control = enable_roll_pitch_control
        
        # VR intervention state and data - MUST be initialized before super().__init__()
        # because ROS subscribers will start receiving messages immediately
        self._is_vr_intervention_active = False
        self._latest_vr_cmd_vel = None
        self._latest_vr_arm_traj = None
        self._vr_action_lock = threading.Lock()
        self._should_publish_action = True
        self._is_first_step = True
        self._vr_intervention_mode = False
        self.latest_vr_cmd_vel = None
        self.latest_vr_arm_traj = None
        self.vr_cmd_vel_lock = threading.Lock()
        self.vr_arm_traj_lock = threading.Lock()

        # Call the base class constructor to set up the node and observation buffer
        super().__init__()
        self.debug = debug
        self.bridge = CvBridge()
        self.image_size = image_size

        # State observation dimension
        # agent_pos: 7 (left_eef) + 7 (right_eef) + 14 (arm_joints) + 3 (imu_ang_vel) + 3 (imu_lin_accel) + 12 (wbc) = 46
        # environment_state: 3 (box_pos) + 4 (box_orn) = 7
        agent_dim = 46
        env_state_dim = 7

        if self.enable_roll_pitch_control:
            vel_dim = 6
            self.vel_action_scale = np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.25])  # m/s and rad/s
        else:
            vel_dim = 4
            self.vel_action_scale = np.array([0.5, 0.5, 0.5, 0.25])  # m/s and rad/s
            
        action_dim = vel_dim + 14  # 14 for arm joints

        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (env_state_dim,), dtype=np.float32)

        # Define observation and action spaces for the Kuavo robot
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "front": spaces.Box(
                            0,
                            255,
                            (self.image_size[0], self.image_size[1], 3),
                            dtype=np.uint8,
                        )
                    }
                ),
                "agent_pos": agent_box,
                "environment_state": env_box,
            }
        )

        # Action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Define arm joint names in order
        self.arm_joint_names = [f'zarm_l{i}_joint' for i in range(1, 8)] + [f'zarm_r{i}_joint' for i in range(1, 8)]

        # Parse URDF for joint limits to define action scaling for arms
        urdf_path = os.path.join(os.path.dirname(__file__), '../assets/biped_s45.urdf')
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            joint_limits = {}
            for joint_name in self.arm_joint_names:
                joint_element = root.find(f".//joint[@name='{joint_name}']")
                if joint_element is not None:
                    limit_element = joint_element.find('limit')
                    if limit_element is not None:
                        lower = float(limit_element.get('lower'))
                        upper = float(limit_element.get('upper'))
                        joint_limits[joint_name] = {'lower': lower, 'upper': upper}
            
            lower_limits = np.array([joint_limits[name]['lower'] for name in self.arm_joint_names])
            upper_limits = np.array([joint_limits[name]['upper'] for name in self.arm_joint_names])

            self.arm_joint_centers = (upper_limits + lower_limits) / 2.0
            self.arm_joint_scales = (upper_limits - lower_limits) / 2.0

        except (ET.ParseError, FileNotFoundError, KeyError) as e:
            rospy.logerr(f"Failed to parse URDF or find all joint limits: {e}")
            # Fallback to a default scaling if URDF parsing fails
            self.arm_joint_centers = np.zeros(14)
            self.arm_joint_scales = np.full(14, np.deg2rad(10.0))

        # Task-specific state
        self.initial_box_pose = None
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        
        # Last converted VR action for recording
        self._last_vr_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _ang_vel_callback(self, msg):
        with self.ang_vel_lock:
            self.latest_ang_vel = msg

    def _lin_accel_callback(self, msg):
        with self.lin_accel_lock:
            self.latest_lin_accel = msg

    def _wbc_callback(self, msg):
        with self.wbc_lock:
            self.latest_wbc = msg

    def _vr_cmd_vel_callback(self, msg):
        """Callback for VR-generated cmd_vel messages."""
        # Only process VR messages when in VR intervention mode
        if not self._is_vr_intervention_active:
            return
            
        with self._vr_action_lock:
            self._latest_vr_cmd_vel = msg

    def _vr_arm_traj_callback(self, msg):
        """Callback for VR-generated arm trajectory messages."""
        # Only process VR messages when in VR intervention mode
        if not self._is_vr_intervention_active:
            return
            
        with self._vr_action_lock:
            self._latest_vr_arm_traj = msg

    def set_vr_intervention_mode(self, active: bool):
        """
        Set whether VR intervention mode is active.
        
        Args:
            active: True to enable VR intervention mode, False to disable
        """
        self._is_vr_intervention_active = active
        self._should_publish_action = not active  # Don't publish when VR is controlling
        
        # Clear VR data when disabling intervention mode
        if not active:
            with self._vr_action_lock:
                self._latest_vr_cmd_vel = None
                self._latest_vr_arm_traj = None
                print("[VR DEBUG] VR intervention mode disabled, cleared VR data")

    def get_vr_action(self) -> np.ndarray:
        """
        Convert VR-generated ROS messages to environment action format.
        
        Returns:
            Action array matching the environment's action space
        """
        
        with self._vr_action_lock:
            if not self._is_vr_intervention_active:
                return np.zeros(self.action_space.shape[0], dtype=np.float32)
            
            if self._latest_vr_cmd_vel is None and self._latest_vr_arm_traj is None:
                return np.zeros(self.action_space.shape[0], dtype=np.float32)
            
            # Create action array with correct dimensions
            action = np.zeros(self.action_space.shape[0], dtype=np.float32)
            
            # Process cmd_vel data if available
            if self._latest_vr_cmd_vel is not None:
                vel_cmd = self._latest_vr_cmd_vel
                vel_action = np.array([
                    vel_cmd.linear.x,
                    vel_cmd.linear.y, 
                    vel_cmd.linear.z,
                    vel_cmd.angular.z
                ], dtype=np.float32)
                
                # Scale the velocity commands
                if hasattr(self, 'vel_action_scale'):
                    vel_action = vel_action * self.vel_action_scale
                
                # Set velocity portion of action
                action[:4] = vel_action
            
            # Process arm trajectory data if available  
            if self._latest_vr_arm_traj is not None and len(self._latest_vr_arm_traj.position) >= 14:
                # JointState message has position array directly
                arm_positions_deg = np.array(self._latest_vr_arm_traj.position[:14], dtype=np.float32)
                
                # Convert degrees to radians if needed
                arm_positions_rad = np.deg2rad(arm_positions_deg)
                
                # Normalize to [-1, 1] using joint centers and scales
                arm_action = (arm_positions_rad - self.arm_joint_centers) / self.arm_joint_scales
                arm_action = np.clip(arm_action, -1.0, 1.0)
                
                action[4:18] = arm_action
            
            return action



    def _setup_ros_communication(self):
        """
        Implement this method to set up ROS publishers, subscribers,
        and service clients specific to the Kuavo robot.
        """
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.arm_traj_pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=1)

        # Service Client
        self.reset_client = rospy.ServiceProxy('/isaac_lab_reset_scene', resetIsaaclab)

        # Subscribers for headerless topics that are not synchronized
        rospy.Subscriber('/state_estimate/imu_data_filtered/angularVel', Float64MultiArray, self._ang_vel_callback)
        rospy.Subscriber('/state_estimate/imu_data_filtered/linearAccel', Float64MultiArray, self._lin_accel_callback)
        rospy.Subscriber('/humanoid_wbc_observation', mpc_observation, self._wbc_callback)
        
        # Subscribers for VR intervention commands - listen to VR-generated control commands
        rospy.Subscriber('/cmd_vel', Twist, self._vr_cmd_vel_callback)
        rospy.Subscriber('/mm_kuavo_arm_traj', JointState, self._vr_arm_traj_callback)

        # Synchronized subscribers
        eef_left_sub = message_filters.Subscriber('/fk/eef_pose_left', PoseStamped)
        eef_right_sub = message_filters.Subscriber('/fk/eef_pose_right', PoseStamped)
        image_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        sensors_sub = message_filters.Subscriber('/sensors_data_raw', sensorsData)
        box_real_sub = message_filters.Subscriber('/box_real_pose', PoseStamped)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [eef_left_sub, eef_right_sub, image_sub, sensors_sub, box_real_sub],
            queue_size=10,
            slop=0.1,  # Increased slop for the large number of topics
            allow_headerless=True,  # Allow synchronizing messages without a header
        )
        self.ts.registerCallback(self._obs_callback)

    def _obs_callback(self, left_eef, right_eef, image, sensors, box_real):
        """Synchronously handles incoming observation messages and populates the observation buffer."""
        # Retrieve the latest data from headerless topics
        with self.ang_vel_lock:
            ang_vel = self.latest_ang_vel
        with self.lin_accel_lock:
            lin_accel = self.latest_lin_accel
        with self.wbc_lock:
            wbc = self.latest_wbc

        # Wait until all data sources are available
        if ang_vel is None or lin_accel is None or wbc is None:
            if self.debug:
                rospy.logwarn_throttle(1.0, "IMU or WBC data not yet available for observation callback.")
            return

        with self.obs_lock:
            try:
                # Process image data
                cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
                # TODO: Resize image if necessary, for now assuming it's the correct size
                # cv_image = cv2.resize(cv_image, self.image_size)
                rgb_image = cv_image[:, :, ::-1].copy() # BGR to RGB

                # Process state data
                left_eef_data = np.array([
                    left_eef.pose.position.x, left_eef.pose.position.y, left_eef.pose.position.z,
                    left_eef.pose.orientation.x, left_eef.pose.orientation.y, left_eef.pose.orientation.z, left_eef.pose.orientation.w
                ])
                right_eef_data = np.array([
                    right_eef.pose.position.x, right_eef.pose.position.y, right_eef.pose.position.z,
                    right_eef.pose.orientation.x, right_eef.pose.orientation.y, right_eef.pose.orientation.z, right_eef.pose.orientation.w
                ])
                arm_data = np.array(sensors.joint_data.joint_q[12:26])
                ang_vel_data = np.array(ang_vel.data[:3])
                lin_accel_data = np.array(lin_accel.data[:3])
                wbc_data = np.array(wbc.state.value[:12])
                box_pos_data = np.array([
                    box_real.pose.position.x, box_real.pose.position.y, box_real.pose.position.z
                ])
                box_orn_data = np.array([
                    box_real.pose.orientation.x, box_real.pose.orientation.y,
                    box_real.pose.orientation.z, box_real.pose.orientation.w
                ])

                agent_pos_obs = np.concatenate([
                    left_eef_data, right_eef_data, arm_data, ang_vel_data,
                    lin_accel_data, wbc_data
                ]).astype(np.float32)

                env_state_obs = np.concatenate([
                    box_pos_data, box_orn_data
                ]).astype(np.float32)

                self.latest_obs = {
                    "pixels": {"front": rgb_image},
                    "agent_pos": agent_pos_obs,
                    "environment_state": env_state_obs,
                }
                self.new_obs_event.set()

            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")

    def _send_action(self, action: np.ndarray):
        """
        Implement this method to publish an action to the Kuavo robot.
        
        Args:
            action: The action array to send
        """
        if not self._should_publish_action:
            # During VR intervention, don't publish actions - VR system handles control
            return
            
        # De-normalize and publish cmd_vel
        twist_cmd = Twist()
        
        if self.enable_roll_pitch_control:
            vel_action = action[:6] * self.vel_action_scale
            twist_cmd.linear.x = vel_action[0]
            twist_cmd.linear.y = vel_action[1]
            twist_cmd.linear.z = vel_action[2]
            twist_cmd.angular.x = vel_action[3]
            twist_cmd.angular.y = vel_action[4]
            twist_cmd.angular.z = vel_action[5]
            arm_action = action[6:]
        else:
            vel_action = action[:4] * self.vel_action_scale
            twist_cmd.linear.x = vel_action[0]
            twist_cmd.linear.y = vel_action[1]
            twist_cmd.linear.z = vel_action[2]
            twist_cmd.angular.x = 0.0
            twist_cmd.angular.y = 0.0
            twist_cmd.angular.z = vel_action[3]
            arm_action = action[4:]

        self.cmd_vel_pub.publish(twist_cmd)

        # De-normalize and publish arm trajectory
        joint_cmd = JointState()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.name = self.arm_joint_names
        
        # Scale action from [-1, 1] to the full joint range in radians
        joint_action_rad = self.arm_joint_centers + arm_action * self.arm_joint_scales
        
        # Convert radians to degrees before publishing
        joint_action_deg = np.rad2deg(joint_action_rad)
        
        joint_cmd.position = joint_action_deg.tolist()
        self.arm_traj_pub.publish(joint_cmd)

    def _reset_simulation(self):
        """
        Implement this method to call the reset service for the Kuavo simulation.
        """
        try:
            rospy.wait_for_service('/isaac_lab_reset_scene', timeout=5.0)
            resp = self.reset_client(0) # 0 for random seed in sim | 在这里等待服务端处理完成并且返回结果
            if not resp.success:
                raise RuntimeError(f"Failed to reset simulation: {resp.message}")
            if self.debug:
                rospy.loginfo("Simulation reset successfully via ROS service.")
        except (rospy.ServiceException, rospy.ROSException) as e:
            raise RuntimeError(f"Service call to reset simulation failed: {str(e)}")

    def _compute_reward_and_done(self, obs: Dict[str, np.ndarray]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculates the reward, done condition, and info dict for the current step.
        The reward is sparse, returning 1.0 for success and 0.0 otherwise.

        Success is defined by three simultaneous conditions:
        1.  The box is lifted more than 20cm vertically from its initial position.
        2.  The box's orientation remains stable (within ~11 degrees of initial orientation).
        3.  Both of the robot's hands are within 50cm of the box's center.
        """
        info = {}
        
        # Extract data from observation
        agent_state = obs['agent_pos']
        env_state = obs['environment_state']
        left_eef_pos = agent_state[0:3]
        right_eef_pos = agent_state[7:10]
        box_pos = env_state[0:3]
        box_orn = env_state[3:7]

        # Calculate distances
        dist_left_hand_to_box = np.linalg.norm(left_eef_pos - box_pos)
        dist_right_hand_to_box = np.linalg.norm(right_eef_pos - box_pos)

        # Check conditions for success
        z_lift = box_pos[2] - self.initial_box_pose['position'][2]
        
        # Quaternion dot product to check orientation deviation
        q1 = box_orn
        q2 = self.initial_box_pose['orientation']
        orientation_similarity = abs(np.dot(q1, q2))

        # Success condition
        lift_success = z_lift > 0.10 # 10cm视作成功
        orientation_success = orientation_similarity > 0.98 # within ~11 degrees
        hands_close_success = dist_left_hand_to_box < 0.5 and dist_right_hand_to_box < 0.5
        
        # reached_goal = lift_success and orientation_success and hands_close_success
        reached_goal = lift_success and hands_close_success

        reward = 1.0 if reached_goal else 0.0
        done = reached_goal

        info["succeed"] = reached_goal
        info["z_lift"] = z_lift
        info["orientation_similarity"] = orientation_similarity
        info["dist_left_hand_to_box"] = dist_left_hand_to_box
        info["dist_right_hand_to_box"] = dist_right_hand_to_box

        if self.debug:
            # print(f"z_lift: {z_lift:.3f}, orient_sim: {orientation_similarity:.3f}, success: {reached_goal}")
            print(f"z_lift: {z_lift:.3f}, success: {reached_goal}")
            
        return reward, done, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Implement the step logic for the Kuavo robot.
        
        Args:
            action: The action to execute
        """
        self._send_action(action)
        obs = self._get_observation()
        
        # If this is the first step after reset, recalibrate initial position
        if self._is_first_step:
            if self.debug:
                old_box_pos = self.initial_box_pose['position']
                new_box_pos = obs['environment_state'][0:3]
                rospy.loginfo(f"First step - Recalibrating initial box position")
                rospy.loginfo(f"  Old initial position: {old_box_pos}")
                rospy.loginfo(f"  New initial position: {new_box_pos}")
                rospy.loginfo(f"  Position drift: {new_box_pos - old_box_pos}")
            
            # Update initial box pose with current observation
            self.initial_box_pose['position'] = obs['environment_state'][0:3]
            self.initial_box_pose['orientation'] = obs['environment_state'][3:7]
            self._is_first_step = False
        
        reward, done, info = self._compute_reward_and_done(obs)
        self.last_action = action
        return obs, reward, done, False, info

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Implement the reset logic for the Kuavo robot. This should include
        calling `super().reset(seed=seed)`, `self._reset_simulation()`, and returning
        the first observation.
        """
        super().reset(seed=seed)
        self._reset_simulation()
        
        # Wait for simulation to stabilize after reset
        import time
        time.sleep(0.5)  # 等待500ms让仿真稳定
        
        # Get initial observation to establish baseline
        obs = self._get_observation()
        
        # Wait a bit more and get another observation to ensure stability
        time.sleep(0.2)  # 再等待200ms
        obs_stable = self._get_observation()

        # Store initial box pose for reward calculation (use the stable observation)
        box_pos = obs_stable['environment_state'][0:3]
        box_orn = obs_stable['environment_state'][3:7]
        self.initial_box_pose = {'position': box_pos, 'orientation': box_orn}
        
        if self.debug:
            rospy.loginfo(f"reset - Initial box position (first): {obs['environment_state'][0:3]}")
            rospy.loginfo(f"reset - Initial box position (stable): {box_pos}")
            rospy.loginfo(f"reset - Position difference: {box_pos - obs['environment_state'][0:3]}")
        
        self.last_action.fill(0.0)
        # Reset the first step flag
        self._is_first_step = True

        return obs_stable, {}


if __name__ == "__main__":
    import traceback
    
    print("Starting RLKuavoGymEnv test script...")

    # The environment itself handles ROS node initialization,
    # but it's good practice to have it here for a standalone script.
    if not rospy.core.is_initialized():
        rospy.init_node('rl_kuavo_env_test', anonymous=True)

    # Instantiate the environment with debugging enabled
    env = RLKuavoGymEnv(debug=True, enable_roll_pitch_control=False)

    try:
        num_episodes = 3
        for i in range(num_episodes):
            print(f"\n--- Starting Episode {i + 1}/{num_episodes} ---")
            
            # Reset the environment
            obs, info = env.reset()
            print(f"Initial observation received.")
            print(f"  Agent_pos shape: {obs['agent_pos'].shape}")
            print(f"  Environment_state shape: {obs['environment_state'].shape}")
            print(f"  Pixels shape: {obs['pixels']['front'].shape}")

            episode_reward = 0
            terminated = False
            step_count = 0
            max_steps = 100

            # Run the episode
            while not terminated and step_count < max_steps:
                # Sample a random action from the normalized space [-1, 1]
                action = env.action_space.sample()
                print(f"\nStep {step_count + 1}/{max_steps}: Sampled normalized action shape: {action.shape}")
                
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Received observation:")
                print(f"  Agent_pos shape: {obs['agent_pos'].shape}")
                print(f"  Environment_state shape: {obs['environment_state'].shape}")
                print(f"  Pixels shape: {obs['pixels']['front'].shape}")
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
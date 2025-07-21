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
# Add the following imports for the new message types
from kuavo_msgs.msg import twoArmHandPoseCmd, twoArmHandPose, armHandPose, ikSolveParam
from kuavo_msgs.srv import changeTorsoCtrlMode, changeTorsoCtrlModeRequest, changeArmCtrlMode, changeArmCtrlModeRequest
from enum import Enum
from gym_hil.isaacLab_gym_env import IsaacLabGymEnv


class IncrementalMpcCtrlMode(Enum):
    """表示Kuavo机器人 Manipulation MPC 控制模式的枚举类"""
    NoControl = 0
    """无控制"""
    ArmOnly = 1
    """仅控制手臂"""
    BaseOnly = 2
    """仅控制底座"""
    BaseArm = 3
    """同时控制底座和手臂"""
    ERROR = -1
    """错误状态"""

class RLKuavoGymEnv(IsaacLabGymEnv):
    """
    A gymnasium environment for the RL Kuavo robot task in Isaac Lab.
    This class will define the task-specific logic, including reward calculation,
    termination conditions, and observation/action spaces.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, debug: bool = True, image_size=(224, 224), enable_roll_pitch_control: bool = False, 
                 vel_smoothing_factor: float = 0.3, arm_smoothing_factor: float = 0.4, 
                 wbc_observation_enabled: bool = True, action_dim: int = None):
        # Separate storage for headerless topics that will be initialized in callbacks.
        # This needs to be done BEFORE super().__init__() which sets up subscribers.
        self.latest_ang_vel = None
        self.latest_lin_accel = None
        self.latest_wbc = None
        self.latest_robot_pose = None  # New: for robot pose when WBC is disabled
        self.ang_vel_lock = threading.Lock()
        self.lin_accel_lock = threading.Lock()
        self.wbc_lock = threading.Lock()
        self.robot_pose_lock = threading.Lock()  # New: lock for robot pose
        
        self.enable_roll_pitch_control = enable_roll_pitch_control
        self.wbc_observation_enabled = wbc_observation_enabled  # New: WBC observation flag
        self.debug = debug  # Set debug before super().__init__() to avoid AttributeError
        
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
        self.bridge = CvBridge()
        self.image_size = image_size

        # State observation dimension - depends on WBC observation mode
        if self.wbc_observation_enabled:
            # agent_pos: 7 (left_eef) + 7 (right_eef) + 14 (arm_joints) + 3 (imu_ang_vel) + 3 (imu_lin_accel) + 12 (wbc) = 46
            # environment_state: 3 (box_pos) + 4 (box_orn) = 7
            agent_dim = 46
            env_state_dim = 7
        else:
            # agent_pos: 3 (left_eef_pos) + 3 (right_eef_pos) + 14 (arm_joints) + 3 (robot_pos) = 23
            # environment_state: 3 (box_pos) = 3
            agent_dim = 23
            env_state_dim = 3

        if self.enable_roll_pitch_control:
            self.vel_dim = 6
            self.vel_action_scale = np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.25])  # m/s and rad/s
        else:
            self.vel_dim = 4
            self.vel_action_scale = np.array([0.5, 0.5, 0.5, 0.25])  # m/s and rad/s
            
        # Use provided action_dim if specified, otherwise use default calculation
        if self.wbc_observation_enabled:
            self.arm_dim = 14 # 关节 joint space
            self.action_dim = self.vel_dim + self.arm_dim # 4 + 14 = 18
        else:
            # Default behavior: 14 for arm joints
            self.arm_dim = 6 # 末端 eef position
            self.action_dim = self.vel_dim + self.arm_dim # 4 + 6 = 10

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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        # Define arm joint names in order (only use the number of joints specified by arm_dim)
        self.arm_joint_names = [f'zarm_l{i}_joint' for i in range(1, 8)] + [f'zarm_r{i}_joint' for i in range(1, 8)]
        # Truncate to match arm_dim if necessary
        if len(self.arm_joint_names) > self.arm_dim:
            self.arm_joint_names = self.arm_joint_names[:self.arm_dim]

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
            self.arm_joint_centers = np.zeros(self.arm_dim)
            self.arm_joint_scales = np.full(self.arm_dim, np.deg2rad(10.0))

        # Task-specific state
        self.initial_box_pose = None
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        
        # Last converted VR action for recording
        self._last_vr_action = np.zeros(self.action_space.shape, dtype=np.float32)
        
        # Action smoothing parameters
        self.vel_smoothing_factor = vel_smoothing_factor  # 0.0 = no smoothing, 1.0 = full smoothing
        self.arm_smoothing_factor = arm_smoothing_factor  # Slightly more smoothing for arm joints
        self.last_smoothed_vel_action = np.zeros(self.vel_dim, dtype=np.float32)
        self.last_smoothed_arm_action = np.zeros(self.arm_dim, dtype=np.float32)
        self.is_first_action = True
        
        if self.debug:
            rospy.loginfo(f"Action space dimension: {self.action_dim} (vel: {self.vel_dim}, arm: {self.arm_dim})")
            rospy.loginfo(f"Action smoothing initialized - Vel factor: {vel_smoothing_factor}, Arm factor: {arm_smoothing_factor}")
            rospy.loginfo(f"WBC observation mode: {'enabled' if self.wbc_observation_enabled else 'disabled'}")

    def change_mobile_ctrl_mode(self, mode: int):
        # print(f"change_mobile_ctrl_mode: {mode}")
        mobile_manipulator_service_name = "/mobile_manipulator_mpc_control"
        try:
            rospy.wait_for_service(mobile_manipulator_service_name)
            changeHandTrackingMode_srv = rospy.ServiceProxy(mobile_manipulator_service_name, changeArmCtrlMode)
            changeHandTrackingMode_srv(mode)
        except rospy.ROSException:
            rospy.logerr(f"Service {mobile_manipulator_service_name} not available")

    def _ang_vel_callback(self, msg):
        with self.ang_vel_lock:
            self.latest_ang_vel = msg

    def _lin_accel_callback(self, msg):
        with self.lin_accel_lock:
            self.latest_lin_accel = msg

    def _wbc_callback(self, msg):
        with self.wbc_lock:
            self.latest_wbc = msg

    def _robot_pose_callback(self, msg):
        """Callback for robot pose messages when WBC observation is disabled."""
        with self.robot_pose_lock:
            self.latest_robot_pose = msg

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
            if self._latest_vr_arm_traj is not None and len(self._latest_vr_arm_traj.position) >= self.arm_dim:
                # JointState message has position array directly
                arm_positions_deg = np.array(self._latest_vr_arm_traj.position[:self.arm_dim], dtype=np.float32)
                
                # Convert degrees to radians if needed
                arm_positions_rad = np.deg2rad(arm_positions_deg)
                
                # Normalize to [-1, 1] using joint centers and scales
                arm_action = (arm_positions_rad - self.arm_joint_centers) / self.arm_joint_scales
                arm_action = np.clip(arm_action, -1.0, 1.0)
                
                action[4:4+self.arm_dim] = arm_action
            
            return action



    def _setup_ros_communication(self):
        """
        Implement this method to set up ROS publishers, subscribers,
        and service clients specific to the Kuavo robot.
        """
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # Replace the arm_traj_pub with the new publisher
        self.ee_pose_pub = rospy.Publisher('/mm/two_arm_hand_pose_cmd', twoArmHandPoseCmd, queue_size=10)

        # Service Client
        self.reset_client = rospy.ServiceProxy('/isaac_lab_reset_scene', resetIsaaclab)

        # Subscribers for headerless topics that are not synchronized
        rospy.Subscriber('/state_estimate/imu_data_filtered/angularVel', Float64MultiArray, self._ang_vel_callback)
        rospy.Subscriber('/state_estimate/imu_data_filtered/linearAccel', Float64MultiArray, self._lin_accel_callback)
        
        # Conditionally subscribe to WBC or robot pose based on flag
        if self.wbc_observation_enabled:
            rospy.Subscriber('/humanoid_wbc_observation', mpc_observation, self._wbc_callback)
            if self.debug:
                rospy.loginfo("WBC observation enabled - subscribing to /humanoid_wbc_observation")
        else:
            rospy.Subscriber('/robot_pose', PoseStamped, self._robot_pose_callback)
            if self.debug:
                rospy.loginfo("WBC observation disabled - subscribing to /robot_pose")

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
            
        # Get WBC or robot pose data based on mode
        if self.wbc_observation_enabled:
            with self.wbc_lock:
                wbc = self.latest_wbc
            robot_pose = None
        else:
            with self.robot_pose_lock:
                robot_pose = self.latest_robot_pose
            wbc = None

        # Wait until all data sources are available
        if ang_vel is None or lin_accel is None:
            if self.debug:
                rospy.logwarn_throttle(1.0, "IMU data not yet available for observation callback.")
            return
            
        if self.wbc_observation_enabled and wbc is None:
            if self.debug:
                rospy.logwarn_throttle(1.0, "WBC data not yet available for observation callback.")
            return
        elif not self.wbc_observation_enabled and robot_pose is None:
            if self.debug:
                rospy.logwarn_throttle(1.0, "Robot pose data not yet available for observation callback.")
            return

        with self.obs_lock:
            try:
                # Process image data
                cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
                # TODO: Resize image if necessary, for now assuming it's the correct size
                # cv_image = cv2.resize(cv_image, self.image_size)
                rgb_image = cv_image[:, :, ::-1].copy() # BGR to RGB

                # Process state data
                left_eef_position = np.array([left_eef.pose.position.x, left_eef.pose.position.y, left_eef.pose.position.z])
                left_eef_orientation = np.array([left_eef.pose.orientation.x, left_eef.pose.orientation.y, left_eef.pose.orientation.z, left_eef.pose.orientation.w])
                left_eef_data = np.concatenate([left_eef_position, left_eef_orientation])
                
                right_eef_position = np.array([right_eef.pose.position.x, right_eef.pose.position.y, right_eef.pose.position.z])
                right_eef_orientation = np.array([right_eef.pose.orientation.x, right_eef.pose.orientation.y, right_eef.pose.orientation.z, right_eef.pose.orientation.w])
                right_eef_data = np.concatenate([right_eef_position, right_eef_orientation])

                # arm_data
                arm_data = np.array(sensors.joint_data.joint_q[12:26])
                
                # imu data 
                ang_vel_data = np.array(ang_vel.data[:3])
                lin_accel_data = np.array(lin_accel.data[:3])
                
                # Process WBC or robot pose data based on mode
                if self.wbc_observation_enabled:
                    wbc_data = np.array(wbc.state.value[:12])
                else:
                    # Convert robot pose to 12-dimensional data similar to WBC
                    # Extract position and orientation components
                    robot_pos = np.array([robot_pose.pose.position.x, robot_pose.pose.position.y, robot_pose.pose.position.z])
                    robot_orn = np.array([robot_pose.pose.orientation.x, robot_pose.pose.orientation.y, 
                                        robot_pose.pose.orientation.z, robot_pose.pose.orientation.w])
                    # Pad with zeros to match WBC data dimension (7)
                    wbc_data = np.concatenate([robot_pos, robot_orn])  # 3 + 4 = 7
                
                box_pos_data = np.array([
                    box_real.pose.position.x, box_real.pose.position.y, box_real.pose.position.z
                ])
                box_orn_data = np.array([
                    box_real.pose.orientation.x, box_real.pose.orientation.y,
                    box_real.pose.orientation.z, box_real.pose.orientation.w
                ])


                if self.wbc_observation_enabled:
                    """
                        46 维度 - agent_pos
                        7 + 7
                        14
                        3 + 3
                        12
                    """
                    agent_pos_obs = np.concatenate([
                        left_eef_data, right_eef_data, 
                        arm_data, 
                        ang_vel_data, lin_accel_data, 
                        wbc_data
                    ]).astype(np.float32)
                else:
                    """
                        23 维度 - agent_pos
                        3 + 3
                        14 
                        3
                    """
                    agent_pos_obs = np.concatenate([
                        left_eef_position, right_eef_position, 
                        arm_data, 
                        robot_pos
                    ]).astype(np.float32)

                if self.wbc_observation_enabled:
                    """
                        7 维度 - environment_state
                    """
                    env_state_obs = np.concatenate([
                        box_pos_data, box_orn_data
                    ]).astype(np.float32)
                else:
                    """
                        3 维度 - environment_state
                    """
                    env_state_obs = box_pos_data.astype(np.float32)

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
            
        # Apply action smoothing for non-VR intervention mode
        if not self._is_vr_intervention_active:
            action = self._smooth_action(action)
        
        # De-normalize and publish cmd_vel
        twist_cmd = Twist()
        
        vel_dim = 6 if self.enable_roll_pitch_control else 4
        if self.enable_roll_pitch_control:
            vel_action = action[:6] * self.vel_action_scale
            twist_cmd.linear.x = vel_action[0]
            twist_cmd.linear.y = vel_action[1]
            twist_cmd.linear.z = vel_action[2]
            twist_cmd.angular.x = vel_action[3]
            twist_cmd.angular.y = vel_action[4]
            twist_cmd.angular.z = vel_action[5]
            ee_action = action[6:]
        else:
            vel_action = action[:4] * self.vel_action_scale
            twist_cmd.linear.x = vel_action[0]
            twist_cmd.linear.y = vel_action[1]
            twist_cmd.linear.z = vel_action[2]
            twist_cmd.angular.x = 0.0
            twist_cmd.angular.y = 0.0
            twist_cmd.angular.z = vel_action[3]
            ee_action = action[4:]

        self.cmd_vel_pub.publish(twist_cmd)

        # Publish end-effector pose command (replace joint-level command)
        # ee_action: [L_pos(3), L_quat(4), R_pos(3), R_quat(4)] or [L_pos(3), R_pos(3)]
        self.change_mobile_ctrl_mode(IncrementalMpcCtrlMode.ArmOnly.value)

        if self.wbc_observation_enabled: # 7 + 7 6dof数据
            if len(ee_action) >= 14:
                """
                    使用action得到的姿态 - 末端eef position
                """
                left_pos = ee_action[0:3]
                left_quat = ee_action[3:7]
                right_pos = ee_action[7:10]
                right_quat = ee_action[10:14]
            else:
                # Handle reduced action space
                left_pos = ee_action[0:3] if len(ee_action) >= 3 else np.zeros(3)
                left_quat = np.array([0.0, -0.70711, 0.0, 0.70711])
                right_pos = ee_action[3:6] if len(ee_action) >= 6 else np.zeros(3)
                right_quat = np.array([0.0, -0.70711, 0.0, 0.70711])
        else: # 3 + 3 position数据
            """
                固定姿态 - 末端eef position
            """
            left_pos = ee_action[0:3] if len(ee_action) >= 3 else np.zeros(3)
            left_quat = np.array([0.0, -0.70711, 0.0, 0.70711])
            right_pos = ee_action[3:6] if len(ee_action) >= 6 else np.zeros(3)
            right_quat = np.array([0.0, -0.70711, 0.0, 0.70711])

        left_elbow_pos = np.zeros(3)
        right_elbow_pos = np.zeros(3)

        msg = twoArmHandPoseCmd()
        msg.hand_poses.left_pose.pos_xyz = left_pos.tolist()
        msg.hand_poses.left_pose.quat_xyzw = left_quat.tolist()
        msg.hand_poses.left_pose.elbow_pos_xyz = left_elbow_pos.tolist()

        msg.hand_poses.right_pose.pos_xyz = right_pos.tolist()
        msg.hand_poses.right_pose.quat_xyzw = right_quat.tolist()
        msg.hand_poses.right_pose.elbow_pos_xyz = right_elbow_pos.tolist()
        
        # Set default IK params (can be customized as needed)
        msg.use_custom_ik_param = False
        msg.joint_angles_as_q0 = False
        msg.ik_param = ikSolveParam()
        msg.frame = 0  # keep current frame
        self.ee_pose_pub.publish(msg)

        # # 打印action_dim
        # rospy.loginfo(f"self.action_dim : {self.action_dim}")
        # rospy.loginfo(f"self.arm_dim : {self.arm_dim}")
        # rospy.loginfo(f"self.vel_dim : {self.vel_dim}")

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """
        Applies smoothing to the action to reduce sudden changes.
        This is particularly useful for velocity and arm joint actions.
        
        Args:
            action: The raw action array from the policy
            
        Returns:
            The smoothed action array
        """
        vel_dim = 4 if not self.enable_roll_pitch_control else 6

        # Handle first action (no previous action to smooth with)
        if self.is_first_action:
            self.last_smoothed_vel_action = action[:vel_dim]
            self.last_smoothed_arm_action = action[vel_dim:vel_dim+self.arm_dim]
            self.is_first_action = False
            return action

        # Smooth velocity action using exponential moving average
        vel_action = action[:vel_dim]
        smoothed_vel_action = (
            self.last_smoothed_vel_action * (1 - self.vel_smoothing_factor) + 
            vel_action * self.vel_smoothing_factor
        )
        self.last_smoothed_vel_action = smoothed_vel_action

        # Smooth arm joint action using exponential moving average
        arm_action = action[vel_dim:vel_dim+self.arm_dim]
        smoothed_arm_action = (
            self.last_smoothed_arm_action * (1 - self.arm_smoothing_factor) + 
            arm_action * self.arm_smoothing_factor
        )
        self.last_smoothed_arm_action = smoothed_arm_action

        # Combine smoothed actions
        smoothed_action = np.concatenate([smoothed_vel_action, smoothed_arm_action])
        
        if self.debug:
            # Log smoothing statistics occasionally
            vel_change = np.linalg.norm(vel_action - self.last_smoothed_vel_action)
            arm_change = np.linalg.norm(arm_action - self.last_smoothed_arm_action)
            if vel_change > 0.5 or arm_change > 0.5:  # Only log significant changes
                # rospy.loginfo(f"Action smoothing - Vel change: {vel_change:.3f}, Arm change: {arm_change:.3f}")
                pass
        
        return smoothed_action

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
        
        if self.wbc_observation_enabled:
            # WBC enabled: agent_state has 46 dimensions
            left_eef_pos = agent_state[0:3]
            right_eef_pos = agent_state[7:10]
            box_pos = env_state[0:3]
            box_orn = env_state[3:7]
        else:
            # WBC disabled: agent_state has 23 dimensions
            left_eef_pos = agent_state[0:3]
            right_eef_pos = agent_state[3:6]
            box_pos = env_state[0:3]
            box_orn = None  # No orientation data when WBC is disabled

        # Calculate distances
        dist_left_hand_to_box = np.linalg.norm(left_eef_pos - box_pos)
        dist_right_hand_to_box = np.linalg.norm(right_eef_pos - box_pos)

        # Check conditions for success
        z_lift = box_pos[2] - self.initial_box_pose['position'][2]
        
        # Orientation similarity (only when WBC is enabled)
        if self.wbc_observation_enabled and box_orn is not None:
            q1 = box_orn
            q2 = self.initial_box_pose['orientation']
            orientation_similarity = abs(np.dot(q1, q2))
            orientation_success = orientation_similarity > 0.98 # within ~11 degrees
        else:
            orientation_similarity = 1.0  # Assume good orientation when WBC is disabled
            orientation_success = True

        # Success condition
        lift_success = z_lift > 0.10 # 10cm视作成功
        hands_close_success = dist_left_hand_to_box < 0.5 and dist_right_hand_to_box < 0.5
        
        # Dense Reward Calculation
        reward = 0.0

        # 1. Reward for successful lift
        if lift_success:
            reward += 100.0
            
        # 2. Reward for torso proximity to box (躯干到box的奖励)
        if self.wbc_observation_enabled:
            # WBC enabled: torso position is in wbc_data (first 3 elements of wbc_data)
            # agent_state structure: [left_eef_data(7), right_eef_data(7), arm_data(14), ang_vel_data(3), lin_accel_data(3), wbc_data(12)]
            # wbc_data starts at index 34, so torso position is at indices 34-36
            torso_pos = agent_state[40:43]  # 躯干位置 (x, y, z)
        else:
            # WBC disabled: torso position is robot_pos
            # agent_state structure: [left_eef_position(3), right_eef_position(3), arm_data(14), robot_pos(3)]
            # robot_pos is at indices 20-22
            torso_pos = agent_state[20:23]  # 躯干位置 (x, y, z)
        
        # Calculate distance from torso to box
        dist_torso_to_box = np.linalg.norm(torso_pos - box_pos)
        # Exponential decay reward for torso proximity to box
        reward += np.exp(-2.0 * dist_torso_to_box) * 8.0  # Exponential decay, scale by 8
        
        # 3. Reward for left and right EEF proximity to box (考虑XYZ三个维度)
        dist_left_eef_to_box = np.linalg.norm(left_eef_pos - box_pos)  # 3D距离
        dist_right_eef_to_box = np.linalg.norm(right_eef_pos - box_pos)  # 3D距离
        # 使用3D距离计算末端接近奖励，提供更全面的接近指导
        reward += (np.exp(-2.0 * dist_left_eef_to_box) + np.exp(-2.0 * dist_right_eef_to_box)) * 3.0 # Exponential decay, scale by 3

        # Check for episode termination (can still terminate on success for sparse tasks)
        terminated = lift_success # Only terminate on lift success for now

        info["succeed"] = lift_success
        info["z_lift"] = z_lift
        info["orientation_similarity"] = orientation_similarity
        info["dist_left_hand_to_box"] = dist_left_hand_to_box
        info["dist_right_hand_to_box"] = dist_right_hand_to_box
        info["dist_torso_to_box"] = dist_torso_to_box

        if self.debug:
            # 计算各个奖励项的贡献
            lift_reward = 100.0 if lift_success else 0.0
            torso_reward = np.exp(-2.0 * dist_torso_to_box) * 8.0
            eef_proximity_reward = (np.exp(-2.0 * dist_left_eef_to_box) + np.exp(-2.0 * dist_right_eef_to_box)) * 3.0
            
            # print(f"z_lift: {z_lift:.3f}, orient_sim: {orientation_similarity:.3f}, torso_dist: {dist_torso_to_box:.3f}, total_reward: {reward:.3f}, terminated: {terminated}")
            # print(f"  Rewards breakdown - Lift: {lift_reward:.3f}, Torso: {torso_reward:.3f}, EEF-proximity: {eef_proximity_reward:.3f}")
            # print(f"  EEF 3D distances - Left: {dist_left_eef_to_box:.3f}, Right: {dist_right_eef_to_box:.3f}")
            
        return reward, terminated, info

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
            if self.wbc_observation_enabled:
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
        if self.wbc_observation_enabled:
            box_orn = obs_stable['environment_state'][3:7]
        else:
            box_orn = np.array([0.0, 0.0, 0.0, 1.0])  # Default orientation when WBC is disabled
        self.initial_box_pose = {'position': box_pos, 'orientation': box_orn}
        
        if self.debug:
            rospy.loginfo(f"reset - Initial box position (first): {obs['environment_state'][0:3]}")
            rospy.loginfo(f"reset - Initial box position (stable): {box_pos}")
            rospy.loginfo(f"reset - Position difference: {box_pos - obs['environment_state'][0:3]}")
        
        self.last_action.fill(0.0)
        # Reset the first step flag
        self._is_first_step = True
        
        # Reset action smoothing state
        self.is_first_action = True
        self.last_smoothed_vel_action.fill(0.0)
        self.last_smoothed_arm_action.fill(0.0)

        return obs_stable, {}


if __name__ == "__main__":
    import traceback
    
    print("Starting RLKuavoGymEnv test script...")

    # The environment itself handles ROS node initialization,
    # but it's good practice to have it here for a standalone script.
    if not rospy.core.is_initialized():
        rospy.init_node('rl_kuavo_env_test', anonymous=True)

    # Instantiate the environment with debugging enabled
    env = RLKuavoGymEnv(debug=True, enable_roll_pitch_control=False, wbc_observation_enabled=True)

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
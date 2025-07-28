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
from collections import deque

TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD = True

# 增加DEMO模式的动作尺度常量
DEMO_MAX_INCREMENT_PER_STEP = 0.08  # DEMO模式下每步最大8cm增量
DEMO_MAX_INCREMENT_RANGE = 0.8     # DEMO模式下最大累积增量范围80cm

# Demo mode target end-effector positions
DEMO_TARGET_LEFT_POS = np.array([0.4678026345146559, 0.2004180715613648, 0.15417275957965042])
DEMO_TARGET_LEFT_QUAT = np.array([0.0, -0.70711, 0.0, 0.70711])
DEMO_TARGET_RIGHT_POS = np.array([0.4678026345146559, -0.2004180715613648, 0.15417275957965042])
DEMO_TARGET_RIGHT_QUAT = np.array([0.0, -0.70711, 0.0, 0.70711])

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

    # 固定基准位置常量 - 基于这些位置进行增量控制
    FIXED_LEFT_POS = np.array([0.3178026345146559, 0.4004180715613648, -0.019417275957965042])
    FIXED_LEFT_QUAT = np.array([0.0, -0.70711, 0.0, 0.70711])
    FIXED_RIGHT_POS = np.array([0.3178026345146559, -0.4004180715613648, -0.019417275957965042])
    FIXED_RIGHT_QUAT = np.array([0.0, -0.70711, 0.0, 0.70711])
    
    # 增量控制的最大范围 (米)
    MAX_INCREMENT_RANGE = 0.2  # ±20cm的增量范围
    MAX_INCREMENT_PER_STEP = 0.02  # 每步最大2cm的增量变化

    def __init__(self, debug: bool = True, image_size=(224, 224), enable_roll_pitch_control: bool = False, 
                 vel_smoothing_factor: float = 0.3, arm_smoothing_factor: float = 0.4, 
                 wbc_observation_enabled: bool = True, action_dim: int = None):
        # Separate storage for headerless topics that will be initialized in callbacks.
        # This needs to be done BEFORE super().__init__() which sets up subscribers.
        self.latest_ang_vel = None
        self.latest_lin_accel = None
        self.latest_wbc = None
        self.latest_robot_pose = None  # New: for robot pose when WBC is disabled
        # Add new variables for base_link end-effector poses
        self.latest_base_link_eef_left = None
        self.latest_base_link_eef_right = None
        self.ang_vel_lock = threading.Lock()
        self.lin_accel_lock = threading.Lock()
        self.wbc_lock = threading.Lock()
        self.robot_pose_lock = threading.Lock()  # New: lock for robot pose
        # Add new locks for base_link end-effector poses
        self.base_link_eef_left_lock = threading.Lock()
        self.base_link_eef_right_lock = threading.Lock()
        
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

        # 添加当前增量状态跟踪
        self.current_left_increment = np.zeros(3, dtype=np.float32)
        self.current_right_increment = np.zeros(3, dtype=np.float32)
        self.last_left_increment = np.zeros(3, dtype=np.float32)
        self.last_right_increment = np.zeros(3, dtype=np.float32)

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
            # agent_pos: 3 (left_eef_pos) + 3 (right_eef_pos) + 14 (arm_joints) + 3 (robot_pos) + 3 (base_link_left_eef) + 3 (base_link_right_eef) = 29
            # environment_state: 3 (box_pos) = 3
            agent_dim = 29
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
        
        # For smooth end-effector motion penalty
        self.last_left_eef_pos = None
        self.last_right_eef_pos = None
        
        # For trajectory tracking rewards - track distance changes over time
        self.last_dist_left_hand_to_box = None
        self.last_dist_right_hand_to_box = None
        self.last_dist_torso_to_box = None
        
        # Step counting for efficiency reward
        self.episode_step_count = 0
        
        # For continuous approach tracking
        self.consecutive_approach_steps_left = 0
        self.consecutive_approach_steps_right = 0
        self.consecutive_approach_steps_torso = 0
        
        # Distance change history for smooth trajectory reward
        self.distance_change_history_left = deque(maxlen=5)
        self.distance_change_history_right = deque(maxlen=5)
        
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

    def _base_link_eef_left_callback(self, msg):
        """Callback for base_link left end-effector pose messages."""
        with self.base_link_eef_left_lock:
            self.latest_base_link_eef_left = msg

    def _base_link_eef_right_callback(self, msg):
        """Callback for base_link right end-effector pose messages."""
        with self.base_link_eef_right_lock:
            self.latest_base_link_eef_right = msg

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
        Now supports incremental position control.
        
        Returns:
            Action array matching the environment's action space (with increments)

            获取vr的如下信息。
            获取/cmd_vel
            获取/mm_kuavo_arm_traj - 现在转换为增量控制
            在RLKuavoMetaVRWrapper的step当中, 将获取到的值映射到action数组中,该数据用于最终的action_intervention record和buffer都会使用这个key里面的action
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
            
            # Process arm trajectory data if available - Convert to increments
            if self._latest_vr_arm_traj is not None and len(self._latest_vr_arm_traj.position) >= self.arm_dim:
                if self.wbc_observation_enabled:
                    # WBC mode: convert joint angles to increments
                    arm_positions_deg = np.array(self._latest_vr_arm_traj.position[:self.arm_dim], dtype=np.float32)
                    arm_positions_rad = np.deg2rad(arm_positions_deg)
                    arm_action = (arm_positions_rad - self.arm_joint_centers) / self.arm_joint_scales
                    arm_action = np.clip(arm_action, -1.0, 1.0)
                    action[4:4+self.arm_dim] = arm_action
                else:
                    # Position mode: convert to increments based on fixed reference
                    # Note: This requires VR to send position data, not joint angles
                    # For now, use a simple conversion - this may need adjustment based on VR data format
                    # Convert VR data to position increments
                    left_increment = np.zeros(3, dtype=np.float32)
                    right_increment = np.zeros(3, dtype=np.float32)
                    
                    # This is a placeholder - actual implementation depends on VR data format
                    # You may need to adjust this based on how VR sends position data
                    if len(self._latest_vr_arm_traj.position) >= 6:
                        # Assume VR sends [left_x, left_y, left_z, right_x, right_y, right_z]
                        vr_left_pos = np.array(self._latest_vr_arm_traj.position[0:3], dtype=np.float32)
                        vr_right_pos = np.array(self._latest_vr_arm_traj.position[3:6], dtype=np.float32)
                        
                        # Calculate increments from fixed reference positions
                        left_increment = vr_left_pos - self.FIXED_LEFT_POS
                        right_increment = vr_right_pos - self.FIXED_RIGHT_POS
                        
                        # Limit increments to reasonable ranges
                        left_increment = np.clip(left_increment, -self.MAX_INCREMENT_RANGE, self.MAX_INCREMENT_RANGE)
                        right_increment = np.clip(right_increment, -self.MAX_INCREMENT_RANGE, self.MAX_INCREMENT_RANGE)
                    
                    action[4:7] = left_increment
                    action[7:10] = right_increment
            
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
            # Subscribe to base_link end-effector poses when WBC is disabled
            rospy.Subscriber('/fk/base_link_eef_left', PoseStamped, self._base_link_eef_left_callback)
            rospy.Subscriber('/fk/base_link_eef_right', PoseStamped, self._base_link_eef_right_callback)
            if self.debug:
                rospy.loginfo("WBC observation disabled - subscribing to /robot_pose, /fk/base_link_eef_left, /fk/base_link_eef_right")

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
            base_link_eef_left = None
            base_link_eef_right = None
        else:
            with self.robot_pose_lock:
                robot_pose = self.latest_robot_pose
            # Get base_link end-effector poses when WBC is disabled
            with self.base_link_eef_left_lock:
                base_link_eef_left = self.latest_base_link_eef_left
            with self.base_link_eef_right_lock:
                base_link_eef_right = self.latest_base_link_eef_right
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
        elif not self.wbc_observation_enabled and (robot_pose is None or base_link_eef_left is None or base_link_eef_right is None):
            if self.debug:
                rospy.logwarn_throttle(1.0, "Robot pose or base_link end-effector data not yet available for observation callback.")
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
                    
                    # Extract base_link end-effector positions
                    base_link_left_eef_pos = np.array([
                        base_link_eef_left.pose.position.x,
                        base_link_eef_left.pose.position.y,
                        base_link_eef_left.pose.position.z
                    ])
                    base_link_right_eef_pos = np.array([
                        base_link_eef_right.pose.position.x,
                        base_link_eef_right.pose.position.y,
                        base_link_eef_right.pose.position.z
                    ])
                
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
                        29 维度 - agent_pos (increased from 23)
                        3 + 3 (world frame eef positions)
                        14 (arm joints)
                        3 (robot position)
                        3 + 3 (base_link frame eef positions)
                    """
                    agent_pos_obs = np.concatenate([
                        left_eef_position, right_eef_position, 
                        arm_data, 
                        robot_pos,
                        base_link_left_eef_pos, base_link_right_eef_pos
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
        
        # Apply action constraints
        action = self._apply_action_constraints(action)
        
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

        # Publish arm commands based on mode
        self.change_mobile_ctrl_mode(IncrementalMpcCtrlMode.ArmOnly.value)
        
        global TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD
        if TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD:
            # ========== DEMO MODE: DIRECT ARM CONTROL ==========
            # Always use action-based arm control for joint space learning
            self._publish_action_based_arm_poses(ee_action)
            if self.debug:
                print(f"[DEMO MODE] Publishing action-based arm poses for joint space learning")
        else:
            # ========== NORMAL MODE: STAGE-BASED ARM CONTROL ==========
            # Determine current stage based on torso-to-box distance
            current_stage = self._get_current_stage()

            if current_stage == "approach":
                # STAGE 1: APPROACH STAGE - Use fixed arm poses
                self._publish_fixed_arm_poses()
                if self.debug:
                    print(f"[STAGE 1] Publishing fixed arm poses during approach stage")
            else:
                # STAGE 2: GRASP STAGE - Use action-based arm control
                self._publish_action_based_arm_poses(ee_action)
                if self.debug:
                    print(f"[STAGE 2] Publishing action-based arm poses during grasp stage")

    def _get_current_stage(self) -> str:
        """
        Determine the current stage based on the latest observation.
        
        Returns:
            "approach" if in stage 1 (approaching box), "grasp" if in stage 2 (grasping box)
        """
        if not hasattr(self, 'latest_obs') or self.latest_obs is None:
            return "approach"  # Default to approach stage if no observation available
        
        try:
            agent_state = self.latest_obs['agent_pos']
            env_state = self.latest_obs['environment_state']
            
            # Extract positions based on observation mode
            if self.wbc_observation_enabled:
                # WBC enabled: agent_state has 46 dimensions
                left_eef_pos = agent_state[0:3]
                right_eef_pos = agent_state[7:10]
                arm_joints = agent_state[14:28]  # 14 joint angles
                torso_pos = agent_state[40:43]
                box_pos = env_state[0:3]
                box_orn = env_state[3:7]
            else:
                # WBC disabled: agent_state has 29 dimensions (increased from 23)
                left_eef_pos = agent_state[0:3]           # world frame left eef pos
                right_eef_pos = agent_state[3:6]          # world frame right eef pos
                arm_joints = agent_state[6:20]            # 14 joint angles
                torso_pos = agent_state[20:23]            # robot position
                # Additional base_link end-effector positions (new data)
                base_link_left_eef_pos = agent_state[23:26]   # base_link frame left eef pos
                base_link_right_eef_pos = agent_state[26:29]  # base_link frame right eef pos
                box_pos = env_state[0:3]
                box_orn = None  # No orientation data when WBC is disabled

            # Calculate distances
            dist_left_hand_to_box = np.linalg.norm(left_eef_pos - box_pos)
            dist_right_hand_to_box = np.linalg.norm(right_eef_pos - box_pos)
            dist_torso_to_box = np.linalg.norm(torso_pos - box_pos)

            # Stage determination (same logic as in reward function)
            if dist_torso_to_box > 0.3:
                return "approach"
            else:
                return "grasp"
                
        except (KeyError, IndexError, TypeError) as e:
            if self.debug:
                rospy.logwarn(f"Error determining stage: {e}, defaulting to approach stage")
            return "approach"

    def _publish_fixed_arm_poses(self):
        """
        Publish fixed arm poses for the approach stage.
        """
        # Fixed poses for approach stage
        left_pos = np.array([0.3178026345146559, 0.4004180715613648, -0.019417275957965042])
        left_quat = np.array([0.0, -0.70711, 0.0, 0.70711])
        right_pos = np.array([0.3178026345146559, -0.4004180715613648, -0.019417275957965042])
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
        
        # Set default IK params
        msg.use_custom_ik_param = False
        msg.joint_angles_as_q0 = False
        msg.ik_param = ikSolveParam()
        msg.frame = 3  # VR Frame
        self.ee_pose_pub.publish(msg)

    def _publish_action_based_arm_poses(self, ee_action: np.ndarray):
        """
        Publish arm poses based on the action input (original logic for grasp stage).
        Now uses incremental position control based on fixed reference positions.
        
        Args:
            ee_action: The arm portion of the action array (increments)
        """
        # ee_action现在是增量: [L_delta_pos(3), R_delta_pos(3)]
        if self.wbc_observation_enabled: # 7 + 7 6dof数据 - 但现在我们只处理位置增量
            if len(ee_action) >= 14:
                # 提取增量
                left_increment = ee_action[0:3]
                left_quat = ee_action[3:7]  # 如果有quaternion数据，保持原样
                right_increment = ee_action[7:10]
                right_quat = ee_action[10:14]
            else:
                # Handle reduced action space - 只有位置增量
                left_increment = ee_action[0:3] if len(ee_action) >= 3 else np.zeros(3)
                left_quat = self.FIXED_LEFT_QUAT.copy()
                right_increment = ee_action[3:6] if len(ee_action) >= 6 else np.zeros(3)
                right_quat = self.FIXED_RIGHT_QUAT.copy()
        else: # 3 + 3 position增量数据
            """
                基于固定姿态的增量控制 - 末端eef position increments
            """
            left_increment = ee_action[0:3] if len(ee_action) >= 3 else np.zeros(3)
            left_quat = self.FIXED_LEFT_QUAT.copy()
            right_increment = ee_action[3:6] if len(ee_action) >= 6 else np.zeros(3)
            right_quat = self.FIXED_RIGHT_QUAT.copy()

        # 应用增量约束和平滑处理
        left_increment, right_increment = self._process_incremental_action(left_increment, right_increment)
        
        # 计算最终的绝对位置 = 基准位置 + 当前累积增量
        global TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD
        if TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD:
            # DEMO 模式：基于新的目标位置进行增量控制
            left_pos = DEMO_TARGET_LEFT_POS + self.current_left_increment
            right_pos = DEMO_TARGET_RIGHT_POS + self.current_right_increment
            left_quat = DEMO_TARGET_LEFT_QUAT.copy()
            right_quat = DEMO_TARGET_RIGHT_QUAT.copy()
        else:
            # 正常模式：基于原始固定位置进行增量控制
            left_pos = self.FIXED_LEFT_POS + self.current_left_increment
            right_pos = self.FIXED_RIGHT_POS + self.current_right_increment

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
        msg.frame = 3  # keep current frame3 | 3 为vr系
        self.ee_pose_pub.publish(msg)
        
        if self.debug:
            print(f"[INCREMENTAL DEBUG] Left increment: {left_increment}, Right increment: {right_increment}")
            print(f"[INCREMENTAL DEBUG] Left absolute pos: {left_pos}, Right absolute pos: {right_pos}")
            print(f"[INCREMENTAL DEBUG] Current cumulative - Left: {self.current_left_increment}, Right: {self.current_right_increment}")

    def _process_incremental_action(self, left_increment: np.ndarray, right_increment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        处理增量动作，应用约束和平滑处理。
        
        Args:
            left_increment: 左手的增量位置
            right_increment: 右手的增量位置
            
        Returns:
            tuple: 处理后的 (left_increment, right_increment)
        """
        # 1. 限制单步增量的最大变化量（确保速度平滑性）
        global TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD
        if TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD:
            max_step_increment = DEMO_MAX_INCREMENT_PER_STEP
        else:
            max_step_increment = self.MAX_INCREMENT_PER_STEP
        
        left_increment = np.clip(left_increment, -max_step_increment, max_step_increment)
        right_increment = np.clip(right_increment, -max_step_increment, max_step_increment)
        
        # 2. 应用平滑处理（如果不是第一次动作）
        if not self.is_first_action:
            # 使用指数移动平均进行平滑
            left_increment = (
                self.last_left_increment * (1 - self.arm_smoothing_factor) + 
                left_increment * self.arm_smoothing_factor
            )
            right_increment = (
                self.last_right_increment * (1 - self.arm_smoothing_factor) + 
                right_increment * self.arm_smoothing_factor
            )
        
        # 3. 更新当前累积增量
        new_left_increment = self.current_left_increment + left_increment
        new_right_increment = self.current_right_increment + right_increment
        
        # 4. 限制累积增量的总范围（确保位置在合理范围内）
        if TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD:
            # DEMO 模式：使用专门的大增量范围
            new_left_increment = np.clip(new_left_increment, -DEMO_MAX_INCREMENT_RANGE, DEMO_MAX_INCREMENT_RANGE)
            new_right_increment = np.clip(new_right_increment, -DEMO_MAX_INCREMENT_RANGE, DEMO_MAX_INCREMENT_RANGE)
        else:
            # 正常模式：使用标准增量范围
            new_left_increment = np.clip(new_left_increment, -self.MAX_INCREMENT_RANGE, self.MAX_INCREMENT_RANGE)
            new_right_increment = np.clip(new_right_increment, -self.MAX_INCREMENT_RANGE, self.MAX_INCREMENT_RANGE)
        
        # 5. 应用任务特定的约束
        new_left_increment, new_right_increment = self._apply_task_specific_constraints(
            new_left_increment, new_right_increment
        )
        
        # 6. 更新状态
        self.last_left_increment = left_increment.copy()
        self.last_right_increment = right_increment.copy()
        self.current_left_increment = new_left_increment.copy()
        self.current_right_increment = new_right_increment.copy()
        
        return left_increment, right_increment

    def _apply_task_specific_constraints(self, left_increment: np.ndarray, right_increment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        应用任务特定的约束条件。
        
        Args:
            left_increment: 左手累积增量
            right_increment: 右手累积增量
            
        Returns:
            tuple: 约束后的增量
        """
        global TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD
        
        if TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD:
            # ========== DEMO MODE: POSITION CONTROL CONSTRAINTS ==========
            # Apply constraints suitable for end-effector position control learning
            # Base calculations on the new target positions
            
            # 计算绝对位置用于约束检查（基于新的目标位置）
            left_abs_pos = DEMO_TARGET_LEFT_POS + left_increment
            right_abs_pos = DEMO_TARGET_RIGHT_POS + right_increment
            
            # 设置合理的安全范围，围绕目标位置
            # 允许充分的探索空间以学习位置控制
            DEMO_WORKSPACE_RADIUS = 0.6  # 扩大工作空间半径到60cm
            
            # 计算相对于目标位置的距离（left_increment本身就是相对于目标的偏移）
            left_distance_from_target = np.linalg.norm(left_increment)
            right_distance_from_target = np.linalg.norm(right_increment)
            
            # 如果距离目标太远，按比例缩放增量
            if left_distance_from_target > DEMO_WORKSPACE_RADIUS:
                scale_factor = DEMO_WORKSPACE_RADIUS / left_distance_from_target
                left_increment = left_increment * scale_factor
                
            if right_distance_from_target > DEMO_WORKSPACE_RADIUS:
                scale_factor = DEMO_WORKSPACE_RADIUS / right_distance_from_target
                right_increment = right_increment * scale_factor
            
            # 额外的基本安全约束
            # 重新计算绝对位置
            left_abs_pos = DEMO_TARGET_LEFT_POS + left_increment
            right_abs_pos = DEMO_TARGET_RIGHT_POS + right_increment
            
            # 确保不会到达地面以下或过高
            SAFETY_Z_MIN = -0.2   
            SAFETY_Z_MAX = 0.8
            
            if left_abs_pos[2] < SAFETY_Z_MIN:
                left_increment[2] = SAFETY_Z_MIN - DEMO_TARGET_LEFT_POS[2]
            elif left_abs_pos[2] > SAFETY_Z_MAX:
                left_increment[2] = SAFETY_Z_MAX - DEMO_TARGET_LEFT_POS[2]
                
            if right_abs_pos[2] < SAFETY_Z_MIN:
                right_increment[2] = SAFETY_Z_MIN - DEMO_TARGET_RIGHT_POS[2]
            elif right_abs_pos[2] > SAFETY_Z_MAX:
                right_increment[2] = SAFETY_Z_MAX - DEMO_TARGET_RIGHT_POS[2]
            
            if self.debug:
                final_left_pos = DEMO_TARGET_LEFT_POS + left_increment
                final_right_pos = DEMO_TARGET_RIGHT_POS + right_increment
                left_dist_to_target = np.linalg.norm(final_left_pos - DEMO_TARGET_LEFT_POS)
                right_dist_to_target = np.linalg.norm(final_right_pos - DEMO_TARGET_RIGHT_POS)
                print(f"[DEMO CONSTRAINTS] Position control constraints applied:")
                print(f"  Left final pos: [{final_left_pos[0]:.3f}, {final_left_pos[1]:.3f}, {final_left_pos[2]:.3f}]")
                print(f"  Right final pos: [{final_right_pos[0]:.3f}, {final_right_pos[1]:.3f}, {final_right_pos[2]:.3f}]")
                print(f"  Distance to targets - Left: {left_dist_to_target:.3f}m, Right: {right_dist_to_target:.3f}m")
                print(f"  Workspace radius: {DEMO_WORKSPACE_RADIUS:.3f}m")
        else:
            # ========== NORMAL MODE: STRICT TASK-SPECIFIC CONSTRAINTS ==========
            # 计算绝对位置用于约束检查
            left_abs_pos = self.FIXED_LEFT_POS + left_increment
            right_abs_pos = self.FIXED_RIGHT_POS + right_increment
            
            # 约束1: X位置必须在前方 [0, 0.7]
            if left_abs_pos[0] < 0.0:
                left_increment[0] = -self.FIXED_LEFT_POS[0]  # 限制到最小值
            elif left_abs_pos[0] > 0.7:
                left_increment[0] = 0.7 - self.FIXED_LEFT_POS[0]  # 限制到最大值
                
            if right_abs_pos[0] < 0.0:
                right_increment[0] = -self.FIXED_RIGHT_POS[0]
            elif right_abs_pos[0] > 0.7:
                right_increment[0] = 0.7 - self.FIXED_RIGHT_POS[0]
            
            # 约束2: 左手Y位置必须在 [0, 0.65]
            if left_abs_pos[1] < 0.0:
                left_increment[1] = -self.FIXED_LEFT_POS[1]
            elif left_abs_pos[1] > 0.65:
                left_increment[1] = 0.65 - self.FIXED_LEFT_POS[1]
                
            # 约束3: 右手Y位置必须在 [-0.65, 0]
            if right_abs_pos[1] > 0.0:
                right_increment[1] = -self.FIXED_RIGHT_POS[1]
            elif right_abs_pos[1] < -0.65:
                right_increment[1] = -0.65 - self.FIXED_RIGHT_POS[1]
            
            # 约束4: Z位置必须在 [-0.20, 0.65]
            for hand_increment, fixed_pos in [(left_increment, self.FIXED_LEFT_POS), (right_increment, self.FIXED_RIGHT_POS)]:
                abs_z = fixed_pos[2] + hand_increment[2]
                if abs_z < -0.20:
                    hand_increment[2] = -0.20 - fixed_pos[2]
                elif abs_z > 0.65:
                    hand_increment[2] = 0.65 - fixed_pos[2]
        
        return left_increment, right_increment

    def _apply_action_constraints(self, action: np.ndarray) -> np.ndarray:
        """
        Apply constraints to the action to ensure safe and physically meaningful motions.
        
        Updated for incremental control:
        1. Disable robot linear z movement (action[2] = 0)
        2. In Stage 2 (grasp stage): Disable all torso movements (action[0], action[1], action[3] = 0)
        3. Limit incremental values to reasonable ranges
        4. Specific constraints are now handled in _apply_task_specific_constraints
        
        NOTE: When TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD is True, stage-based constraints are bypassed
        to allow direct hand control for joint space learning.
        
        Args:
            action: The original action array (now with increments)
            
        Returns:
            The constrained action array
        """
        global TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD
        constrained_action = action.copy()
        
        # Extract velocity and end-effector actions
        vel_dim = 6 if self.enable_roll_pitch_control else 4
        
        if TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD:
            # ========== DEMO MODE: BYPASS STAGE CONSTRAINTS ==========
            # Allow direct control for joint space learning
            # Only apply basic safety constraints, no stage-based restrictions
            
            # Still disable linear z movement for safety
            constrained_action[0] = 0.0  # Disable linear x movement
            constrained_action[1] = 0.0  # Disable linear y movement
            constrained_action[2] = 0.0  # Disable linear z movement
            constrained_action[3] = 0.0  # Disable angular yaw movement
            
            # Allow hand movements with more relaxed constraints
            ee_action = constrained_action[vel_dim:]
            
            if len(ee_action) >= 6:
                left_increment = ee_action[0:3]
                right_increment = ee_action[3:6]
                
                # Use dedicated DEMO mode increment limits (much larger)
                left_increment = np.clip(left_increment, -DEMO_MAX_INCREMENT_PER_STEP, DEMO_MAX_INCREMENT_PER_STEP)
                right_increment = np.clip(right_increment, -DEMO_MAX_INCREMENT_PER_STEP, DEMO_MAX_INCREMENT_PER_STEP)
                
                # Update the action array
                constrained_action[vel_dim:vel_dim+3] = left_increment
                constrained_action[vel_dim+3:vel_dim+6] = right_increment
                
                if self.debug:
                    print(f"[DEMO CONSTRAINT] Relaxed constraints - only linear z disabled:")
                    print(f"    action[2] (linear z): {constrained_action[2]:.3f}")
                    print(f"  Left hand increment: x={left_increment[0]:.3f}, y={left_increment[1]:.3f}, z={left_increment[2]:.3f}")
                    print(f"  Right hand increment: x={right_increment[0]:.3f}, y={right_increment[1]:.3f}, z={right_increment[2]:.3f}")
                    print(f"  Max increment limit: ±{DEMO_MAX_INCREMENT_PER_STEP:.3f}")
        else:
            # ========== NORMAL MODE: STAGE-BASED CONSTRAINTS ==========
            # Get current stage to determine constraints
            current_stage = self._get_current_stage()
            
            # Stage-based torso movement constraints
            if current_stage == "grasp":
                # STAGE 2: GRASP STAGE - Disable ALL torso movements
                # action[0] - linear x, action[1] - linear y, action[2] - linear z, action[3] - angular yaw
                constrained_action[0] = 0.0  # Disable linear x movement
                constrained_action[1] = 0.0  # Disable linear y movement
                constrained_action[2] = 0.0  # Disable linear z movement
                constrained_action[3] = 0.0  # Disable angular yaw movement
                
                if self.debug:
                    print(f"[STAGE 2 CONSTRAINT] All torso movements disabled during grasp stage")
            else:
                # STAGE 1: APPROACH STAGE - Only disable linear z movement
                constrained_action[2] = 0.0  # Disable linear z movement
            
            ee_action = constrained_action[vel_dim:]
            
            # For incremental control: [left_increment(3), right_increment(3)]
            if len(ee_action) >= 6:
                left_increment = ee_action[0:3]
                right_increment = ee_action[3:6]
                
                # 限制单步增量的范围（基础约束）
                left_increment = np.clip(left_increment, -self.MAX_INCREMENT_PER_STEP * 2, self.MAX_INCREMENT_PER_STEP * 2)
                right_increment = np.clip(right_increment, -self.MAX_INCREMENT_PER_STEP * 2, self.MAX_INCREMENT_PER_STEP * 2)
                
                # Update the action array
                constrained_action[vel_dim:vel_dim+3] = left_increment
                constrained_action[vel_dim+3:vel_dim+6] = right_increment
                
                # Debug output for constraint verification
                if self.debug:
                    print(f"[CONSTRAINT DEBUG] Applied incremental constraints:")
                    if current_stage == "grasp":
                        print(f"  Stage 2 - All torso movements disabled:")
                        print(f"    action[0] (linear x): {constrained_action[0]:.3f}")
                        print(f"    action[1] (linear y): {constrained_action[1]:.3f}")
                        print(f"    action[2] (linear z): {constrained_action[2]:.3f}")
                        print(f"    action[3] (angular yaw): {constrained_action[3]:.3f}")
                    else:
                        print(f"  Stage 1 - Only linear z disabled:")
                        print(f"    action[2] (linear z): {constrained_action[2]:.3f}")
                    print(f"  Left hand increment: x={left_increment[0]:.3f}, y={left_increment[1]:.3f}, z={left_increment[2]:.3f}")
                    print(f"  Right hand increment: x={right_increment[0]:.3f}, y={right_increment[1]:.3f}, z={right_increment[2]:.3f}")
        
        return constrained_action

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
        分阶段奖励函数（修复累积奖励问题版）：
        - 阶段1 (dist_torso_to_box > 0.5): 靠近箱子阶段
        - 阶段2 (dist_torso_to_box <= 0.5): 抓取箱子阶段
        - 修复了奖励累积和终端条件问题
        """
        global TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD

        if not TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD:
            info = {}
            
            # Extract data from observation
            agent_state = obs['agent_pos']
            env_state = obs['environment_state']
            
            # Extract positions based on observation mode
            if self.wbc_observation_enabled:
                # WBC enabled: agent_state has 46 dimensions
                left_eef_pos = agent_state[0:3]
                right_eef_pos = agent_state[7:10]
                arm_joints = agent_state[14:28]  # 14 joint angles
                torso_pos = agent_state[40:43]
                box_pos = env_state[0:3]
                box_orn = env_state[3:7]
            else:
                # WBC disabled: agent_state has 29 dimensions (increased from 23)
                left_eef_pos = agent_state[0:3]           # world frame left eef pos
                right_eef_pos = agent_state[3:6]          # world frame right eef pos
                arm_joints = agent_state[6:20]            # 14 joint angles
                torso_pos = agent_state[20:23]            # robot position
                # Additional base_link end-effector positions (new data)
                base_link_left_eef_pos = agent_state[23:26]   # base_link frame left eef pos
                base_link_right_eef_pos = agent_state[26:29]  # base_link frame right eef pos
                box_pos = env_state[0:3]
                box_orn = None  # No orientation data when WBC is disabled

            # Calculate distances
            dist_left_hand_to_box = np.linalg.norm(left_eef_pos - box_pos)
            dist_right_hand_to_box = np.linalg.norm(right_eef_pos - box_pos)
            dist_torso_to_box = np.linalg.norm(torso_pos - box_pos)

            # Base step penalty to encourage efficiency
            reward = -0.01  # Increased from -0.005 to discourage time-wasting
            
            # Check success conditions
            z_lift = box_pos[2] - self.initial_box_pose['position'][2]
            
            # FIXED: Much stricter box fallen detection and severe penalty
            box_fallen = z_lift < -0.2  # STRICTER: from -0.5 to -0.2
            if box_fallen:
                reward -= 100.0  # SEVERE PENALTY: from -10.0 to -100.0
                terminated = True
                info["box_fallen"] = True
                info["failure_reason"] = "box_dropped"
            else:
                terminated = False
                info["box_fallen"] = False

            # Success conditions
            lift_success = z_lift > 0.15  # Must lift box at least 15cm
            hands_close_success = (dist_left_hand_to_box < 0.3) and (dist_right_hand_to_box < 0.3)
            
            # FIXED: Reasonable timeout penalty - per-step penalty, not cumulative
            timeout_penalty = 0.0
            extreme_delay_penalty = 0.0
            
            if self.episode_step_count >= 150:  # Start gentle penalty
                timeout_penalty = 0.05  # Small fixed penalty per step after 150
                reward -= timeout_penalty
                info["approaching_timeout"] = True
                info["timeout_penalty"] = timeout_penalty
            else:
                info["approaching_timeout"] = False
                info["timeout_penalty"] = 0.0
                
            # More aggressive penalty for extreme delays
            if self.episode_step_count >= 180:
                extreme_delay_penalty = 0.2  # Higher but still reasonable penalty per step
                reward -= extreme_delay_penalty
                info["extreme_delay_penalty"] = extreme_delay_penalty
            else:
                info["extreme_delay_penalty"] = 0.0
            
            # ============= STAGE-BASED REWARD SYSTEM =============
            
            # Determine current stage
            is_approach_stage = dist_torso_to_box > 0.5
            is_grasp_stage = dist_torso_to_box <= 0.5
            
            info["current_stage"] = "approach" if is_approach_stage else "grasp"
            info["dist_torso_to_box"] = dist_torso_to_box
            
            # Default joint reward (used in both stages)
            # Use specific default joint angles in radians
            default_joint_angles_rad = np.array([
                # Left arm (7 joints)
                -0.16152124106884003, 0.015288513153791428, -0.21087729930877686, -1.3677302598953247, -0.009610041975975037, 0.16676270961761475, -0.14105713367462158, 
                # Right arm (7 joints)
                -0.16032356023788452, -0.015272354707121849, 0.21103379130363464, -1.3697106838226318, 0.010835006833076477, -0.16762582957744598, -0.1407204419374466, 
            ])
            
            # Calculate deviation from default joint positions
            joint_deviation = np.abs(arm_joints - default_joint_angles_rad)
            # Mean deviation across all joints (in radians)
            mean_joint_deviation_rad = np.mean(joint_deviation)
            # Convert to degrees for better interpretability
            mean_joint_deviation_deg = np.rad2deg(mean_joint_deviation_rad)
            # Reward for being close to default positions (higher reward for smaller deviation)
            # Use degree-based deviation for more intuitive scaling
            default_joint_reward = np.exp(-0.05 * mean_joint_deviation_deg) * 0.5
            
            if is_approach_stage:
                # ========== STAGE 1: APPROACH STAGE ==========
                
                # 1. Default joint reward (correct weight as requested)
                reward += default_joint_reward * 0.3
                
                # 2. Torso approaching box reward
                if self.last_dist_torso_to_box is not None:
                    torso_distance_change = self.last_dist_torso_to_box - dist_torso_to_box
                    if torso_distance_change > 0:  # Getting closer
                        torso_approach_reward = torso_distance_change * 2.0
                        reward += torso_approach_reward
                        info["torso_approach_reward"] = torso_approach_reward
                    elif torso_distance_change < -0.02:  # Moving away penalty
                        torso_retreat_penalty = max(torso_distance_change * 1.5, -1.0)  # Capped penalty
                        reward += torso_retreat_penalty
                        info["torso_retreat_penalty"] = torso_retreat_penalty
                
                # 3. Basic guidance reward for getting torso close
                torso_proximity_reward = max(0, (2.0 - dist_torso_to_box) * 0.2)  # Reward within 2m
                reward += torso_proximity_reward
                info["torso_proximity_reward"] = torso_proximity_reward
                
                # 4. Progressive approach reward to make success less sparse
                if dist_torso_to_box < 1.0:
                    approach_progress_reward = (1.0 - dist_torso_to_box) * 1.0  # Max 1.0 when very close
                    reward += approach_progress_reward
                    info["approach_progress_reward"] = approach_progress_reward
                
                # No hand-to-box distance rewards in approach stage
                info["stage_focus"] = "torso_approach"
                
            else:
                # ========== STAGE 2: GRASP STAGE ==========
                
                # 1. Default joint reward (maintain comfortable posture)
                reward += default_joint_reward * 0.2
                
                # 2. Final success reward - INCREASED to make success more attractive than time-wasting
                if lift_success and hands_close_success:
                    # Base success reward (INCREASED to dominate over process rewards)
                    base_success_reward = 50.0  # INCREASED from 15.0
                    
                    # Efficiency bonus based on episode steps
                    optimal_steps = 120  # More realistic for 200-step episodes
                    max_efficiency_bonus = 50.0  # INCREASED from 10.0
                    
                    if self.episode_step_count <= optimal_steps:
                        efficiency_bonus = max_efficiency_bonus
                    else:
                        step_penalty = (self.episode_step_count - optimal_steps) * 0.5  # MORE aggressive penalty
                        efficiency_bonus = max(0.0, max_efficiency_bonus - step_penalty)
                    
                    total_success_reward = base_success_reward + efficiency_bonus
                    reward += total_success_reward
                    terminated = True
                    
                    info["base_success_reward"] = base_success_reward
                    info["efficiency_bonus"] = efficiency_bonus
                    info["total_success_reward"] = total_success_reward
                    info["episode_steps"] = self.episode_step_count
                
                else:
                    # Guidance rewards for grasping behavior - FIXED: All rewards now based on IMPROVEMENT
                    
                    # FIXED: Hand-to-box approach rewards - Only reward IMPROVEMENT, not maintaining position
                    if self.last_dist_left_hand_to_box is not None:
                        left_distance_change = self.last_dist_left_hand_to_box - dist_left_hand_to_box
                        if left_distance_change > 0.01:  # Only significant improvement
                            left_hand_reward = left_distance_change * 10.0  # Reward improvement
                            reward += left_hand_reward
                            info["left_hand_approach_reward"] = left_hand_reward
                    else:
                        info["left_hand_approach_reward"] = 0.0
                    
                    if self.last_dist_right_hand_to_box is not None:
                        right_distance_change = self.last_dist_right_hand_to_box - dist_right_hand_to_box
                        if right_distance_change > 0.01:  # Only significant improvement
                            right_hand_reward = right_distance_change * 10.0  # Reward improvement
                            reward += right_hand_reward
                            info["right_hand_approach_reward"] = right_hand_reward
                    else:
                        info["right_hand_approach_reward"] = 0.0
                    
                    # FIXED: Bonus for both hands being close - ONLY once when first achieved
                    if not hasattr(self, 'both_hands_close_achieved'):
                        self.both_hands_close_achieved = False
                    
                    if (dist_left_hand_to_box < 0.4 and dist_right_hand_to_box < 0.4 and 
                        not self.both_hands_close_achieved):
                        both_hands_close_bonus = 5.0  # One-time bonus
                        reward += both_hands_close_bonus
                        self.both_hands_close_achieved = True
                        info["both_hands_close_bonus"] = both_hands_close_bonus
                        info["first_time_both_hands_close"] = True
                    else:
                        info["both_hands_close_bonus"] = 0.0
                        info["first_time_both_hands_close"] = False
                    
                    # Reset the flag if hands move away
                    if (dist_left_hand_to_box > 0.5 or dist_right_hand_to_box > 0.5):
                        self.both_hands_close_achieved = False
                    
                    # FIXED: Box lifting progress reward with ANTI-EXPLOITATION measures
                    if z_lift > 0.02:
                        # FIXED: Prevent exploitation by limiting lift reward accumulation
                        # Only reward meaningful lift progress, not just maintaining position
                        if not hasattr(self, 'max_z_lift_achieved'):
                            self.max_z_lift_achieved = 0.0
                        
                        # Only give lift reward for NEW progress, not maintaining old progress
                        if z_lift > self.max_z_lift_achieved:
                            lift_progress = z_lift - self.max_z_lift_achieved
                            self.max_z_lift_achieved = z_lift
                            
                            if z_lift > 0.15:
                                box_lift_reward = 15.0  # One-time reward for achieving success height
                            elif z_lift > 0.10:
                                box_lift_reward = lift_progress * 50.0  # Reward for significant progress
                            elif z_lift > 0.05:
                                box_lift_reward = lift_progress * 30.0  # Reward for moderate progress
                            else:
                                box_lift_reward = lift_progress * 20.0  # Reward for small progress
                            
                            reward += box_lift_reward
                            info["box_lift_reward"] = box_lift_reward
                            info["lift_progress"] = lift_progress
                        else:
                            # NO reward for just maintaining position - prevents time-wasting
                            info["box_lift_reward"] = 0.0
                            info["maintaining_position"] = True
                    else:
                        # Reset max achievement if box falls below threshold
                        self.max_z_lift_achieved = 0.0
                    
                    # FIXED: Hand symmetry reward - ONLY once when first achieved
                    if not hasattr(self, 'good_symmetry_achieved'):
                        self.good_symmetry_achieved = False
                    
                    if (dist_left_hand_to_box < 0.5 and dist_right_hand_to_box < 0.5 and 
                        not self.good_symmetry_achieved):
                        box_to_left = left_eef_pos - box_pos
                        box_to_right = right_eef_pos - box_pos
                        left_yz = box_to_left[1:]  # y,z components
                        right_yz = box_to_right[1:]  # y,z components
                        symmetry_error = np.linalg.norm(left_yz + right_yz)
                        
                        if symmetry_error < 0.2:  # Good symmetry threshold
                            symmetry_reward = 3.0  # One-time bonus for achieving good symmetry
                            reward += symmetry_reward
                            self.good_symmetry_achieved = True
                            info["symmetry_reward"] = symmetry_reward
                            info["first_time_good_symmetry"] = True
                        else:
                            info["symmetry_reward"] = 0.0
                            info["first_time_good_symmetry"] = False
                    else:
                        info["symmetry_reward"] = 0.0
                        info["first_time_good_symmetry"] = False
                    
                    # Reset symmetry flag if hands move away
                    if (dist_left_hand_to_box > 0.6 or dist_right_hand_to_box > 0.6):
                        self.good_symmetry_achieved = False
                
                info["stage_focus"] = "grasp_manipulation"
            
            # ============= COMMON CONSTRAINTS AND PENALTIES =============
            
            # Position constraints (light penalties)
            position_penalty = 0.0
            
            # Keep hands in positive X (forward) region  
            if left_eef_pos[0] < 0:
                position_penalty += abs(left_eef_pos[0]) * 0.1
            if right_eef_pos[0] < 0:
                position_penalty += abs(right_eef_pos[0]) * 0.1
                
            # Prevent hand crossing
            y_separation = left_eef_pos[1] - right_eef_pos[1]
            if y_separation <= 0:  # Hands crossed
                position_penalty += abs(y_separation) * 0.2
                
            reward -= position_penalty
            info["position_penalty"] = position_penalty
            
            # Velocity smoothness penalty (encourage smooth motion)
            velocity_penalty = 0.0
            if self.last_left_eef_pos is not None and self.last_right_eef_pos is not None:
                left_eef_velocity = np.linalg.norm(left_eef_pos - self.last_left_eef_pos)
                right_eef_velocity = np.linalg.norm(right_eef_pos - self.last_right_eef_pos)
                velocity_penalty = (left_eef_velocity + right_eef_velocity) * 0.02
                velocity_penalty = min(velocity_penalty, 0.3)  # Cap penalty
                reward -= velocity_penalty
                info["velocity_penalty"] = velocity_penalty
            
            # Update position history
            self.last_left_eef_pos = left_eef_pos.copy()
            self.last_right_eef_pos = right_eef_pos.copy()
            self.last_dist_torso_to_box = dist_torso_to_box
            
            # FIXED: Update hand distance tracking for improvement-based rewards
            self.last_dist_left_hand_to_box = dist_left_hand_to_box
            self.last_dist_right_hand_to_box = dist_right_hand_to_box
            
            # Final reward clipping - ADJUSTED for new improvement-based reward scale
            reward = np.clip(reward, -150.0, 120.0)  # ADJUSTED: accommodate success rewards and timeout penalties
            
            # Termination check
            if not box_fallen:
                terminated = lift_success and hands_close_success
            
            # ============= INFO DICTIONARY =============
            info["succeed"] = lift_success and hands_close_success
            info["z_lift"] = z_lift
            info["dist_left_hand_to_box"] = dist_left_hand_to_box
            info["dist_right_hand_to_box"] = dist_right_hand_to_box
            info["default_joint_reward"] = default_joint_reward
            info["mean_joint_deviation"] = mean_joint_deviation_deg
            info["reward_total"] = reward
            info["is_approach_stage"] = is_approach_stage
            info["is_grasp_stage"] = is_grasp_stage
            
            # Stage-specific debug info
            if self.debug:
                stage_name = "APPROACH" if is_approach_stage else "GRASP"
                print(f"Step {self.episode_step_count} [{stage_name}]: dist_torso={dist_torso_to_box:.3f}, z_lift={z_lift:.3f}")
                print(f"  Total reward: {reward:.3f}, default_joint: {default_joint_reward:.3f}")
                print(f"  Joint deviation: {mean_joint_deviation_deg:.3f}, terminated: {terminated}")
                
                if is_approach_stage:
                    print(f"  Focus: Torso approach to box")
                else:
                    print(f"  Focus: Hand manipulation and grasping")
                    print(f"  Hand distances - L: {dist_left_hand_to_box:.3f}, R: {dist_right_hand_to_box:.3f}")
                    if hasattr(self, 'max_z_lift_achieved'):
                        print(f"  Max lift achieved: {self.max_z_lift_achieved:.3f}")
                        
                    # Print achievement flags for debugging
                    hands_close_flag = getattr(self, 'both_hands_close_achieved', False)
                    symmetry_flag = getattr(self, 'good_symmetry_achieved', False)
                    print(f"  Achievement flags - Hands close: {hands_close_flag}, Good symmetry: {symmetry_flag}")
        else:
            # ========== DEMO MODE: END-EFFECTOR POSITION CONTROL ==========
            info = {}
            
            # Extract data from observation
            agent_state = obs['agent_pos']
            env_state = obs['environment_state']
            
            # Extract end-effector positions based on observation mode
            if self.wbc_observation_enabled:
                # WBC enabled: agent_state has 46 dimensions
                left_eef_pos = agent_state[0:3]   # Left end-effector position
                right_eef_pos = agent_state[7:10] # Right end-effector position (skip orientation)
            else:
                # WBC disabled: agent_state has 29 dimensions | 获取相对位置
                left_eef_pos = agent_state[23:26]   # Left end-effector position
                right_eef_pos = agent_state[26:29]  # Right end-effector position
                # base_link_left_eef_pos = agent_state[23:26]   # base_link frame left eef pos
                # base_link_right_eef_pos = agent_state[26:29]  # base_link frame right eef pos
            
            # Calculate distances to target positions
            left_distance = np.linalg.norm(left_eef_pos - DEMO_TARGET_LEFT_POS)
            right_distance = np.linalg.norm(right_eef_pos - DEMO_TARGET_RIGHT_POS)
            mean_distance = (left_distance + right_distance) / 2.0
            
            # 改进的奖励函数设计：对大距离也提供有意义的学习信号
            # 使用混合奖励：线性奖励 + 指数奖励
            
            # 1. 线性距离奖励：对远距离提供基础学习信号
            max_distance = 1.0  # 假设最大可能距离为1m
            left_linear_reward = max(0, (max_distance - left_distance) / max_distance) * 1.0
            right_linear_reward = max(0, (max_distance - right_distance) / max_distance) * 1.0
            
            # 2. 指数距离奖励：对近距离提供强化信号
            left_exp_reward = np.exp(-3.0 * left_distance) * 2.0   # 降低敏感性从5.0到3.0
            right_exp_reward = np.exp(-3.0 * right_distance) * 2.0
            
            # 3. 组合两种奖励
            left_position_reward = left_linear_reward + left_exp_reward
            right_position_reward = right_linear_reward + right_exp_reward
            total_position_reward = left_position_reward + right_position_reward
            
            # 4. 轴向精度奖励：鼓励各轴精确对齐
            left_axis_deviations = np.abs(left_eef_pos - DEMO_TARGET_LEFT_POS)
            right_axis_deviations = np.abs(right_eef_pos - DEMO_TARGET_RIGHT_POS)
            left_axis_rewards = np.exp(-8.0 * left_axis_deviations) * 0.3  # 降低敏感性并增加权重
            right_axis_rewards = np.exp(-8.0 * right_axis_deviations) * 0.3
            dense_axis_reward = np.sum(left_axis_rewards) + np.sum(right_axis_rewards)
            
            # 5. 组合所有奖励
            reward = total_position_reward + dense_axis_reward
            
            # Small step penalty to encourage efficiency
            reward -= 0.001
            
            # Success condition: when both hands are close enough to target positions
            success_threshold_m = 0.02  # Within 2cm is considered success
            left_close = left_distance < success_threshold_m
            right_close = right_distance < success_threshold_m
            both_hands_close = left_close and right_close
            
            # Termination conditions
            if both_hands_close:
                # Success reward and terminate
                reward += 15.0  # Success bonus (increased for position control)
                terminated = True
                info["success"] = True
                info["success_reason"] = "both_hands_at_target_positions"
            elif self.episode_step_count >= 200:
                # Timeout - terminate but no success
                terminated = True
                info["success"] = False
                info["success_reason"] = "timeout"
            else:
                terminated = False
                info["success"] = False
            
            # Progress tracking reward - reward improvement over time
            if not hasattr(self, 'best_mean_distance'):
                self.best_mean_distance = float('inf')
            
            if mean_distance < self.best_mean_distance:
                improvement = self.best_mean_distance - mean_distance
                improvement_reward = improvement * 10.0  # Reward improvement (scaled for meters)
                reward += improvement_reward
                self.best_mean_distance = mean_distance
                info["improvement_reward"] = improvement_reward
                info["new_best_achieved"] = True
            else:
                info["improvement_reward"] = 0.0
                info["new_best_achieved"] = False
            
            # Bonus for individual hand achievements
            if not hasattr(self, 'left_hand_achieved'):
                self.left_hand_achieved = False
            if not hasattr(self, 'right_hand_achieved'):
                self.right_hand_achieved = False
            
            # One-time bonus when each hand first reaches target
            if left_close and not self.left_hand_achieved:
                reward += 5.0
                self.left_hand_achieved = True
                info["left_hand_bonus"] = 5.0
            else:
                info["left_hand_bonus"] = 0.0
                
            if right_close and not self.right_hand_achieved:
                reward += 5.0
                self.right_hand_achieved = True
                info["right_hand_bonus"] = 5.0
            else:
                info["right_hand_bonus"] = 0.0
            
            # Reset achievement flags if hands move away
            if left_distance > success_threshold_m * 1.5:
                self.left_hand_achieved = False
            if right_distance > success_threshold_m * 1.5:
                self.right_hand_achieved = False
            
            # Clip reward to reasonable range
            reward = np.clip(reward, -5.0, 30.0)  # Increased upper bound for position control
            
            # Info dictionary for debugging and monitoring
            info["left_distance_m"] = left_distance
            info["right_distance_m"] = right_distance
            info["mean_distance_m"] = mean_distance
            info["left_position_reward"] = left_position_reward
            info["right_position_reward"] = right_position_reward
            info["total_position_reward"] = total_position_reward
            info["dense_axis_reward"] = dense_axis_reward
            info["best_distance_so_far"] = self.best_mean_distance
            info["left_hand_close"] = left_close
            info["right_hand_close"] = right_close
            info["both_hands_close"] = both_hands_close
            info["success_threshold_m"] = success_threshold_m
            info["reward_total"] = reward
            info["episode_steps"] = self.episode_step_count
            info["mode"] = "demo_eef_position_control"
            
            # 添加详细的奖励组件信息用于调试
            info["left_linear_reward"] = left_linear_reward
            info["right_linear_reward"] = right_linear_reward
            info["left_exp_reward"] = left_exp_reward
            info["right_exp_reward"] = right_exp_reward
            info["current_increment_left"] = np.linalg.norm(self.current_left_increment)
            info["current_increment_right"] = np.linalg.norm(self.current_right_increment)
            
            # # Current positions for detailed monitoring (as numpy arrays, not lists)
            # info["current_left_pos"] = left_eef_pos.copy()
            # info["current_right_pos"] = right_eef_pos.copy()
            # info["target_left_pos"] = DEMO_TARGET_LEFT_POS.copy()
            # info["target_right_pos"] = DEMO_TARGET_RIGHT_POS.copy()
            
            # Individual axis deviations for detailed monitoring
            axis_names = ['x', 'y', 'z']
            for i, axis in enumerate(axis_names):
                info[f"left_{axis}_deviation_m"] = left_axis_deviations[i]
                info[f"right_{axis}_deviation_m"] = right_axis_deviations[i]
            
            # Debug output
            if self.debug:
                print(f"[DEMO MODE] Step {self.episode_step_count}: Mean distance: {mean_distance:.4f}m")
                print(f"  Left distance: {left_distance:.4f}m, Right distance: {right_distance:.4f}m")
                print(f"  Reward breakdown - Linear: L={left_linear_reward:.3f}, R={right_linear_reward:.3f}")
                print(f"                    Exp: L={left_exp_reward:.3f}, R={right_exp_reward:.3f}")
                print(f"                    Axis: {dense_axis_reward:.3f}, Total: {reward:.3f}")
                print(f"  Best distance so far: {self.best_mean_distance:.4f}m")
                print(f"  Current increments - Left: {np.linalg.norm(self.current_left_increment):.4f}m, Right: {np.linalg.norm(self.current_right_increment):.4f}m")
                print(f"  Hands close - Left: {left_close}, Right: {right_close}, Both: {both_hands_close}")
                print(f"  Terminated: {terminated}")
                
                # Show current vs target positions
                print(f"  Current Left pos:  [{left_eef_pos[0]:.4f}, {left_eef_pos[1]:.4f}, {left_eef_pos[2]:.4f}]")
                print(f"  Target Left pos:   [{DEMO_TARGET_LEFT_POS[0]:.4f}, {DEMO_TARGET_LEFT_POS[1]:.4f}, {DEMO_TARGET_LEFT_POS[2]:.4f}]")
                print(f"  Current Right pos: [{right_eef_pos[0]:.4f}, {right_eef_pos[1]:.4f}, {right_eef_pos[2]:.4f}]")
                print(f"  Target Right pos:  [{DEMO_TARGET_RIGHT_POS[0]:.4f}, {DEMO_TARGET_RIGHT_POS[1]:.4f}, {DEMO_TARGET_RIGHT_POS[2]:.4f}]")
                
                # Show worst deviating axis
                all_deviations = np.concatenate([left_axis_deviations, right_axis_deviations])
                worst_axis_idx = np.argmax(all_deviations)
                axis_names = ['x', 'y', 'z']
                if worst_axis_idx < 3:
                    worst_hand = "Left"
                    worst_axis = axis_names[worst_axis_idx]
                else:
                    worst_hand = "Right"
                    worst_axis = axis_names[worst_axis_idx - 3]
                worst_deviation = all_deviations[worst_axis_idx]
                print(f"  Worst deviation: {worst_hand} {worst_axis} ({worst_deviation:.4f}m)")
        
        return reward, terminated, info


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Implement the step logic for the Kuavo robot.
        
        Args:
            action: The action to execute
        """
        # Increment step counter for efficiency reward calculation
        self.episode_step_count += 1
        
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
        
        # Reset end-effector position history for velocity penalty
        self.last_left_eef_pos = None
        self.last_right_eef_pos = None
        
        # Reset trajectory tracking variables
        self.last_dist_left_hand_to_box = None
        self.last_dist_right_hand_to_box = None
        self.last_dist_torso_to_box = None
        
        # Reset consecutive approach counters
        self.consecutive_approach_steps_left = 0
        self.consecutive_approach_steps_right = 0
        self.consecutive_approach_steps_torso = 0
        
        # Clear distance change history
        self.distance_change_history_left.clear()
        self.distance_change_history_right.clear()
        
        # Reset step counter for efficiency reward
        self.episode_step_count = 0
        
        # FIXED: Reset lift progress tracking to prevent cross-episode exploitation
        if hasattr(self, 'max_z_lift_achieved'):
            self.max_z_lift_achieved = 0.0
            
        # FIXED: Reset achievement flags to prevent cross-episode exploitation
        if hasattr(self, 'both_hands_close_achieved'):
            self.both_hands_close_achieved = False
        if hasattr(self, 'good_symmetry_achieved'):
            self.good_symmetry_achieved = False

        # Reset incremental control state
        self.current_left_increment.fill(0.0)
        self.current_right_increment.fill(0.0)
        self.last_left_increment.fill(0.0)
        self.last_right_increment.fill(0.0)
        
        # Reset demo mode progress tracking
        global TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD
        if TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD:
            # Reset end-effector position control tracking
            if hasattr(self, 'best_mean_distance'):
                self.best_mean_distance = float('inf')
            if hasattr(self, 'left_hand_achieved'):
                self.left_hand_achieved = False
            if hasattr(self, 'right_hand_achieved'):
                self.right_hand_achieved = False
        else:
            # Reset joint space tracking (if ever used)
            if hasattr(self, 'best_joint_deviation'):
                self.best_joint_deviation = float('inf')
        
        if self.debug:
            rospy.loginfo("reset - Incremental control state reset to zero")
            if TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD:
                rospy.loginfo("reset - Demo mode: end-effector position control")
                rospy.loginfo(f"  Target left position: {DEMO_TARGET_LEFT_POS}")
                rospy.loginfo(f"  Target right position: {DEMO_TARGET_RIGHT_POS}")
                
                # 显示初始机器人位置用于诊断坐标系问题
                if hasattr(self, 'latest_obs') and self.latest_obs is not None:
                    agent_state = obs_stable['agent_pos']
                    if self.wbc_observation_enabled:
                        initial_left_pos = agent_state[0:3]
                        initial_right_pos = agent_state[7:10]
                    else:
                        initial_left_pos = agent_state[23:26]
                        initial_right_pos = agent_state[26:29]
                    
                    rospy.loginfo(f"  Initial left pos:  [{initial_left_pos[0]:.4f}, {initial_left_pos[1]:.4f}, {initial_left_pos[2]:.4f}]")
                    rospy.loginfo(f"  Initial right pos: [{initial_right_pos[0]:.4f}, {initial_right_pos[1]:.4f}, {initial_right_pos[2]:.4f}]")
                    
                    # 计算初始偏差
                    left_initial_error = np.linalg.norm(initial_left_pos - DEMO_TARGET_LEFT_POS)
                    right_initial_error = np.linalg.norm(initial_right_pos - DEMO_TARGET_RIGHT_POS)
                    rospy.loginfo(f"  Initial errors - Left: {left_initial_error:.4f}m, Right: {right_initial_error:.4f}m")
                    
                    if left_initial_error > 0.1 or right_initial_error > 0.1:
                        rospy.logwarn("  WARNING: Large initial position error detected!")
                        rospy.logwarn("  This suggests potential coordinate system mismatch or robot initialization issues.")

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
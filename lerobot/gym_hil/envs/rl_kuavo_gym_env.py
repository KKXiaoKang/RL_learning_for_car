from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces
import rospy
import message_filters
import threading
import os
import xml.etree.ElementTree as ET

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
import time

"""
# 躯干 2 + 14joint = 16 action dim 16控制 | True
# 躯干 2 + 6pose = 8 action dim 8控制 | False
"""
TEST_DEMO_USE_ACTION_16_DIM = False 
USE_CMD_VEL = False

IF_USE_ZERO_OBS_FLAG = False # 是否使用0观测
IF_USE_ARM_MPC_CONTROL = False # 是否使用运动学mpc | ik作为末端控制手段
LEARN_TARGET_EEF_POSE_TARGET = True # 是否使用目标末端位置作为学习目标

# 增加DEMO模式的动作尺度常量
DEMO_MAX_INCREMENT_PER_STEP = 0.02  # DEMO模式下每步最大2cm增量（精细控制）
DEMO_MAX_INCREMENT_RANGE = 0.4     # DEMO模式下最大累积增量范围40cm

# 手肘关节配置
DEMO_LEFT_ELBOW_POS = np.array([0.0178026345146559, 0.3004180715613648, 0.19417275957965042])
DEMO_RIGHT_ELBOW_POS = np.array([0.0178026345146559, -0.3004180715613648, 0.19417275957965042])

# Target Key-Points
DEMO_TARGET_LEFT_POS_WORLD = np.array([0.49856212735176086, 0.22971099615097046, 0.9128270149230957])
DEMO_TARGET_LEFT_QUAT = np.array([0.0, -0.70711, 0.0, 0.70711])
DEMO_TARGET_RIGHT_POS_WORLD = np.array([0.49856212735176086, -0.22971099615097046, 0.9128270149230957])
DEMO_TARGET_RIGHT_QUAT = np.array([0.0, -0.70711, 0.0, 0.70711])
# normal box pose: 0.4083724915981293 -0.009208906441926956 0.9235677719116211
DEMO_TARGET_BOX_POS_WORLD = np.array([0.4000142514705658, 0.0020876736380159855, 1.1273181653022766]) 

# Joint control mode target joint angles (in radians)
DEMO_TARGET_LEFT_JOINT_ANGLES = np.array([
    -0.0974561870098114, -0.3386945128440857, 0.14182303845882416, 
    -1.5295206308364868, -0.35505950450897217, -0.06419740617275238, 0.057071615010499954, 
])

DEMO_TARGET_RIGHT_JOINT_ANGLES = np.array([
    -0.10812032222747803, 0.33889538049697876, -0.16906462609767914, 
    -1.522423505783081, 0.37139785289764404, 0.12298937886953354, 0.04463687911629677,
])

# Combined target joint angles for convenience (left + right)
DEMO_TARGET_ALL_JOINT_ANGLES = np.concatenate([DEMO_TARGET_LEFT_JOINT_ANGLES, DEMO_TARGET_RIGHT_JOINT_ANGLES])

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
    
    # End-effector position constraints (absolute world coordinates)
    # 双手x范围: [0.2, 0.5]
    # 左手y范围: [0.0, 0.5], 右手y范围: [-0.5, 0.0]  
    # 双手z范围: [0.0, 0.2]
    EEF_POS_LIMITS = {
        'x_min': 0.3, 'x_max': 0.6,
        'left_y_min': 0.0, 'left_y_max': 0.5,
        'right_y_min': -0.5, 'right_y_max': 0.0,
        'z_min': 0.0, 'z_max': 0.3
    }

    def __init__(self, debug: bool = False, image_size=(224, 224), enable_roll_pitch_control: bool = False, 
                 vel_smoothing_factor: float = 0.3, arm_smoothing_factor: float = 0.4, 
                 wbc_observation_enabled: bool = False, action_dim: int = None, image_obs: bool = True,
                 render_mode: str = None, use_gripper: bool = True, gripper_penalty: float = 0.0):
        # Store initialization parameters
        self.image_obs = image_obs
        self.render_mode = render_mode  
        self.use_gripper = use_gripper
        self.gripper_penalty = gripper_penalty
        
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

        # 添加当前末端执行器位置状态跟踪（用于增量控制）
        self.current_left_pos = np.array([0.3178026345146559, 0.4004180715613648, -0.019417275957965042], dtype=np.float32)
        self.current_right_pos = np.array([0.3178026345146559, -0.4004180715613648, -0.019417275957965042], dtype=np.float32)
        
        # 增量控制参数
        self.INCREMENT_SCALE = 0.01  # 将action[-1,1]缩放到±0.01m的增量范围

        # Call the base class constructor to set up the node and observation buffer
        super().__init__()
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

        """
            vel_dim 
        """
        if self.enable_roll_pitch_control:
            self.vel_dim = 6
            self.vel_action_scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # m/s and rad/s
        else:
            self.vel_dim = 4
            self.vel_action_scale = np.array([1.0, 1.0, 1.0, 1.0])  # m/s and rad/s x y z yaw
        
        if TEST_DEMO_USE_ACTION_16_DIM:
            self.vel_dim = 2
            self.vel_action_scale = np.array([1.0, 1.0])  # m/s and rad/s 控制x和yaw
        else:
            if USE_CMD_VEL:
                self.vel_dim = 2
                self.vel_action_scale = np.array([1.0, 1.0])  # m/s and rad/s 控制x和yaw
            else:
                self.vel_dim = 0
                self.vel_action_scale = np.array([1.0, 1.0])  # m/s and rad/s 控制x和yaw
        
        """
            action_dim = vel_dim + arm_dim
        """
        # Use provided action_dim if specified, otherwise use default calculation
        if TEST_DEMO_USE_ACTION_16_DIM:
            # vel_dim 2 + arm angle 7 + 7 
            self.arm_dim = 14
            self.action_dim = self.vel_dim + self.arm_dim  # 2 + 7 + 7 = 16
        else:
            # vel_dim 2 + eef pose 3 + 3
            if USE_CMD_VEL:
                self.arm_dim = 6
                self.action_dim = self.vel_dim + self.arm_dim # vel_dim 2 + eef pose 3 + 3 = 8
            else:
                self.arm_dim = 6
                self.action_dim = self.arm_dim # ef pose 3 + 3 = 6
        
        # elif self.wbc_observation_enabled:
        #     self.arm_dim = 14 # 关节 joint space
        #     self.action_dim = self.vel_dim + self.arm_dim # 4 + 14 = 18
        # else:
        #     # Default behavior: 14 for arm joints
        #     self.arm_dim = 6 # 末端 eef position
        #     self.action_dim = self.vel_dim + self.arm_dim # 4 + 6 = 10

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
            
            # FIXME
            self.arm_joint_centers = np.zeros(self.arm_dim)
            self.arm_joint_scales = np.full(self.arm_dim, np.deg2rad(10.0))

            print(f" arm_joint_centers: {self.arm_joint_centers}")
            print(f" arm_joint_scales: {self.arm_joint_scales}")
            print(f" arm_joint_center rad2deg: {np.rad2deg(self.arm_joint_centers)}") 

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
        
    def change_mobile_ctrl_mode(self, mode: int):
        # print(f"change_mobile_ctrl_mode: {mode}")
        mobile_manipulator_service_name = "/mobile_manipulator_mpc_control"
        try:
            rospy.wait_for_service(mobile_manipulator_service_name)
            changeHandTrackingMode_srv = rospy.ServiceProxy(mobile_manipulator_service_name, changeArmCtrlMode)
            changeHandTrackingMode_srv(mode)
        except rospy.ROSException:
            rospy.logerr(f"Service {mobile_manipulator_service_name} not available")

    def _scale_action_to_eef_positions(self, ee_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将归一化的action [-1, 1] scale到指定的end-effector位置范围
        
        Args:
            ee_action: 归一化的末端执行器动作 [left_x, left_y, left_z, right_x, right_y, right_z]
        
        Returns:
            left_pos, right_pos: 缩放后的绝对世界坐标位置
        """
        if len(ee_action) < 6:
            rospy.logwarn(f"ee_action length {len(ee_action)} < 6, padding with zeros")
            padded_action = np.zeros(6)
            padded_action[:len(ee_action)] = ee_action
            ee_action = padded_action
        
        # 提取左右手的动作
        left_action = ee_action[0:3]  # [x, y, z]
        right_action = ee_action[3:6]  # [x, y, z]
        
        # Scale左手位置: action [-1,1] -> world coordinates
        left_x = (left_action[0] + 1) / 2 * (self.EEF_POS_LIMITS['x_max'] - self.EEF_POS_LIMITS['x_min']) + self.EEF_POS_LIMITS['x_min']
        left_y = (left_action[1] + 1) / 2 * (self.EEF_POS_LIMITS['left_y_max'] - self.EEF_POS_LIMITS['left_y_min']) + self.EEF_POS_LIMITS['left_y_min']
        left_z = (left_action[2] + 1) / 2 * (self.EEF_POS_LIMITS['z_max'] - self.EEF_POS_LIMITS['z_min']) + self.EEF_POS_LIMITS['z_min']
        
        # Scale右手位置: action [-1,1] -> world coordinates  
        right_x = (right_action[0] + 1) / 2 * (self.EEF_POS_LIMITS['x_max'] - self.EEF_POS_LIMITS['x_min']) + self.EEF_POS_LIMITS['x_min']
        right_y = (right_action[1] + 1) / 2 * (self.EEF_POS_LIMITS['right_y_max'] - self.EEF_POS_LIMITS['right_y_min']) + self.EEF_POS_LIMITS['right_y_min']
        right_z = (right_action[2] + 1) / 2 * (self.EEF_POS_LIMITS['z_max'] - self.EEF_POS_LIMITS['z_min']) + self.EEF_POS_LIMITS['z_min']
        
        left_pos = np.array([left_x, left_y, left_z], dtype=np.float32)
        right_pos = np.array([right_x, right_y, right_z], dtype=np.float32)
        
        if self.debug:
            print(f"[EEF SCALING] Action: {ee_action[:6]}")
            print(f"[EEF SCALING] Left pos: {left_pos}, Right pos: {right_pos}")
        
        return left_pos, right_pos

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
                if TEST_DEMO_USE_ACTION_16_DIM:
                    # Joint control mode: convert joint angles directly
                    arm_positions_deg = np.array(self._latest_vr_arm_traj.position[:self.arm_dim], dtype=np.float32)
                    arm_positions_rad = np.deg2rad(arm_positions_deg)
                    arm_action = (arm_positions_rad - self.arm_joint_centers) / self.arm_joint_scales
                    arm_action = np.clip(arm_action, -1.0, 1.0)
                    action[4:4+self.arm_dim] = arm_action
                    
                    if self.debug:
                        print(f"[VR JOINT CONTROL] VR joint input converted to action (mean: {np.mean(np.abs(arm_action)):.3f})")
                elif self.wbc_observation_enabled:
                    # WBC mode: convert joint angles to increments (legacy logic)
                    arm_positions_deg = np.array(self._latest_vr_arm_traj.position[:self.arm_dim], dtype=np.float32)
                    arm_positions_rad = np.deg2rad(arm_positions_deg)
                    arm_action = (arm_positions_rad - self.arm_joint_centers) / self.arm_joint_scales
                    arm_action = np.clip(arm_action, -1.0, 1.0)
                    action[4:4+self.arm_dim] = arm_action
                else:
                    # Position mode: convert to increments based on appropriate reference positions
                    left_increment = np.zeros(3, dtype=np.float32)
                    right_increment = np.zeros(3, dtype=np.float32)
                    
                    if len(self._latest_vr_arm_traj.position) >= 6:
                        # Assume VR sends [left_x, left_y, left_z, right_x, right_y, right_z]
                        vr_left_pos = np.array(self._latest_vr_arm_traj.position[0:3], dtype=np.float32)
                        vr_right_pos = np.array(self._latest_vr_arm_traj.position[3:6], dtype=np.float32)
                        
                        # Choose appropriate reference positions and increment limits based on mode
                        # Normal mode: use fixed positions and normal limits
                        left_increment = vr_left_pos - self.FIXED_LEFT_POS
                        right_increment = vr_right_pos - self.FIXED_RIGHT_POS
                        # Limit increments to normal ranges
                        left_increment = np.clip(left_increment, -self.MAX_INCREMENT_RANGE, self.MAX_INCREMENT_RANGE)
                        right_increment = np.clip(right_increment, -self.MAX_INCREMENT_RANGE, self.MAX_INCREMENT_RANGE)
                        
                        # Only debug non-zero increments to reduce spam
                        if self.debug and (np.linalg.norm(left_increment) > 0.001 or np.linalg.norm(right_increment) > 0.001):
                            print(f"[VR POSITION] Non-zero increments - Left: {np.linalg.norm(left_increment):.3f}m, Right: {np.linalg.norm(right_increment):.3f}m")
                    
                    action[4:7] = left_increment
                    action[7:10] = right_increment
            
            return action



    def _setup_ros_communication(self):
        """
        Implement this method to set up ROS publishers, subscribers,
        and service clients specific to the Kuavo robot.
        """
        global IF_USE_ARM_MPC_CONTROL
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Replace the arm_traj_pub with the new publisher
        if IF_USE_ARM_MPC_CONTROL:
            self.ee_pose_pub = rospy.Publisher('/mm/two_arm_hand_pose_cmd', twoArmHandPoseCmd, queue_size=10)
        else:
            self.ee_pose_pub = rospy.Publisher('/ik/two_arm_hand_pose_cmd', twoArmHandPoseCmd, queue_size=10)
        
        # kuavo_arm_traj pub
        self.robot_arm_traj_pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=10)

        # Service Client
        self.reset_client = rospy.ServiceProxy('/isaac_lab_reset_scene', resetIsaaclab)

        # # Subscribers for headerless topics that are not synchronized
        # if self.wbc_observation_enabled:
        #     rospy.Subscriber('/state_estimate/imu_data_filtered/angularVel', Float64MultiArray, self._ang_vel_callback)
        #     rospy.Subscriber('/state_estimate/imu_data_filtered/linearAccel', Float64MultiArray, self._lin_accel_callback)
        
        # # Conditionally subscribe to WBC or robot pose based on flag
        # if self.wbc_observation_enabled:
        #     rospy.Subscriber('/humanoid_wbc_observation', mpc_observation, self._wbc_callback)
        #     if self.debug:
        #         rospy.loginfo("WBC observation enabled - subscribing to /humanoid_wbc_observation")
        
        if not self.wbc_observation_enabled:
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

    def pub_control_robot_arm_traj(self, joint_q: list)->bool:
        try:
            msg = JointState()
            msg.name = ['zarm_l1_joint', 'zarm_l2_joint', 'zarm_l3_joint', 'zarm_l4_joint', 'zarm_l5_joint', 'zarm_l6_joint', 'zarm_l7_joint',
                        'zarm_r1_joint', 'zarm_r2_joint', 'zarm_r3_joint', 'zarm_r4_joint', 'zarm_r5_joint', 'zarm_r6_joint', 'zarm_r7_joint']
            msg.header.stamp = rospy.Time.now()
            msg.position = (180.0 / np.pi * np.array(joint_q)).tolist()
            print(f"publish robot arm traj: {msg.position}")
            self.robot_arm_traj_pub.publish(msg)
            return True
        except Exception as e:
            print(f"publish robot arm traj: {e}")
        return False

    def pub_control_robot_arm_traj_deg(self,joint_q: list)->bool:
        try:
            msg = JointState()
            msg.name = ['zarm_l1_joint', 'zarm_l2_joint', 'zarm_l3_joint', 'zarm_l4_joint', 'zarm_l5_joint', 'zarm_l6_joint', 'zarm_l7_joint',
                        'zarm_r1_joint', 'zarm_r2_joint', 'zarm_r3_joint', 'zarm_r4_joint', 'zarm_r5_joint', 'zarm_r6_joint', 'zarm_r7_joint']
            msg.header.stamp = rospy.Time.now()
            msg.position = np.array(joint_q).tolist()
            self.robot_arm_traj_pub.publish(msg)
            return True
        except Exception as e:
            print(f"publish robot arm traj: {e}")
        return False

    def _obs_callback(self, left_eef, right_eef, image, sensors, box_real):
        """Synchronously handles incoming observation messages and populates the observation buffer."""
        # Retrieve the latest data from headerless topics
        if self.wbc_observation_enabled:
            with self.ang_vel_lock:
                ang_vel = self.latest_ang_vel
            with self.lin_accel_lock:
                lin_accel = self.latest_lin_accel
        else:
            # When WBC is disabled, use dummy IMU data
            ang_vel = type('dummy', (), {'data': [0.0, 0.0, 0.0]})()
            lin_accel = type('dummy', (), {'data': [0.0, 0.0, 0.0]})()
            
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

        # Wait until all data sources are available (only check IMU data if WBC is enabled)
        if self.wbc_observation_enabled and (ang_vel is None or lin_accel is None):
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
                # # Process image data
                # cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
                # 创建224x224x3的零填充图像（BGR格式）
                cv_image = np.zeros((224, 224, 3), dtype=np.uint8)

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
                        base_link_left_eef_pos, base_link_right_eef_pos,
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

            except Exception as e:
                rospy.logerr(f"Error in observation callback: {e}")

    def _send_action(self, action: np.ndarray, episode_step_count: int):
        """
        Implement this method to publish an action to the Kuavo robot.
        
        Args:
            action: The action array to send
        """
        global TEST_DEMO_USE_ACTION_16_DIM
        if not self._should_publish_action:
            # During VR intervention, don't publish actions - VR system handles control
            return
        
        # De-normalize and publish cmd_vel
        if TEST_DEMO_USE_ACTION_16_DIM:
            """
                2 + 7 + 7 = 16
            """
            twist_cmd = Twist()
            vel_action = action[:self.vel_dim] * self.vel_action_scale
            twist_cmd.linear.x = vel_action[0]
            twist_cmd.linear.y = 0.0 
            twist_cmd.linear.z = 0.0 
            twist_cmd.angular.x = 0.0
            twist_cmd.angular.y = 0.0
            twist_cmd.angular.z = vel_action[1]
            ee_action = action[self.vel_dim:]
            print(" ============================================================ ")
            print(" ==================== step begin ============================")
            print( " ==============  send_action | action: ", action)
            print(" ================ send_action | len action: ", len(action))
            print(" ================ send_action | vel_action: ", vel_action)
            print(" ================ send_action | ee_action: ", ee_action)
            # vel pub
            self.cmd_vel_pub.publish(twist_cmd) 
            # joint pub
            self._publish_joint_control_arm_poses(ee_action)
        else:
            """
                2 + 3 + 3 = 8 (增量控制模式)
                action[0:2]: 底盘速度控制 (x, yaw)
                action[2:5]: 左手位置增量 (x, y, z) - 每步±0.01m
                action[5:8]: 右手位置增量 (x, y, z) - 每步±0.01m
            """
            if USE_CMD_VEL:
                twist_cmd = Twist()
                vel_action = action[:self.vel_dim] * self.vel_action_scale
                twist_cmd.linear.x = vel_action[0]
                twist_cmd.linear.y = 0.0 
                twist_cmd.linear.z = 0.0 
                twist_cmd.angular.x = 0.0
                twist_cmd.angular.y = 0.0
                twist_cmd.angular.z = vel_action[1]
                ee_action = action[self.vel_dim:]
                print(" ============================================================ ")
                print(" ==================== vel eef step begin ============================")
                print( " ==============  send_action | action: ", action)
                print(" ================ send_action | len action: ", len(action))
                print(" ================ send_action | vel_action: ", vel_action)
                print(" ================ send_action | ee_action (increments): ", ee_action)
                # vel pub
                self.cmd_vel_pub.publish(twist_cmd) 
                # eef pub (增量控制)
                self._publish_action_based_arm_poses(ee_action)
            else:
                ee_action = action
                print(" ============================================================ ")
                print(" ==================== only eef step begin ============================")
                print( " ==============  send_action | action: ", action)
                print(" ================ send_action | len action: ", len(action))
                print(" ================ send_action | ee_action (increments): ", ee_action)
                # eef pub (增量控制)
                self._publish_action_based_arm_poses(ee_action)

    def _publish_fixed_arm_poses(self):
        """
        Publish fixed arm poses for the approach stage.
        Also updates the current position state for incremental control.
        """
        global DEMO_LEFT_ELBOW_POS
        global DEMO_RIGHT_ELBOW_POS

        if IF_USE_ARM_MPC_CONTROL:
            self.change_mobile_ctrl_mode(IncrementalMpcCtrlMode.ArmOnly.value)
            print( "=============== change_mobile_ctrl_mode to ArmOnly ================")
        
        # 使用初始位置作为固定姿态
        left_pos = np.array([0.3178026345146559, 0.4004180715613648, -0.019417275957965042])
        left_quat = np.array([0.0, -0.70711, 0.0, 0.70711])
        right_pos = np.array([0.3178026345146559, -0.4004180715613648, -0.019417275957965042])
        right_quat = np.array([0.0, -0.70711, 0.0, 0.70711])
        
        # 更新当前位置状态以匹配发布的位置
        self.current_left_pos = left_pos.copy()
        self.current_right_pos = right_pos.copy()
        
        left_elbow_pos = DEMO_LEFT_ELBOW_POS
        right_elbow_pos = DEMO_RIGHT_ELBOW_POS

        msg = twoArmHandPoseCmd()
        msg.hand_poses.left_pose.pos_xyz = left_pos.tolist()
        msg.hand_poses.left_pose.quat_xyzw = left_quat.tolist()
        msg.hand_poses.left_pose.elbow_pos_xyz = left_elbow_pos # left_elbow_pos.tolist()

        msg.hand_poses.right_pose.pos_xyz = right_pos.tolist()
        msg.hand_poses.right_pose.quat_xyzw = right_quat.tolist()
        msg.hand_poses.right_pose.elbow_pos_xyz = right_elbow_pos # right_elbow_pos.tolist()
        
        # Set default IK params
        msg.use_custom_ik_param = False
        if not IF_USE_ARM_MPC_CONTROL:
            msg.joint_angles_as_q0 = True
        else:
            msg.joint_angles_as_q0 = False
        
        msg.ik_param = ikSolveParam()
        msg.frame = 3  # VR Frame
        self.ee_pose_pub.publish(msg)
        
        if self.debug:
            print(f"[FIXED POSES] Set initial positions - Left: {left_pos}, Right: {right_pos}")

    def _publish_action_based_arm_poses(self, ee_action: np.ndarray):
        """
        Publish arm poses based on the action input using incremental control.
        
        Action mapping:
        - action[0:3]: left hand x,y,z position increments (scaled from [-1,1] to ±0.01m)
        - action[3:6]: right hand x,y,z position increments (scaled from [-1,1] to ±0.01m)
        
        Args:
            ee_action: The arm portion of the action array (normalized [-1,1])
        """
        global DEMO_LEFT_ELBOW_POS
        global DEMO_RIGHT_ELBOW_POS
        
        if IF_USE_ARM_MPC_CONTROL:
            self.change_mobile_ctrl_mode(IncrementalMpcCtrlMode.ArmOnly.value)
            print( "=============== change_mobile_ctrl_mode to ArmOnly ================")
        
        # 确保action长度足够
        if len(ee_action) < 6:
            rospy.logwarn(f"ee_action length {len(ee_action)} < 6, padding with zeros")
            padded_action = np.zeros(6)
            padded_action[:len(ee_action)] = ee_action
            ee_action = padded_action
        
        # 提取左右手的增量action
        left_increment_action = ee_action[0:3]  # [x, y, z] 增量
        right_increment_action = ee_action[3:6]  # [x, y, z] 增量
        
        # 将action[-1,1]缩放到±0.01m的增量范围
        left_increment = left_increment_action * self.INCREMENT_SCALE
        right_increment = right_increment_action * self.INCREMENT_SCALE
        
        # 更新当前位置（基于增量）
        self.current_left_pos += left_increment
        self.current_right_pos += right_increment
        
        # 保持固定的姿态
        left_quat = self.FIXED_LEFT_QUAT.copy()
        right_quat = self.FIXED_RIGHT_QUAT.copy()
        left_elbow_pos = DEMO_LEFT_ELBOW_POS
        right_elbow_pos = DEMO_RIGHT_ELBOW_POS

        msg = twoArmHandPoseCmd()
        msg.hand_poses.left_pose.pos_xyz = self.current_left_pos.tolist()
        msg.hand_poses.left_pose.quat_xyzw = left_quat.tolist()
        msg.hand_poses.left_pose.elbow_pos_xyz = left_elbow_pos.tolist()

        msg.hand_poses.right_pose.pos_xyz = self.current_right_pos.tolist()
        msg.hand_poses.right_pose.quat_xyzw = right_quat.tolist()
        msg.hand_poses.right_pose.elbow_pos_xyz = right_elbow_pos.tolist()
        
        # Set default IK params (can be customized as needed)
        msg.use_custom_ik_param = False
        msg.joint_angles_as_q0 = False
        # if not IF_USE_ARM_MPC_CONTROL:
        #     msg.joint_angles_as_q0 = True
        # else:
        #     msg.joint_angles_as_q0 = False
        
        msg.ik_param = ikSolveParam()
        msg.frame = 3  # keep current frame3 | 3 为vr系
        self.ee_pose_pub.publish(msg)
        
        if self.debug:
            print(f"[INCREMENT CONTROL] Left increment: {left_increment}, New pos: {self.current_left_pos}")
            print(f"[INCREMENT CONTROL] Right increment: {right_increment}, New pos: {self.current_right_pos}")

    def _publish_joint_control_arm_poses(self, joint_action: np.ndarray):
        """
        Publish arm poses using joint control mode.
        
        Args:
            joint_action: Normalized joint action array [-1, 1] with 14 values 
                         (7 for left arm, 7 for right arm)
        """
        # Extract left and right arm actions (normalized [-1, 1])
        left_arm_action = joint_action[0:7]  # First 7 joints for left arm
        right_arm_action = joint_action[7:14]  # Next 7 joints for right arm
        
        # FIXME:打印
        print(f"publish two joint control: {left_arm_action} {right_arm_action}")

        # # Convert normalized actions to actual joint angles using centers and scales
        # left_arm_centers = self.arm_joint_centers[0:7]
        # left_arm_scales = self.arm_joint_scales[0:7]
        # right_arm_centers = self.arm_joint_centers[7:14]
        # right_arm_scales = self.arm_joint_scales[7:14]
        
        # Calculate target joint angles: center + (action * scale)
        left_joint_angles_rad = left_arm_action
        right_joint_angles_rad = right_arm_action
        #left_joint_angles_rad = left_arm_centers + (left_arm_action * left_arm_scales)
        #right_joint_angles_rad = right_arm_centers + (right_arm_action * right_arm_scales)

        # Combine all joint angles in the correct order
        all_joint_angles_rad = np.concatenate([left_joint_angles_rad, right_joint_angles_rad])
        
        # Publish joint angles using existing method
        success = self.pub_control_robot_arm_traj(all_joint_angles_rad.tolist())
        
    def _reset_simulation(self):
        """
        Implement this method to call the reset service for the Kuavo simulation.
        """
        try:
            # call 服务
            rospy.wait_for_service('/isaac_lab_reset_scene', timeout=5.0)
            resp = self.reset_client(0) # 0 for random seed in sim | 在这里等待服务端处理完成并且返回结果
            if not resp.success:
                raise RuntimeError(f"Failed to reset simulation: {resp.message}")
            if self.debug:
                rospy.loginfo("Simulation reset successfully via ROS service.")

            # 等待3秒 让手臂自然归位
            time.sleep(3)

            # 使用eef pose控制时每次回到固定位置
            if not TEST_DEMO_USE_ACTION_16_DIM:
                self._publish_fixed_arm_poses()

            time.sleep(1)
            
        except (rospy.ServiceException, rospy.ROSException) as e:
            raise RuntimeError(f"Service call to reset simulation failed: {str(e)}")

    def _compute_reward_and_done(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> Tuple[float, bool, Dict[str, Any]]:
        """
        分阶段奖励函数（修复累积奖励问题版）：
        - 阶段1 (dist_torso_to_box > 0.5): 靠近箱子阶段
        - 阶段2 (dist_torso_to_box <= 0.5): 抓取箱子阶段
        - 修复了奖励累积和终端条件问题
        """
        # ========== DEMO MODE: CHOOSE CONTROL MODE ==========
        info = {}
        
        # Extract data from observation
        agent_state = obs['agent_pos']
        env_state = obs['environment_state']
        
        global TEST_DEMO_USE_ACTION_16_DIM
        global LEARN_TARGET_EEF_POSE_TARGET
        
        global DEMO_TARGET_LEFT_POS_WORLD
        global DEMO_TARGET_RIGHT_POS_WORLD
        global DEMO_TARGET_BOX_POS_WORLD

        if not LEARN_TARGET_EEF_POSE_TARGET:
            """
                学习控制eef pose or joint | 使用关节角度作为学习目标
            """
            # ========== JOINT CONTROL MODE ==========
            if self.episode_step_count >= 200:
                # Timeout - terminate but no success
                terminated = True
                info["success"] = False
            else:
                terminated = False
                info["success"] = False
            
            # FIXME:=========== 重新设计的reward - 基于目标关节角度 ====================
            reward = 0.0
            
            current_joint_angles = agent_state[6:20]  # arm_data位置
            
            # 目标关节角度 (左手7个 + 右手7个 = 14个)
            target_joint_angles = DEMO_TARGET_ALL_JOINT_ANGLES
            
            # 计算关节角度差异的MSE
            joint_angle_diff = current_joint_angles - target_joint_angles
            mse_joint_angles = np.mean(joint_angle_diff ** 2) # reward = -np.mean((action - target_action) ** 2)
            
            # 选择奖励函数类型 (可以切换不同的实现)
            reward_type = "MSE_joint_angles"  # 新的基于关节角度的MSE奖励
            
            if reward_type == "MSE_joint_angles":
                # 基于关节角度MSE的奖励函数
                # 使用负的MSE，让agent最小化与目标的差异
                reward = -mse_joint_angles
                
                # 可选：添加一个scale factor让奖励范围更合理
                reward_scale = 1.0  # 可以调整这个值
                reward *= reward_scale
                
                if self.debug and self.episode_step_count % 10 == 0:  # 每10步打印一次
                    print(f"[REWARD DEBUG] MSE joint angles: {mse_joint_angles:.6f}, Reward: {reward:.6f}")
                    print(f"[REWARD DEBUG] Max joint diff: {np.max(np.abs(joint_angle_diff)):.4f} rad ({np.rad2deg(np.max(np.abs(joint_angle_diff))):.2f} deg)")
            elif reward_type == "original":
                # 原始的指数衰减函数 (保留作为备选)
                target_action = np.ones_like(self.last_action) * 0.0
                action_distance = np.linalg.norm(action - target_action)
                reward = np.exp(-action_distance)
            elif reward_type == "shaped":
                # 更平缓的奖励函数，提供更好的梯度信号
                target_action = np.ones_like(self.last_action) * 0.0
                action_distance = np.linalg.norm(action - target_action)
                reward = 1.0 / (1.0 + action_distance)
        else:
            """
                学习控制eef pose or joint | 使用目标eef pose作为学习目标
            """
            print(" === use MSE reward function === ")
            # ========== EEF POSITION CONTROL MODE ==========
            if self.episode_step_count >= 200:
                # Timeout - terminate but no success
                terminated = True
                info["success"] = False
            else:
                terminated = False
                info["success"] = False
            
            # FIXME:=========== 重新设计的reward - 基于目标末端执行器位置 ====================
            """
                L_rewad = w1 * Mse_box_left + w2 * Mse_box_right + alpha * box_down_fail
            """
            mse_eef = 0.0
            mse_box = 0.0
            box_down_fail = 0.0
                        
            # box lift target MSE - 根据WBC模式选择正确的数据源
            if self.wbc_observation_enabled:
                # WBC模式：box位置在agent_state[29:32]
                current_box_pos_world = agent_state[29:32]
            else:
                # 非WBC模式：box位置在environment_state[0:3]
                current_box_pos_world = env_state[0:3]
            
            current_box_pos_z = current_box_pos_world[2]
            target_box_pos_world = DEMO_TARGET_BOX_POS_WORLD
            mse_box = np.mean((current_box_pos_world - target_box_pos_world) ** 2)
            mse_box = -mse_box
            
            # 计算左右手位置差异的MSE
            current_left_eef_pos_world = agent_state[0:3]   # world坐标系左手位置
            current_right_eef_pos_world = agent_state[3:6]  # world坐标系右手位置
            target_left_eef_pos_world = DEMO_TARGET_LEFT_POS_WORLD
            target_right_eef_pos_world = DEMO_TARGET_RIGHT_POS_WORLD
            mse_left_eef = np.mean((current_left_eef_pos_world - target_left_eef_pos_world) ** 2)
            mse_right_eef = np.mean((current_right_eef_pos_world - target_right_eef_pos_world) ** 2)
            mse_eef = -(mse_left_eef + mse_right_eef)
            
            # 使用负的MSE，让agent最小化与目标的差异
            reward = mse_eef + mse_box + box_down_fail
            
            # success condition - 成功判断
            if current_box_pos_z > target_box_pos_world[2]:
                terminated = True
                info["success"] = True

            # box fail - 箱子掉了
            if current_box_pos_z < 0.20:
                terminated = True
                info["success"] = False
                reward = -30.0

            # 可选：添加一个scale factor让奖励范围更合理
            reward_scale = 10.0  # 位置误差通常比较小，需要放大
            reward *= reward_scale

        return reward, terminated, info


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Implement the step logic for the Kuavo robot.
        
        Args:
            action: The action to execute
        """
        global IF_USE_ZERO_OBS_FLAG

        # Increment step counter for efficiency reward calculation
        self.episode_step_count += 1
        
        self._send_action(action, self.episode_step_count)
        obs = self._get_observation()

        # Zero out all observations for testing
        if IF_USE_ZERO_OBS_FLAG:
            obs['agent_pos'] = np.zeros_like(obs['agent_pos'])
            obs['environment_state'] = np.zeros_like(obs['environment_state'])
            obs['pixels']['front'] = np.zeros_like(obs['pixels']['front'])
        
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
        
        reward, done, info = self._compute_reward_and_done(obs, action)
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

        # Reset incremental control state - 重置到初始位置
        self.current_left_pos = np.array([0.3178026345146559, 0.4004180715613648, -0.019417275957965042], dtype=np.float32)
        self.current_right_pos = np.array([0.3178026345146559, -0.4004180715613648, -0.019417275957965042], dtype=np.float32)

        # Reset reward tracking variables for improved reward function
        self.last_mean_distance = None
        self.last_ee_action = None

        # Reset reward tracking variables for improved reward function
        if hasattr(self, 'last_mse_total_eef'):
            self.last_mse_total_eef = None
        if hasattr(self, 'last_ee_action'):
            self.last_ee_action = None
        
        # Reset demo mode progress tracking
        global TEST_DEMO_USE_ACTION_16_DIM
        if TEST_DEMO_USE_ACTION_16_DIM:
            # Reset joint control tracking for DEMO mode
            if hasattr(self, 'best_joint_deviation'):
                self.best_joint_deviation = float('inf')
            if hasattr(self, 'left_arm_achieved'):
                self.left_arm_achieved = False
            if hasattr(self, 'right_arm_achieved'):
                self.right_arm_achieved = False
        else:
            # Reset end-effector position control tracking for DEMO mode
            if hasattr(self, 'best_mean_distance'):
                self.best_mean_distance = float('inf')
            if hasattr(self, 'left_hand_achieved'):
                self.left_hand_achieved = False
            if hasattr(self, 'right_hand_achieved'):
                self.right_hand_achieved = False

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
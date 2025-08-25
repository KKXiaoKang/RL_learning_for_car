#!/usr/bin/env python3
"""
Robotic Bézier Action Record Tool

This tool interpolates between Cartesian key-points using Bézier curves for robotic end-effector trajectories.
It ensures continuity between consecutive segments and provides visualization capabilities.
Also publishes trajectories as ROS messages for real-time execution.
"""
"""
    # 仅仅发布actions
    python3 robotic_bezier_action_record_tool.py --mode actions

    # 播放轨迹 + 发布actions
    python3 robotic_bezier_action_record_tool.py --mode play_actions --rate 10.0 --debug

    # 循环播放轨迹 + Actions
    python3 robotic_bezier_action_record_tool.py --mode play_actions --rate 10.0 --loop --debug
    
    # 使用当前机器人位置作为初始点来生成轨迹
    python3 robotic_bezier_action_record_tool.py --mode actions --use-current-pose --debug
    
    # 播放轨迹时使用当前机器人位置作为起点
    python3 robotic_bezier_action_record_tool.py --mode play_actions --use-current-pose --rate 10.0 --debug
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Dict, Any
import os
import time

# ROS imports
try:
    import rospy
    from kuavo_msgs.msg import twoArmHandPoseCmd, twoArmHandPose, ikSolveParam
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point, PoseStamped
    from std_msgs.msg import ColorRGBA, Float64MultiArray, Bool
    import threading
    ROS_AVAILABLE = True
except ImportError:
    print("Warning: ROS packages not available. ROS functionality will be disabled.")
    ROS_AVAILABLE = False


class BezierTrajectoryGenerator:
    """
    Generates smooth Bézier trajectories for robotic end-effector motion.
    """
    
    def __init__(self, key_points_file: str = "key_point.json", enable_ros: bool = True, debug: bool = False, 
                 use_current_robot_pose: bool = False):
        """
        Initialize the Bézier trajectory generator.
        
        Args:
            key_points_file: Path to the JSON file containing key-points
            enable_ros: Whether to enable ROS functionality
            debug: Whether to enable debug output
            use_current_robot_pose: Whether to use current robot pose as initial key-point
        """
        self.key_points_file = key_points_file
        self.enable_ros = enable_ros and ROS_AVAILABLE
        self.debug = debug
        self.use_current_robot_pose = use_current_robot_pose
        
        # Current robot pose storage
        self.latest_left_eef_pose = None
        self.latest_right_eef_pose = None
        self.left_eef_lock = threading.Lock()
        self.right_eef_lock = threading.Lock()
        self.poses_received = False
        
        # Trajectory control
        self.trajectory_paused = False
        self.pause_sub = None
        
        # ROS setup
        if self.enable_ros:
            self._setup_ros()
        
        # Load key-points (after ROS setup if using current robot pose)
        self.key_points = self._load_key_points()
    
    def _setup_ros(self):
        """Setup ROS node and publishers/subscribers."""
        try:
            # Initialize ROS node
            rospy.init_node("bezier_trajectory_publisher", anonymous=True)
            
            # Create publisher for IK commands
            self.ik_pub = rospy.Publisher('/ik/two_arm_hand_pose_cmd', twoArmHandPoseCmd, queue_size=10)
            
            # Create publishers for trajectory visualization
            self.left_trajectory_pub = rospy.Publisher(
                '/kuavo_strategy/arm_key_point/left', 
                MarkerArray, 
                queue_size=10
            )
            self.right_trajectory_pub = rospy.Publisher(
                '/kuavo_strategy/arm_key_point/right', 
                MarkerArray, 
                queue_size=10
            )
            
            # Create publishers for scaled action differences
            self.left_action_pub = rospy.Publisher(
                '/sac/kuavo_eef_action_scale_left',
                Float64MultiArray,
                queue_size=10
            )
            self.right_action_pub = rospy.Publisher(
                '/sac/kuavo_eef_action_scale_right', 
                Float64MultiArray,
                queue_size=10
            )
            
            # Create subscriber for IK results (optional callback)
            self.ik_result_sub = rospy.Subscriber("/ik/result", twoArmHandPose, self._ik_result_callback)
            
            # Create subscriber for trajectory pause/resume requests
            self.pause_sub = rospy.Subscriber("/sac/trajectory_pause_request", Bool, self._pause_callback)
            
            # Create subscribers for current robot end-effector poses
            if self.use_current_robot_pose:
                self.left_eef_sub = rospy.Subscriber(
                    "/fk/base_link_eef_left", PoseStamped, self._left_eef_pose_callback
                )
                self.right_eef_sub = rospy.Subscriber(
                    "/fk/base_link_eef_right", PoseStamped, self._right_eef_pose_callback
                )
                print("Subscribed to current robot end-effector poses:")
                print("  - /fk/base_link_eef_left")
                print("  - /fk/base_link_eef_right")
            
            # Wait for connections
            time.sleep(1.0)
            
            print("ROS node initialized successfully")
            print("Listening for trajectory pause requests on /sac/trajectory_pause_request")
            
        except Exception as e:
            print(f"Failed to setup ROS: {e}")
            self.enable_ros = False
    
    def _ik_result_callback(self, msg):
        """Callback for IK result messages."""
        # Optional: Handle IK result feedback
        pass
    
    def _pause_callback(self, msg):
        """Callback for trajectory pause/resume requests."""
        if msg.data:
            self.trajectory_paused = True
            print("[BEZIER DEBUG] Trajectory paused!")
        else:
            self.trajectory_paused = False
            print("[BEZIER DEBUG] Trajectory resumed!")
    
    def _left_eef_pose_callback(self, msg):
        """Callback for left end-effector pose messages."""
        with self.left_eef_lock:
            self.latest_left_eef_pose = msg
            if not self.poses_received and self.latest_right_eef_pose is not None:
                self.poses_received = True
                if self.debug:
                    print("[BEZIER DEBUG] Received initial left end-effector pose")
    
    def _right_eef_pose_callback(self, msg):
        """Callback for right end-effector pose messages."""
        with self.right_eef_lock:
            self.latest_right_eef_pose = msg
            if not self.poses_received and self.latest_left_eef_pose is not None:
                self.poses_received = True
                if self.debug:
                    print("[BEZIER DEBUG] Received initial right end-effector pose")
    
    def get_current_robot_poses(self, timeout: float = 5.0) -> Tuple[bool, Dict[str, Any]]:
        """
        Get current robot end-effector poses from ROS topics.
        
        Args:
            timeout: Maximum time to wait for poses (seconds)
            
        Returns:
            Tuple of (success, poses_dict) where poses_dict contains current poses
        """
        if not self.enable_ros or not self.use_current_robot_pose:
            return False, {}
        
        print(f"Waiting for current robot end-effector poses (timeout: {timeout}s)...")
        start_time = time.time()
        
        # Wait for poses to be received
        while not self.poses_received and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            if rospy.is_shutdown():
                return False, {}
        
        if not self.poses_received:
            print(f"Warning: Failed to receive robot poses within {timeout}s timeout")
            return False, {}
        
        # Extract poses with thread safety
        with self.left_eef_lock:
            left_pose = self.latest_left_eef_pose
        with self.right_eef_lock:
            right_pose = self.latest_right_eef_pose
            
        if left_pose is None or right_pose is None:
            print("Warning: One or both end-effector poses are None")
            return False, {}
        
        # Convert PoseStamped to position and quaternion arrays
        current_poses = {
            'left_hand': {
                'position': [
                    left_pose.pose.position.x,
                    left_pose.pose.position.y,
                    left_pose.pose.position.z
                ],
                'quaternion': [
                    left_pose.pose.orientation.x,
                    left_pose.pose.orientation.y,
                    left_pose.pose.orientation.z,
                    left_pose.pose.orientation.w
                ]
            },
            'right_hand': {
                'position': [
                    right_pose.pose.position.x,
                    right_pose.pose.position.y,
                    right_pose.pose.position.z
                ],
                'quaternion': [
                    right_pose.pose.orientation.x,
                    right_pose.pose.orientation.y,
                    right_pose.pose.orientation.z,
                    right_pose.pose.orientation.w
                ]
            }
        }
        
        if self.debug:
            print("[BEZIER DEBUG] Current robot poses:")
            print(f"  Left hand: pos={current_poses['left_hand']['position']}, quat={current_poses['left_hand']['quaternion']}")
            print(f"  Right hand: pos={current_poses['right_hand']['position']}, quat={current_poses['right_hand']['quaternion']}")
        
        return True, current_poses
        
    def _load_key_points(self) -> Dict[str, Any]:
        """Load key-points from JSON file, optionally updating initial pose from robot."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, self.key_points_file)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        key_points = data['key_points']
        
        # Update initial keyframe (frame_id: 0) with current robot pose if requested
        if self.use_current_robot_pose and self.enable_ros:
            success, current_poses = self.get_current_robot_poses(timeout=10.0)
            
            if success:
                # Find frame_id 0 and update it
                for i, keyframe in enumerate(key_points['keyframes']):
                    if keyframe.get('frame_id', -1) == 0:
                        print(f"[BEZIER INFO] Updating initial keyframe (frame_id: 0) with current robot pose")
                        
                        # Store original pose for reference
                        if self.debug:
                            print(f"  Original left hand: {keyframe['left_hand']}")
                            print(f"  Original right hand: {keyframe['right_hand']}")
                        
                        # Update with current robot pose
                        keyframe['left_hand']['position'] = current_poses['left_hand']['position']
                        keyframe['left_hand']['quaternion'] = current_poses['left_hand']['quaternion']
                        keyframe['right_hand']['position'] = current_poses['right_hand']['position']
                        keyframe['right_hand']['quaternion'] = current_poses['right_hand']['quaternion']
                        
                        if self.debug:
                            print(f"  Updated left hand: {keyframe['left_hand']}")
                            print(f"  Updated right hand: {keyframe['right_hand']}")
                        
                        print(f"[BEZIER INFO] Successfully updated initial keyframe with current robot pose")
                        break
                else:
                    print(f"[BEZIER WARNING] No keyframe with frame_id: 0 found, cannot update initial pose")
            else:
                print(f"[BEZIER WARNING] Failed to get current robot poses, using original keyframe data")
        
        return key_points
    
    def cubic_bezier(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Calculate cubic Bézier curve points.
        
        Cubic Bézier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
        
        Args:
            p0, p1, p2, p3: Control points for the Bézier curve
            t: Parameter array from 0 to 1
            
        Returns:
            Array of interpolated points
        """
        t = t.reshape(-1, 1)  # Make t column vector for broadcasting
        
        return ((1 - t) ** 3 * p0 + 
                3 * (1 - t) ** 2 * t * p1 + 
                3 * (1 - t) * t ** 2 * p2 + 
                t ** 3 * p3)
    
    def generate_control_points(self, positions: List[np.ndarray], smoothness_factor: float = 0.3) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate control points for cubic Bézier curves ensuring C1 continuity.
        
        Args:
            positions: List of key-point positions
            smoothness_factor: Factor controlling the smoothness (0-1)
            
        Returns:
            List of tuples containing (p0, p1, p2, p3) for each segment
        """
        if len(positions) < 2:
            raise ValueError("At least 2 positions are required")
        
        segments = []
        
        for i in range(len(positions) - 1):
            p0 = positions[i]
            p3 = positions[i + 1]
            
            # Calculate direction vectors for smooth transitions
            if i == 0:
                # First segment: use forward difference
                if len(positions) > 2:
                    direction_out = (positions[i + 2] - positions[i]) * smoothness_factor
                else:
                    direction_out = (p3 - p0) * smoothness_factor
                p1 = p0 + direction_out / 3
            else:
                # Use previous segment's direction for continuity
                prev_direction = (p0 - positions[i - 1])
                p1 = p0 + prev_direction * smoothness_factor / 3
            
            if i == len(positions) - 2:
                # Last segment: use backward difference
                if i > 0:
                    direction_in = (positions[i + 1] - positions[i - 1]) * smoothness_factor
                else:
                    direction_in = (p3 - p0) * smoothness_factor
                p2 = p3 - direction_in / 3
            else:
                # Use next segment's direction for continuity
                next_direction = (positions[i + 2] - p0)
                p2 = p3 - next_direction * smoothness_factor / 3
            
            segments.append((p0, p1, p2, p3))
        
        return segments
    
    def _manual_slerp(self, q1: R, q2: R, t: float) -> R:
        """
        Manual spherical linear interpolation for older scipy versions.
        
        Args:
            q1: Starting rotation
            q2: Ending rotation  
            t: Interpolation parameter (0-1)
            
        Returns:
            Interpolated rotation
        """
        # Convert to quaternions
        quat1 = q1.as_quat()
        quat2 = q2.as_quat()
        
        # Ensure shortest path (dot product should be positive)
        dot = np.dot(quat1, quat2)
        if dot < 0.0:
            quat2 = -quat2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = quat1 + t * (quat2 - quat1)
            result = result / np.linalg.norm(result)
            return R.from_quat(result)
        
        # Calculate angle between quaternions
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        # Calculate interpolated quaternion
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        result = s0 * quat1 + s1 * quat2
        
        return R.from_quat(result)
    
    def create_trajectory_markers(self, positions: np.ndarray, hand: str) -> MarkerArray:
        """
        Create MarkerArray for trajectory visualization in RViz.
        
        Args:
            positions: Array of 3D positions
            hand: 'left' or 'right' for color coding
            
        Returns:
            MarkerArray containing trajectory visualization markers
        """
        if not self.enable_ros:
            return None
            
        marker_array = MarkerArray()
        
        # Create line strip marker for trajectory
        trajectory_marker = Marker()
        trajectory_marker.header.frame_id = "base_link"
        trajectory_marker.header.stamp = rospy.Time.now()
        trajectory_marker.ns = f"{hand}_trajectory"
        trajectory_marker.id = 0
        trajectory_marker.type = Marker.LINE_STRIP
        trajectory_marker.action = Marker.ADD
        
        # Set marker scale
        trajectory_marker.scale.x = 0.01  # Line width
        
        # Set color based on hand
        if hand == 'left':
            trajectory_marker.color = ColorRGBA(1.0, 0.0, 1.0, 0.8)  # Blue
        else:
            trajectory_marker.color = ColorRGBA(1.0, 0.0, 1.0, 0.8)  # Red
        
        # Add all trajectory points
        for pos in positions:
            point = Point()
            point.x = pos[0]
            point.y = pos[1]
            point.z = pos[2]
            trajectory_marker.points.append(point)
        
        marker_array.markers.append(trajectory_marker)
        
        # Create sphere markers for key points (every 10th point for clarity)
        for i in range(0, len(positions), 10):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "base_link"
            sphere_marker.header.stamp = rospy.Time.now()
            sphere_marker.ns = f"{hand}_points"
            sphere_marker.id = i // 10
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            
            # Set position
            sphere_marker.pose.position.x = positions[i][0]
            sphere_marker.pose.position.y = positions[i][1]
            sphere_marker.pose.position.z = positions[i][2]
            sphere_marker.pose.orientation.w = 1.0
            
            # Set scale
            sphere_marker.scale.x = 0.02
            sphere_marker.scale.y = 0.02
            sphere_marker.scale.z = 0.02
            
            # Set color (slightly different from trajectory)
            if hand == 'left':
                sphere_marker.color = ColorRGBA(0.0, 1.0, 1.0, 1.0)  # Lighter blue
            else:
                sphere_marker.color = ColorRGBA(0.0, 1.0, 1.0, 1.0)  # Lighter red
            
            marker_array.markers.append(sphere_marker)
        
        return marker_array
    
    def publish_trajectory_visualization(self):
        """
        Publish trajectory visualization markers to RViz.
        """
        if not self.enable_ros:
            print("Warning: ROS is not enabled. Cannot publish visualization.")
            return
        
        # Generate trajectories
        left_positions, _ = self.interpolate_trajectory('left_hand')
        right_positions, _ = self.interpolate_trajectory('right_hand')
        
        # Create and publish markers
        left_markers = self.create_trajectory_markers(left_positions, 'left')
        right_markers = self.create_trajectory_markers(right_positions, 'right')
        
        if left_markers and right_markers:
            self.left_trajectory_pub.publish(left_markers)
            self.right_trajectory_pub.publish(right_markers)
            print("Published trajectory visualization markers")
    
    def calculate_trajectory_actions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算贝塞尔轨迹点之间的增量，并缩放回action空间[-1,1]。
        
        Returns:
            left_actions, right_actions: 每个轨迹点对应的action数组
        """
        # 生成轨迹
        left_positions, _ = self.interpolate_trajectory('left_hand')
        right_positions, _ = self.interpolate_trajectory('right_hand')
        
        # 计算位置增量（相邻点之间的差值）
        left_increments = np.diff(left_positions, axis=0)  # shape: (n-1, 3)
        right_increments = np.diff(right_positions, axis=0)  # shape: (n-1, 3)
        
        # 参考rl_kuavo_gym_env.py中的缩放方法
        # INCREMENT_SCALE = 0.01，即 action[-1,1] * 0.01 = ±0.01m的增量范围
        INCREMENT_SCALE = 0.01
        
        # 将增量从米制缩放回action空间 [-1,1]
        # increment = action * INCREMENT_SCALE -> action = increment / INCREMENT_SCALE
        left_actions = left_increments / INCREMENT_SCALE  # shape: (n-1, 3)
        right_actions = right_increments / INCREMENT_SCALE  # shape: (n-1, 3)
        
        # 限制action范围在[-1, 1]内
        left_actions = np.clip(left_actions, -1.0, 1.0)
        right_actions = np.clip(right_actions, -1.0, 1.0)
        
        return left_actions, right_actions
    
    def publish_trajectory_actions(self):
        """
        发布轨迹点之间的action到ROS话题。
        """
        if not self.enable_ros:
            print("Warning: ROS is not enabled. Cannot publish trajectory actions.")
            return
        
        # 计算action序列
        left_actions, right_actions = self.calculate_trajectory_actions()
        
        # 发布左手action序列
        left_msg = Float64MultiArray()
        left_msg.data = left_actions.flatten().tolist()  # 展平为一维数组
        self.left_action_pub.publish(left_msg)
        
        # 发布右手action序列
        right_msg = Float64MultiArray()
        right_msg.data = right_actions.flatten().tolist()  # 展平为一维数组
        self.right_action_pub.publish(right_msg)
        
        print(f"Published trajectory actions:")
        print(f"  Left actions shape: {left_actions.shape}, range: [{np.min(left_actions):.3f}, {np.max(left_actions):.3f}]")
        print(f"  Right actions shape: {right_actions.shape}, range: [{np.min(right_actions):.3f}, {np.max(right_actions):.3f}]")
        print(f"  Published to topics:")
        print(f"    - /sac/kuavo_eef_action_scale_left")
        print(f"    - /sac/kuavo_eef_action_scale_right")
        
        # 打印action统计信息
        self._print_action_statistics(left_actions, right_actions)
    
    def _print_action_statistics(self, left_actions: np.ndarray, right_actions: np.ndarray):
        """
        打印action序列的详细统计信息。
        
        Args:
            left_actions: 左手action序列 (n, 3)
            right_actions: 右手action序列 (n, 3)
        """
        print("\n" + "="*50)
        print("ACTION STATISTICS")
        print("="*50)
        
        # 左手统计
        print(f"Left Hand Actions:")
        print(f"  Shape: {left_actions.shape}")
        print(f"  X-axis: mean={np.mean(left_actions[:, 0]):.4f}, std={np.std(left_actions[:, 0]):.4f}, range=[{np.min(left_actions[:, 0]):.4f}, {np.max(left_actions[:, 0]):.4f}]")
        print(f"  Y-axis: mean={np.mean(left_actions[:, 1]):.4f}, std={np.std(left_actions[:, 1]):.4f}, range=[{np.min(left_actions[:, 1]):.4f}, {np.max(left_actions[:, 1]):.4f}]")
        print(f"  Z-axis: mean={np.mean(left_actions[:, 2]):.4f}, std={np.std(left_actions[:, 2]):.4f}, range=[{np.min(left_actions[:, 2]):.4f}, {np.max(left_actions[:, 2]):.4f}]")
        
        # 右手统计
        print(f"Right Hand Actions:")
        print(f"  Shape: {right_actions.shape}")
        print(f"  X-axis: mean={np.mean(right_actions[:, 0]):.4f}, std={np.std(right_actions[:, 0]):.4f}, range=[{np.min(right_actions[:, 0]):.4f}, {np.max(right_actions[:, 0]):.4f}]")
        print(f"  Y-axis: mean={np.mean(right_actions[:, 1]):.4f}, std={np.std(right_actions[:, 1]):.4f}, range=[{np.min(right_actions[:, 1]):.4f}, {np.max(right_actions[:, 1]):.4f}]")
        print(f"  Z-axis: mean={np.mean(right_actions[:, 2]):.4f}, std={np.std(right_actions[:, 2]):.4f}, range=[{np.min(right_actions[:, 2]):.4f}, {np.max(right_actions[:, 2]):.4f}]")
        
        # 总体统计
        all_actions = np.concatenate([left_actions, right_actions], axis=0)
        print(f"Overall Actions:")
        print(f"  Total steps: {len(all_actions)}")
        print(f"  Mean magnitude: {np.mean(np.linalg.norm(all_actions, axis=1)):.4f}")
        print(f"  Max magnitude: {np.max(np.linalg.norm(all_actions, axis=1)):.4f}")
        print(f"  Actions in range [-1,1]: {np.all(np.abs(all_actions) <= 1.0)}")
        
        # 超出范围的action统计
        out_of_range = np.abs(all_actions) > 1.0
        if np.any(out_of_range):
            print(f"  WARNING: {np.sum(out_of_range)} action values exceed [-1,1] range!")
            print(f"  Max absolute value: {np.max(np.abs(all_actions)):.4f}")
        
        print("="*50)
    
    def play_trajectory_with_actions(self, playback_rate: float = 10.0, loop: bool = False):
        """
        播放轨迹的同时发布对应的action序列。
        
        Args:
            playback_rate: 播放速率 (Hz)
            loop: 是否循环播放
        """
        if not self.enable_ros:
            print("Warning: ROS is not enabled. Cannot play trajectory with actions.")
            return
        
        # 生成轨迹和actions
        trajectory_data = self.get_trajectory_data()
        left_positions = trajectory_data['left_hand']['positions']
        left_orientations = trajectory_data['left_hand']['orientations']
        right_positions = trajectory_data['right_hand']['positions']
        right_orientations = trajectory_data['right_hand']['orientations']
        
        # 计算action序列
        left_actions, right_actions = self.calculate_trajectory_actions()
        
        # 发布可视化
        self.publish_trajectory_visualization()
        
        # 显示action统计信息
        print(f"Calculated trajectory actions:")
        print(f"  Left actions shape: {left_actions.shape}, range: [{np.min(left_actions):.3f}, {np.max(left_actions):.3f}]")
        print(f"  Right actions shape: {right_actions.shape}, range: [{np.min(right_actions):.3f}, {np.max(right_actions):.3f}]")
        
        # 如果启用debug，显示详细统计
        if self.debug:
            self._print_action_statistics(left_actions, right_actions)
        
        rate = rospy.Rate(playback_rate)
        
        print(f"Starting trajectory playback with actions at {playback_rate} Hz...")
        print(f"Total trajectory points: {len(left_positions)}")
        print(f"Total action steps: {len(left_actions)}")
        print("Press Ctrl+C to stop")
        
        try:
            while not rospy.is_shutdown():
                # 发布轨迹点和对应的actions
                for i in range(len(left_positions)):
                    if rospy.is_shutdown():
                        break
                    
                    # 检查是否暂停
                    while self.trajectory_paused and not rospy.is_shutdown():
                        if self.debug:
                            print("[BEZIER DEBUG] Trajectory paused, waiting for resume...")
                        rate.sleep()
                    
                    # 如果在暂停等待期间收到关闭信号，退出
                    if rospy.is_shutdown():
                        break
                    
                    # 发布当前pose
                    self.publish_single_pose(
                        left_positions[i], left_orientations[i],
                        right_positions[i], right_orientations[i]
                    )
                    
                    # 发布对应的action（如果有的话）
                    if i < len(left_actions):
                        left_action_msg = Float64MultiArray()
                        left_action_msg.data = left_actions[i].tolist()
                        self.left_action_pub.publish(left_action_msg)
                        
                        right_action_msg = Float64MultiArray()
                        right_action_msg.data = right_actions[i].tolist()
                        self.right_action_pub.publish(right_action_msg)
                        
                        if self.debug and i % 10 == 0:  # 每10步打印一次
                            print(f"Step {i}: Left action: {left_actions[i]}, Right action: {right_actions[i]}")
                    
                    rate.sleep()
                
                if not loop:
                    break
                else:
                    print("Looping trajectory with actions...")
        
        except rospy.ROSInterruptException:
            print("Trajectory playback with actions interrupted")
        except KeyboardInterrupt:
            print("Trajectory playback with actions stopped by user")
    
    def interpolate_trajectory(self, hand: str, num_points_per_segment: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate trajectory for specified hand using Bézier curves.
        
        Args:
            hand: 'left_hand' or 'right_hand'
            num_points_per_segment: Number of interpolated points per segment
            
        Returns:
            Tuple of (positions, orientations) arrays
        """
        keyframes = self.key_points['keyframes']
        
        # Extract positions and quaternions
        positions = []
        quaternions = []
        
        for frame in keyframes:
            positions.append(np.array(frame[hand]['position']))
            quaternions.append(np.array(frame[hand]['quaternion']))
        
        # Generate control points for position interpolation
        position_segments = self.generate_control_points(positions)
        
        # Interpolate positions using Bézier curves
        interpolated_positions = []
        t_values = np.linspace(0, 1, num_points_per_segment)
        
        for p0, p1, p2, p3 in position_segments:
            segment_positions = self.cubic_bezier(p0, p1, p2, p3, t_values)
            interpolated_positions.append(segment_positions)
        
        # Concatenate all segments (remove duplicate points at junctions)
        all_positions = []
        for i, segment in enumerate(interpolated_positions):
            if i == 0:
                all_positions.append(segment)
            else:
                # Skip the first point to avoid duplication
                all_positions.append(segment[1:])
        
        final_positions = np.vstack(all_positions)
        
        # For orientations, use spherical linear interpolation (SLERP)
        interpolated_orientations = []
        
        for i in range(len(quaternions) - 1):
            q1 = R.from_quat(quaternions[i])
            q2 = R.from_quat(quaternions[i + 1])
            
            # Create rotation interpolation
            segment_rotations = []
            for t in t_values:
                # SLERP between quaternions - compatible with older scipy versions
                try:
                    # For scipy >= 1.2.0
                    interpolated_rot = q1.slerp(q2, t)
                except AttributeError:
                    # For older scipy versions, use manual SLERP
                    interpolated_rot = self._manual_slerp(q1, q2, t)
                segment_rotations.append(interpolated_rot.as_quat())
            
            interpolated_orientations.append(np.array(segment_rotations))
        
        # Concatenate orientation segments
        all_orientations = []
        for i, segment in enumerate(interpolated_orientations):
            if i == 0:
                all_orientations.append(segment)
            else:
                all_orientations.append(segment[1:])
        
        final_orientations = np.vstack(all_orientations)
        
        return final_positions, final_orientations
    
    def visualize_trajectories(self, save_plot: bool = True, show_plot: bool = True):
        """
        Visualize the interpolated trajectories for both hands.
        
        Args:
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
        """
        # Generate trajectories for both hands
        left_positions, left_orientations = self.interpolate_trajectory('left_hand')
        right_positions, right_orientations = self.interpolate_trajectory('right_hand')
        
        # Extract key-point positions for visualization
        left_keypoints = []
        right_keypoints = []
        
        for frame in self.key_points['keyframes']:
            left_keypoints.append(frame['left_hand']['position'])
            right_keypoints.append(frame['right_hand']['position'])
        
        left_keypoints = np.array(left_keypoints)
        right_keypoints = np.array(right_keypoints)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot interpolated trajectories
        ax.plot(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2], 
                'b-', linewidth=2, label='Left Hand Trajectory', alpha=0.8)
        ax.plot(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2], 
                'r-', linewidth=2, label='Right Hand Trajectory', alpha=0.8)
        
        # Plot key-points
        ax.scatter(left_keypoints[:, 0], left_keypoints[:, 1], left_keypoints[:, 2], 
                  c='blue', s=100, marker='o', label='Left Hand Key-points', alpha=1.0)
        ax.scatter(right_keypoints[:, 0], right_keypoints[:, 1], right_keypoints[:, 2], 
                  c='red', s=100, marker='s', label='Right Hand Key-points', alpha=1.0)
        
        # Add key-point labels
        for i, (left_kp, right_kp) in enumerate(zip(left_keypoints, right_keypoints)):
            ax.text(left_kp[0], left_kp[1], left_kp[2], f'L{i}', fontsize=10, color='blue')
            ax.text(right_kp[0], right_kp[1], right_kp[2], f'R{i}', fontsize=10, color='red')
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robotic End-Effector Bézier Trajectories (base_link frame)')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([left_positions.max() - left_positions.min(),
                             right_positions.max() - right_positions.min()]).max() / 2.0
        
        mid_x = (left_positions[:, 0].max() + left_positions[:, 0].min()) * 0.5
        mid_y = (left_positions[:, 1].max() + left_positions[:, 1].min()) * 0.5
        mid_z = (left_positions[:, 2].max() + left_positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig('bezier_trajectories.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'bezier_trajectories.png'")
        
        if show_plot:
            plt.show()
    
    def get_trajectory_data(self) -> Dict[str, np.ndarray]:
        """
        Get the complete trajectory data for both hands.
        
        Returns:
            Dictionary containing trajectory data
        """
        left_positions, left_orientations = self.interpolate_trajectory('left_hand')
        right_positions, right_orientations = self.interpolate_trajectory('right_hand')
        
        return {
            'left_hand': {
                'positions': left_positions,
                'orientations': left_orientations
            },
            'right_hand': {
                'positions': right_positions,
                'orientations': right_orientations
            }
        }
    
    def export_trajectory(self, filename: str = 'trajectory_data.json'):
        """
        Export trajectory data to JSON file.
        
        Args:
            filename: Output filename
        """
        trajectory_data = self.get_trajectory_data()
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {
            'left_hand': {
                'positions': trajectory_data['left_hand']['positions'].tolist(),
                'orientations': trajectory_data['left_hand']['orientations'].tolist()
            },
            'right_hand': {
                'positions': trajectory_data['right_hand']['positions'].tolist(),
                'orientations': trajectory_data['right_hand']['orientations'].tolist()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Trajectory data exported to '{filename}'")
    
    def create_pose_command_msg(self, left_pos: np.ndarray, left_quat: np.ndarray, 
                               right_pos: np.ndarray, right_quat: np.ndarray) -> 'twoArmHandPoseCmd':
        """
        Create a twoArmHandPoseCmd message for the given poses.
        
        Args:
            left_pos: Left hand position [x, y, z]
            left_quat: Left hand quaternion [x, y, z, w]
            right_pos: Right hand position [x, y, z]
            right_quat: Right hand quaternion [x, y, z, w]
            
        Returns:
            twoArmHandPoseCmd message
        """
        if not self.enable_ros:
            raise RuntimeError("ROS is not enabled or available")
        
        eef_pose_msg = twoArmHandPoseCmd()
        
        # Set IK parameters from configuration
        ik_params = self.key_points.get('ik_parameters', {})
        eef_pose_msg.use_custom_ik_param = ik_params.get('use_custom_ik_param', False)
        eef_pose_msg.joint_angles_as_q0 = ik_params.get('joint_angles_as_q0', False)
        
        # Create ikSolveParam object
        eef_pose_msg.ik_param = ikSolveParam()
        
        # Set frame (0: current frame, 1: world frame, 2: local frame, 3: manipulation world frame)
        eef_pose_msg.frame = 3
        
        # Set joint angles (not used when joint_angles_as_q0 is False)
        eef_pose_msg.hand_poses.left_pose.joint_angles = np.zeros(7)
        eef_pose_msg.hand_poses.right_pose.joint_angles = np.zeros(7)
        
        # Set left hand pose
        eef_pose_msg.hand_poses.left_pose.pos_xyz = left_pos.tolist()
        eef_pose_msg.hand_poses.left_pose.quat_xyzw = left_quat.tolist()
        
        # Set right hand pose
        eef_pose_msg.hand_poses.right_pose.pos_xyz = right_pos.tolist()
        eef_pose_msg.hand_poses.right_pose.quat_xyzw = right_quat.tolist()
        
        # Set elbow positions
        elbow_positions = self.key_points.get('elbow_positions', {})
        left_elbow = np.array(elbow_positions.get('left_elbow', [0.0, 0.0, 0.0]))
        right_elbow = np.array(elbow_positions.get('right_elbow', [0.0, 0.0, 0.0]))
        
        eef_pose_msg.hand_poses.left_pose.elbow_pos_xyz = left_elbow.tolist()
        eef_pose_msg.hand_poses.right_pose.elbow_pos_xyz = right_elbow.tolist()
        
        return eef_pose_msg
    
    def publish_single_pose(self, left_pos: np.ndarray, left_quat: np.ndarray, 
                           right_pos: np.ndarray, right_quat: np.ndarray):
        """
        Publish a single pose command.
        
        Args:
            left_pos: Left hand position [x, y, z]
            left_quat: Left hand quaternion [x, y, z, w]
            right_pos: Right hand position [x, y, z]
            right_quat: Right hand quaternion [x, y, z, w]
        """
        if not self.enable_ros:
            print("Warning: ROS is not enabled. Cannot publish pose command.")
            return
        
        try:
            eef_pose_msg = self.create_pose_command_msg(left_pos, left_quat, right_pos, right_quat)
            self.ik_pub.publish(eef_pose_msg)
            print(f"Published pose command - Left: {left_pos}, Right: {right_pos}")
        except Exception as e:
            print(f"Error publishing pose command: {e}")
    
    def play_trajectory(self, playback_rate: float = 10.0, loop: bool = False):
        """
        Play the interpolated trajectory by publishing poses at specified rate.
        
        Args:
            playback_rate: Publishing rate in Hz
            loop: Whether to loop the trajectory continuously
        """
        if not self.enable_ros:
            print("Warning: ROS is not enabled. Cannot play trajectory.")
            return
        
        # Generate trajectories
        trajectory_data = self.get_trajectory_data()
        left_positions = trajectory_data['left_hand']['positions']
        left_orientations = trajectory_data['left_hand']['orientations']
        right_positions = trajectory_data['right_hand']['positions']
        right_orientations = trajectory_data['right_hand']['orientations']
        
        rate = rospy.Rate(playback_rate)
        
        # Publish trajectory visualization
        self.publish_trajectory_visualization()
        
        print(f"Starting trajectory playback at {playback_rate} Hz...")
        print(f"Total trajectory points: {len(left_positions)}")
        print("Press Ctrl+C to stop")
        
        try:
            while not rospy.is_shutdown():
                for i in range(len(left_positions)):
                    if rospy.is_shutdown():
                        break
                    
                    # 检查是否暂停
                    while self.trajectory_paused and not rospy.is_shutdown():
                        if self.debug:
                            print("[BEZIER DEBUG] Trajectory paused, waiting for resume...")
                        rate.sleep()
                    
                    # 如果在暂停等待期间收到关闭信号，退出
                    if rospy.is_shutdown():
                        break
                    
                    # Publish current pose
                    self.publish_single_pose(
                        left_positions[i], left_orientations[i],
                        right_positions[i], right_orientations[i]
                    )
                    
                    rate.sleep()
                
                if not loop:
                    break
                else:
                    print("Looping trajectory...")
        
        except rospy.ROSInterruptException:
            print("Trajectory playback interrupted")
        except KeyboardInterrupt:
            print("Trajectory playback stopped by user")
    
    def publish_keyframe(self, frame_id: int):
        """
        Publish a specific keyframe pose.
        
        Args:
            frame_id: ID of the keyframe to publish
        """
        if not self.enable_ros:
            print("Warning: ROS is not enabled. Cannot publish keyframe.")
            return
        
        keyframes = self.key_points['keyframes']
        
        if frame_id >= len(keyframes):
            print(f"Error: Frame ID {frame_id} is out of range (0-{len(keyframes)-1})")
            return
        
        frame = keyframes[frame_id]
        left_pos = np.array(frame['left_hand']['position'])
        left_quat = np.array(frame['left_hand']['quaternion'])
        right_pos = np.array(frame['right_hand']['position'])
        right_quat = np.array(frame['right_hand']['quaternion'])
        
        self.publish_single_pose(left_pos, left_quat, right_pos, right_quat)
        print(f"Published keyframe {frame_id}")


def main():
    """Main function to demonstrate the Bézier trajectory generation."""
    print("Robotic Bézier Action Record Tool")
    print("=" * 40)
    
    import argparse
    parser = argparse.ArgumentParser(description="Bézier trajectory generator for robotic end-effectors")
    parser.add_argument('--mode', choices=['visualize', 'play', 'keyframe', 'export', 'rviz', 'actions', 'play_actions'], 
                       default='visualize', help='Operation mode')
    parser.add_argument('--rate', type=float, default=10.0, 
                       help='Playback rate in Hz (for play mode)')
    parser.add_argument('--loop', action='store_true', 
                       help='Loop trajectory continuously (for play mode)')
    parser.add_argument('--frame-id', type=int, default=0, 
                       help='Keyframe ID to publish (for keyframe mode)')
    parser.add_argument('--no-ros', action='store_true', 
                       help='Disable ROS functionality')
    parser.add_argument('--no-plot', action='store_true', 
                       help='Disable plot display (for visualization mode)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    parser.add_argument('--use-current-pose', action='store_true',
                       help='Use current robot pose as initial keyframe (frame_id: 0)')
    parser.add_argument('--pose-timeout', type=float, default=5.0,
                       help='Timeout for waiting current robot pose (seconds)')
    
    args = parser.parse_args()
    
    try:
        # Initialize the trajectory generator
        generator = BezierTrajectoryGenerator(
            enable_ros=not args.no_ros, 
            debug=args.debug,
            use_current_robot_pose=args.use_current_pose
        )
        
        if args.mode == 'visualize':
            # Generate and visualize trajectories
            print("Generating Bézier trajectories for robotic end-effectors...")
            generator.visualize_trajectories(show_plot=not args.no_plot)
            
            # Print trajectory statistics
            trajectory_data = generator.get_trajectory_data()
            left_positions = trajectory_data['left_hand']['positions']
            right_positions = trajectory_data['right_hand']['positions']
            
            print(f"\nTrajectory Statistics:")
            print(f"Left hand trajectory points: {len(left_positions)}")
            print(f"Right hand trajectory points: {len(right_positions)}")
            print(f"Total trajectory length (left): {np.sum(np.linalg.norm(np.diff(left_positions, axis=0), axis=1)):.3f} m")
            print(f"Total trajectory length (right): {np.sum(np.linalg.norm(np.diff(right_positions, axis=0), axis=1)):.3f} m")
        
        elif args.mode == 'play':
            # Play trajectory via ROS
            if not generator.enable_ros:
                print("Error: ROS is required for trajectory playback")
                return 1
            generator.play_trajectory(playback_rate=args.rate, loop=args.loop)
        
        elif args.mode == 'keyframe':
            # Publish single keyframe
            if not generator.enable_ros:
                print("Error: ROS is required for keyframe publishing")
                return 1
            generator.publish_keyframe(args.frame_id)
        
        elif args.mode == 'export':
            # Export trajectory data
            generator.export_trajectory()
            print("Trajectory data exported successfully")
        
        elif args.mode == 'rviz':
            # Publish trajectory visualization to RViz
            if not generator.enable_ros:
                print("Error: ROS is required for RViz visualization")
                return 1
            generator.publish_trajectory_visualization()
            print("Trajectory visualization published to RViz")
            print("Topics:")
            print("  - /kuavo_strategy/arm_key_point/left")
            print("  - /kuavo_strategy/arm_key_point/right")
            print("Add MarkerArray displays in RViz with these topics")
        
        elif args.mode == 'actions':
            # Calculate and publish trajectory actions
            if not generator.enable_ros:
                print("Error: ROS is required for action publishing")
                return 1
            generator.publish_trajectory_actions()
            print("Trajectory actions calculated and published")
        
        elif args.mode == 'play_actions':
            # Play trajectory with synchronized action publishing
            if not generator.enable_ros:
                print("Error: ROS is required for trajectory playback with actions")
                return 1
            generator.play_trajectory_with_actions(playback_rate=args.rate, loop=args.loop)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


def demo_usage():
    """Demonstrate different usage modes of the tool."""
    print("\nDemo Usage Examples:")
    print("=" * 50)
    
    try:
        # Initialize generator
        generator = BezierTrajectoryGenerator(enable_ros=False, debug=False)  # Disable ROS for demo
        
        print("1. Visualizing trajectories...")
        generator.visualize_trajectories(show_plot=False, save_plot=True)
        
        print("2. Getting trajectory data...")
        trajectory_data = generator.get_trajectory_data()
        print(f"   Generated {len(trajectory_data['left_hand']['positions'])} trajectory points")
        
        print("3. Exporting trajectory data...")
        generator.export_trajectory('demo_trajectory.json')
        
        print("\nDemo completed successfully!")
        
        if ROS_AVAILABLE:
            print("\nROS Usage Examples:")
            print("   python robotic_bezier_action_record_tool.py --mode play --rate 5.0")
            print("   python robotic_bezier_action_record_tool.py --mode keyframe --frame-id 1")
            print("   python robotic_bezier_action_record_tool.py --mode play --loop")
            print("   python robotic_bezier_action_record_tool.py --mode rviz")
            print("   python robotic_bezier_action_record_tool.py --mode actions")
            print("   python robotic_bezier_action_record_tool.py --mode play_actions --rate 10.0")
            print("\nNew Feature - Use Current Robot Pose:")
            print("   python robotic_bezier_action_record_tool.py --mode actions --use-current-pose --debug")
            print("   python robotic_bezier_action_record_tool.py --mode play_actions --use-current-pose --rate 10.0")
            print("   python robotic_bezier_action_record_tool.py --mode visualize --use-current-pose --debug")
        else:
            print("\nNote: ROS packages not available. Install them to enable ROS functionality.")
            
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    import sys
    
    # Check if demo mode
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_usage()
    else:
        sys.exit(main())
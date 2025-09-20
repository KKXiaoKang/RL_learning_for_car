#!/usr/bin/env python3
"""
可视化节点：直接在base_link坐标系中可视化Sequence ACT Actor的预测轨迹
监听话题：
- /policy/action/eef_pose_marker_all (Sequence ACT Actor发布的action buffer数据)

发布话题：
- /policy/base_link/eef_pose_marker_left
- /policy/base_link/eef_pose_marker_right
- /policy/base_link/eef_marker_array_all (合并的左右手可视化markers)
"""

import sys, os
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray
from collections import deque
import threading
import os
import numpy as np
from kuavo_msgs.srv import fkSrv

FLOAT_BASE_COM_HEIGTH = True
STL_INTERVAL = 50

class PoseTransformNode:
    def __init__(self):
        rospy.init_node('pose_transform_node', anonymous=True)

        # 获取机器人质心高度参数
        try:
            self.com_height = rospy.get_param('/com_height', 0.8135007665261358)
            self.com_height = 0.8267460465431213
            rospy.loginfo(f"Using robot COM height: {self.com_height}")
        except Exception as e:
            self.com_height = 0.8267460465431213  # 默认值
            rospy.logwarn(f"Could not get COM height parameter, using default: {self.com_height}")

        # 初始化TF2监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # # 等待TF变换可用
        # try:
        #     rospy.loginfo("Waiting for TF transform between 'odom' and 'base_link'...")
        #     # 使用can_transform检查变换是否可用
        #     if self.tf_buffer.can_transform("odom", "base_link", rospy.Time(), rospy.Duration(10.0)):
        #         rospy.loginfo("TF transform is available")
        #     else:
        #         rospy.logwarn("TF transform not available within timeout. Will try on demand.")
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        #     rospy.logwarn(f"Could not check transform: {str(e)}. Will try on demand.")
        
        # 发布器 - 发布base_link坐标系下的位姿和MarkerArray可视化
        self.left_pose_pub = rospy.Publisher('/policy/base_link/eef_pose_marker_left', PoseStamped, queue_size=10)
        self.right_pose_pub = rospy.Publisher('/policy/base_link/eef_pose_marker_right', PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('/policy/base_link/eef_marker_array_all', MarkerArray, queue_size=1)
        
        # 发布器 - 发布ground truth可视化
        self.gt_marker_pub = rospy.Publisher('/policy/base_link/gt_marker_array_all', MarkerArray, queue_size=1)
        
        # 订阅器 - 监听合并后的关节位置数据
        self.joint_sub = rospy.Subscriber('/policy/action/eef_pose_marker_all', Float64MultiArray, 
                                         self.joint_callback, queue_size=10)
        
        # 订阅器 - 监听ground truth关节位置数据
        self.gt_joint_sub = rospy.Subscriber('/policy/GT/eef_pose_marker_all', Float64MultiArray, 
                                           self.gt_joint_callback, queue_size=10)
        
        # 保留原有的pose订阅器（如果需要的话）
        # self.left_sub = rospy.Subscriber('/policy/base/eef_pose_marker_left', PoseStamped, 
        #                                 self.left_pose_callback, queue_size=10)
        # self.right_sub = rospy.Subscriber('/policy/base/eef_pose_marker_right', PoseStamped, 
        #                                  self.right_pose_callback, queue_size=10)
        
        # 用于缓存poses，实现批量发布
        self.left_poses_buffer = deque(maxlen=100)
        self.right_poses_buffer = deque(maxlen=100)
        self.buffer_lock = threading.Lock()
        
        # 用于缓存完整的动作块，确保一次性发布完整的50个markers
        self.current_left_chunk = []
        self.current_right_chunk = []
        self.chunk_lock = threading.Lock()
        
        # 用于缓存ground truth数据
        self.current_gt_left_chunk = []
        self.current_gt_right_chunk = []
        self.gt_chunk_lock = threading.Lock()
        
        # 缓存当前预测步骤的所有markers，用于下次清空
        self.current_markers = []
        self.current_gt_markers = []
        self.marker_id_counter = 0
        self.gt_marker_id_counter = 0
        
        # STL文件路径
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.left_hand_mesh_path = f"file://{os.path.join(self.script_dir, 'meshes', 'eef_grasp_tool_offset_scale_02_left_scale.STL')}"
        self.right_hand_mesh_path = f"file://{os.path.join(self.script_dir, 'meshes', 'eef_grasp_tool_offset_scale_02_right_scale.STL')}"
        
        # 定时器，定期发布缓存的poses和markers - 提高频率减少延迟
        self.publish_timer = rospy.Timer(rospy.Duration(0.005), self.publish_buffered_data)  # 200Hz
        
        # 初始化FK服务客户端
        rospy.loginfo("Waiting for FK service...")
        rospy.wait_for_service('/ik/fk_srv')
        self.fk_srv = rospy.ServiceProxy('/ik/fk_srv', fkSrv)
        rospy.loginfo("FK service is ready")
        
        rospy.loginfo("Pose transform node initialized")
        rospy.loginfo("Listening to /policy/action/eef_pose_marker_all (combined left and right arm data)")
        rospy.loginfo("Listening to /policy/GT/eef_pose_marker_all (ground truth data)")
        rospy.loginfo("Publishing poses to /policy/base_link/eef_pose_marker_left and /policy/base_link/eef_pose_marker_right")
        rospy.loginfo("Publishing combined markers to /policy/base_link/eef_marker_array_all")
        rospy.loginfo("Publishing ground truth markers to /policy/base_link/gt_marker_array_all")
        rospy.loginfo(f"Using mesh files: {self.left_hand_mesh_path}, {self.right_hand_mesh_path}")

    def transform_pose_with_tf2(self, pose_stamped, com_offset_z=0.0):
        """使用TF2将pose从base_link转换到odom坐标系，并应用质心偏移量"""
        global FLOAT_BASE_COM_HEIGTH

        if pose_stamped.header.frame_id != "base_link":
            rospy.logwarn(f"Expected frame_id 'base_link', got '{pose_stamped.header.frame_id}'")
            return None
            
        try:
            # 使用最新可用的变换时间
            pose_stamped.header.stamp = rospy.Time(0)  # 使用最新可用的变换
            
            # 获取TF2变换信息用于调试
            transform = self.tf_buffer.lookup_transform("odom", "base_link", rospy.Time(0))
            
            # 质心偏移添加到这里
            if FLOAT_BASE_COM_HEIGTH:
                """ float base """
                transform.transform.translation.z = self.com_height + com_offset_z
            else:
                """ fixed base """
                transform.transform.translation.z = self.com_height
            # 使用TF2进行坐标变换
            pose_odom = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
            
            # 查看末端eef pose
            odom_z_after_tf2 = pose_odom.pose.position.z
                            
            # 调试信息：显示变换前后的高度和偏移量
            print(f"eef Base_link z: {pose_stamped.pose.position.z:.4f}, "
                  f"com height after offset TF2 transform z: {transform.transform.translation.z:.4f}, "
                  f"COM offset: {com_offset_z:.4f}, "
                  f"eef Odom z (after TF2): {odom_z_after_tf2:.4f}, "
                  f"Method: {'TF-based' if FLOAT_BASE_COM_HEIGTH else 'Direct offset'}")
            pose_odom.header.stamp = rospy.Time.now()
            return pose_odom
            
        except tf2_ros.LookupException as e:
            rospy.logwarn_throttle(5.0, f"TF2 LookupException: {str(e)}")
            return None
        except tf2_ros.ConnectivityException as e:
            rospy.logwarn_throttle(5.0, f"TF2 ConnectivityException: {str(e)}")
            return None
        except tf2_ros.ExtrapolationException as e:
            rospy.logwarn_throttle(5.0, f"TF2 ExtrapolationException: {str(e)}")
            return None
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Unexpected TF2 error: {str(e)}")
            return None

    def compute_forward_kinematics(self, joint_positions):
        """
        使用ROS服务计算正向运动学，得到base_link坐标系下的末端位置
        
        Args:
            joint_positions: shape为(action_chunk_size, 14)的关节角度数组，前7个是左手，后7个是右手
            
        Returns:
            Tuple of (left_poses, right_poses) - List of PoseStamped messages in base_link frame
        """
        try:
            left_poses = []
            right_poses = []
            failed_steps = []
            
            for i, joint_angles in enumerate(joint_positions):
                try:
                    # 调用FK服务
                    fk_result = self.fk_srv(joint_angles)
                    
                    if fk_result.success:
                        # 创建左手pose
                        left_pose = PoseStamped()
                        left_pose.header.frame_id = "base_link"
                        left_pose.header.stamp = rospy.Time(0)
                        
                        # 设置位置
                        left_pose.pose.position.x = fk_result.hand_poses.left_pose.pos_xyz[0]
                        left_pose.pose.position.y = fk_result.hand_poses.left_pose.pos_xyz[1]
                        left_pose.pose.position.z = fk_result.hand_poses.left_pose.pos_xyz[2]
                        
                        # 设置姿态（四元数）
                        left_pose.pose.orientation.x = fk_result.hand_poses.left_pose.quat_xyzw[0]
                        left_pose.pose.orientation.y = fk_result.hand_poses.left_pose.quat_xyzw[1]
                        left_pose.pose.orientation.z = fk_result.hand_poses.left_pose.quat_xyzw[2]
                        left_pose.pose.orientation.w = fk_result.hand_poses.left_pose.quat_xyzw[3]
                        
                        left_poses.append(left_pose)
                        
                        # 创建右手pose
                        right_pose = PoseStamped()
                        right_pose.header.frame_id = "base_link"
                        right_pose.header.stamp = rospy.Time(0)
                        
                        # 设置位置
                        right_pose.pose.position.x = fk_result.hand_poses.right_pose.pos_xyz[0]
                        right_pose.pose.position.y = fk_result.hand_poses.right_pose.pos_xyz[1]
                        right_pose.pose.position.z = fk_result.hand_poses.right_pose.pos_xyz[2]
                        
                        # 设置姿态（四元数）
                        right_pose.pose.orientation.x = fk_result.hand_poses.right_pose.quat_xyzw[0]
                        right_pose.pose.orientation.y = fk_result.hand_poses.right_pose.quat_xyzw[1]
                        right_pose.pose.orientation.z = fk_result.hand_poses.right_pose.quat_xyzw[2]
                        right_pose.pose.orientation.w = fk_result.hand_poses.right_pose.quat_xyzw[3]
                        
                        right_poses.append(right_pose)
                    else:
                        rospy.logwarn(f"FK service failed for step {i}")
                        failed_steps.append(i)
                    
                except rospy.ServiceException as e:
                    rospy.logwarn(f"FK service call failed for step {i}: {str(e)}")
                    failed_steps.append(i)
                    continue
                except Exception as e:
                    rospy.logwarn(f"FK computation failed for step {i}: {str(e)}")
                    failed_steps.append(i)
                    continue
            
            # 报告失败的时间步
            if failed_steps:
                rospy.logwarn(f"FK failed for {len(failed_steps)} steps: {failed_steps[:10]}{'...' if len(failed_steps) > 10 else ''}")
                    
            rospy.logdebug(f"Computed {len(left_poses)} left poses and {len(right_poses)} right poses from {len(joint_positions)} joint configurations")
            return left_poses, right_poses
            
        except Exception as e:
            rospy.logerr(f"Error in forward kinematics computation: {str(e)}")
            return None, None
    
    def transform_pose(self, pose_stamped, com_offset_z=0.0):
        """将pose从base_link转换到odom坐标系 (使用TF2)"""
        return self.transform_pose_with_tf2(pose_stamped, com_offset_z)

    def left_pose_callback(self, msg):
        """左手pose回调函数（保留，但现在不使用）"""
        transformed_pose = self.transform_pose(msg)
        if transformed_pose is not None:
            with self.buffer_lock:
                self.left_poses_buffer.append((transformed_pose, 'left'))

    def joint_callback(self, msg):
        """处理Sequence ACT Actor发布的action buffer数据"""
        try:
            # 解析数据：数据格式为flatten的数组，需要重新reshape
            action_data = np.array(msg.data)
            
            # 新的数据格式：每7个数据为一个时间步
            # [left_x, left_y, left_z, right_x, right_y, right_z, cmd_vel_linear_z] * chunk_size
            num_data_per_step = 7
            action_chunk_size = len(action_data) // num_data_per_step
            
            if len(action_data) % num_data_per_step != 0:
                rospy.logwarn(f"Action data size {len(action_data)} is not divisible by {num_data_per_step}")
                return
                
            # 重新reshape为(action_chunk_size, 7)
            action_positions = action_data.reshape(action_chunk_size, num_data_per_step)
            
            # 分离左右手位置和质心偏移量
            left_eef_positions = action_positions[:, :3]   # 前3个是左手位置 [x, y, z]
            right_eef_positions = action_positions[:, 3:6] # 后3个是右手位置 [x, y, z]
            cmd_vel_linear_z = action_positions[:, 6]      # 第7个是cmd_vel_linear_z质心偏移量
            
            # 添加调试信息
            rospy.logdebug(f"Received {len(action_data)} action values, reshaped to {action_positions.shape}")
            rospy.logdebug(f"Left eef positions shape: {left_eef_positions.shape}, Right eef positions shape: {right_eef_positions.shape}")
            rospy.loginfo(f"COM offset shape: {cmd_vel_linear_z.shape}, range: [{cmd_vel_linear_z.min():.4f}, {cmd_vel_linear_z.max():.4f}], "
                         f"mean: {cmd_vel_linear_z.mean():.4f}, std: {cmd_vel_linear_z.std():.4f}")
            
            # 初始化base_link坐标系下的poses列表
            left_base_link_poses = []
            right_base_link_poses = []
            
            # 处理左手数据 - 直接在base_link坐标系中创建PoseStamped
            for i, pos in enumerate(left_eef_positions):
                # 创建base_link坐标系下的pose
                left_pose = PoseStamped()
                left_pose.header.frame_id = "base_link"
                left_pose.header.stamp = rospy.Time.now()
                
                # 设置位置
                left_pose.pose.position.x = pos[0]
                left_pose.pose.position.y = pos[1]
                left_pose.pose.position.z = pos[2]
                
                # 设置默认姿态（可以根据需要调整）
                left_pose.pose.orientation.x = 0.0
                left_pose.pose.orientation.y = -0.70711
                left_pose.pose.orientation.z = 0.0
                left_pose.pose.orientation.w = 0.70711
                
                # 直接添加到base_link坐标系列表，不进行坐标变换
                left_base_link_poses.append(left_pose)
            
            # 处理右手数据 - 直接在base_link坐标系中创建PoseStamped
            for i, pos in enumerate(right_eef_positions):
                # 创建base_link坐标系下的pose
                right_pose = PoseStamped()
                right_pose.header.frame_id = "base_link"
                right_pose.header.stamp = rospy.Time.now()
                
                # 设置位置
                right_pose.pose.position.x = pos[0]
                right_pose.pose.position.y = pos[1]
                right_pose.pose.position.z = pos[2]
                
                # 设置默认姿态（可以根据需要调整）
                right_pose.pose.orientation.x = 0.0
                right_pose.pose.orientation.y = -0.70711
                right_pose.pose.orientation.z = 0.0
                right_pose.pose.orientation.w = 0.70711
                
                # 直接添加到base_link坐标系列表，不进行坐标变换
                right_base_link_poses.append(right_pose)
            
            # 🔥 实时发布 - 减少延迟，立即发布可视化数据
            if left_base_link_poses or right_base_link_poses:
                # 立即发布单个poses（实时性更好）
                for pose in left_base_link_poses:
                    self.left_pose_pub.publish(pose)
                for pose in right_base_link_poses:
                    self.right_pose_pub.publish(pose)
                
                # 立即发布合并后的MarkerArray（减少批量处理延迟）
                self.publish_combined_marker_array(left_base_link_poses, right_base_link_poses)
                
                rospy.loginfo(f"Sequence ACT action buffer processed: Published {len(left_base_link_poses)} left + {len(right_base_link_poses)} right poses in real-time")
            else:
                rospy.logwarn("No valid poses computed for either arm")
                            
        except Exception as e:
            rospy.logerr(f"Error in joint callback: {str(e)}")
    
    def right_pose_callback(self, msg):
        """右手pose回调函数（保留，但现在不使用）"""
        transformed_pose = self.transform_pose(msg)
        if transformed_pose is not None:
            with self.buffer_lock:
                self.right_poses_buffer.append((transformed_pose, 'right'))

    def gt_joint_callback(self, msg):
        """Ground truth关节位置回调函数，处理数据集中的真值数据"""
        try:
            # 解析ground truth关节数据：数据格式为flatten的数组，需要重新reshape
            joint_data = np.array(msg.data)
            
            # 计算action_chunk_size（每15个数据为一个时间步：7个左手 + 7个右手 + 1个cmd_vel_linear_z）
            num_data_per_step = 15
            action_chunk_size = len(joint_data) // num_data_per_step
            
            if len(joint_data) % num_data_per_step != 0:
                rospy.logwarn(f"GT joint data size {len(joint_data)} is not divisible by {num_data_per_step}")
                return
                
            # 重新reshape为(action_chunk_size, 15)
            joint_positions = joint_data.reshape(action_chunk_size, num_data_per_step)
            
            # 分离左右手关节位置和质心偏移量
            left_joint_positions = joint_positions[:, :7]   # 前7个是左手关节
            right_joint_positions = joint_positions[:, 7:14] # 后7个是右手关节
            cmd_vel_linear_z = joint_positions[:, 14]       # 第15个是cmd_vel_linear_z质心偏移量
            
            # 添加调试信息
            rospy.logdebug(f"Received GT {len(joint_data)} joint values, reshaped to {joint_positions.shape}")
            rospy.logdebug(f"GT Left joints shape: {left_joint_positions.shape}, Right joints shape: {right_joint_positions.shape}")
            rospy.loginfo(f"GT COM offset shape: {cmd_vel_linear_z.shape}, range: [{cmd_vel_linear_z.min():.4f}, {cmd_vel_linear_z.max():.4f}], "
                         f"mean: {cmd_vel_linear_z.mean():.4f}, std: {cmd_vel_linear_z.std():.4f}")
            
            # 合并左右手关节数据用于FK服务调用
            combined_joint_positions = np.concatenate([left_joint_positions, right_joint_positions], axis=1)
            
            # 使用ROS服务计算正向运动学
            left_base_link_poses, right_base_link_poses = self.compute_forward_kinematics(combined_joint_positions)
            
            # 初始化转换后的poses列表
            left_transformed_poses = []
            right_transformed_poses = []
            
            # 处理左手数据
            if left_base_link_poses is not None:
                # 转换到odom坐标系，应用对应的质心偏移量
                for i, pose in enumerate(left_base_link_poses):
                    # 使用对应时间步的质心偏移量
                    com_offset = cmd_vel_linear_z[i] if i < len(cmd_vel_linear_z) else 0.0
                    transformed_pose = self.transform_pose(pose, com_offset)
                    if transformed_pose is not None:
                        left_transformed_poses.append(transformed_pose)
            else:
                rospy.logwarn("GT Left arm FK computation failed")
            
            # 处理右手数据
            if right_base_link_poses is not None:
                # 转换到odom坐标系，应用对应的质心偏移量
                for i, pose in enumerate(right_base_link_poses):
                    # 使用对应时间步的质心偏移量
                    com_offset = cmd_vel_linear_z[i] if i < len(cmd_vel_linear_z) else 0.0
                    transformed_pose = self.transform_pose(pose, com_offset)
                    if transformed_pose is not None:
                        right_transformed_poses.append(transformed_pose)
            else:
                rospy.logwarn("GT Right arm FK computation failed")
            
            # 只有当左右手都处理完成后，才一次性缓存数据
            if left_transformed_poses or right_transformed_poses:
                with self.gt_chunk_lock:
                    self.current_gt_left_chunk = left_transformed_poses.copy()
                    self.current_gt_right_chunk = right_transformed_poses.copy()
                    rospy.loginfo(f"GT Both arms processed: Cached {len(left_transformed_poses)} left + {len(right_transformed_poses)} right poses for batch publishing")
            else:
                rospy.logwarn("No valid GT poses computed for either arm")
                            
        except Exception as e:
            rospy.logerr(f"Error in GT joint callback: {str(e)}")

    def publish_buffered_data(self, event):
        """定期发布缓存的完整动作块"""
        # 处理预测数据
        with self.chunk_lock:
            # 检查是否有完整的动作块可以发布
            if self.current_left_chunk or self.current_right_chunk:
                left_poses = self.current_left_chunk.copy()
                right_poses = self.current_right_chunk.copy()
                
                # 清空缓存，准备接收下一批
                self.current_left_chunk.clear()
                self.current_right_chunk.clear()
                
                # 发布单个poses
                for pose in left_poses:
                    self.left_pose_pub.publish(pose)
                for pose in right_poses:
                    self.right_pose_pub.publish(pose)
                
                # 发布合并后的MarkerArray
                if left_poses or right_poses:
                    self.publish_combined_marker_array(left_poses, right_poses)
        
        # 处理ground truth数据
        with self.gt_chunk_lock:
            # 检查是否有ground truth数据可以发布
            if self.current_gt_left_chunk or self.current_gt_right_chunk:
                gt_left_poses = self.current_gt_left_chunk.copy()
                gt_right_poses = self.current_gt_right_chunk.copy()
                
                # 清空缓存，准备接收下一批
                self.current_gt_left_chunk.clear()
                self.current_gt_right_chunk.clear()
                
                # 发布ground truth MarkerArray
                if gt_left_poses or gt_right_poses:
                    self.publish_gt_marker_array(gt_left_poses, gt_right_poses)
    
    def publish_combined_marker_array(self, left_poses, right_poses):
        """
        创建并发布合并后的MarkerArray可视化
        每次预测都会清空上次的markers并创建新的

        对于50个时间步的情况
        STL模型时间步 0, 10, 20, 30, 40 共5个左手 + 5个右手 = 10个STL
        球体 时间步 1-9, 11-19, 21-29, 31-39, 41-49 共45个左手 + 45个右手 = 90个球体
        总计 100个markers 10个STL + 90个球体
        """
        global STL_INTERVAL
        current_time = rospy.Time.now()
        
        # 创建合并后的MarkerArray
        combined_marker_array = MarkerArray()
        
        # 首先清空之前的markers
        for old_marker in self.current_markers:
            delete_marker = Marker()
            delete_marker.header.frame_id = "base_link"
            delete_marker.header.stamp = current_time
            delete_marker.id = old_marker.id
            delete_marker.action = Marker.DELETE
            combined_marker_array.markers.append(delete_marker)
        
        # 创建新的markers
        self.current_markers = []
        
        # 设置STL模型显示间隔（每隔10个时间步显示一次）
        stl_interval = STL_INTERVAL
        
        # 添加左手markers
        for i, pose in enumerate(left_poses):
            # 每隔stl_interval个时间步显示STL模型，或者最后一个时间步必定显示STL模型
            if i % stl_interval == 0 or i == len(left_poses) - 1:
                marker = self.create_mesh_marker(pose, self.left_hand_mesh_path, self.marker_id_counter, current_time, 
                                               time_step=i, total_steps=len(left_poses), arm_type='left')
            else:
                # 其他时间步显示球体
                marker = self.create_sphere_marker(pose, self.marker_id_counter, current_time, 
                                                 time_step=i, total_steps=len(left_poses), arm_type='left')
            
            combined_marker_array.markers.append(marker)
            self.current_markers.append(marker)
            self.marker_id_counter += 1
        
        # 添加右手markers
        for i, pose in enumerate(right_poses):
            # 每隔stl_interval个时间步显示STL模型，或者最后一个时间步必定显示STL模型
            if i % stl_interval == 0 or i == len(right_poses) - 1:
                marker = self.create_mesh_marker(pose, self.right_hand_mesh_path, self.marker_id_counter, current_time,
                                               time_step=i, total_steps=len(right_poses), arm_type='right')
            else:
                # 其他时间步显示球体
                marker = self.create_sphere_marker(pose, self.marker_id_counter, current_time, 
                                                 time_step=i, total_steps=len(right_poses), arm_type='right')
            
            combined_marker_array.markers.append(marker)
            self.current_markers.append(marker)
            self.marker_id_counter += 1
        
        # 发布合并后的MarkerArray
        if combined_marker_array.markers:
            self.marker_pub.publish(combined_marker_array)
            new_marker_count = len([m for m in combined_marker_array.markers if m.action != Marker.DELETE])
            if new_marker_count > 0:
                # 计算STL和球体的数量（包括最后一个时间步的STL）
                stl_count = len([i for i in range(len(left_poses)) if i % stl_interval == 0 or i == len(left_poses) - 1]) + \
                           len([i for i in range(len(right_poses)) if i % stl_interval == 0 or i == len(right_poses) - 1])
                sphere_count = new_marker_count - stl_count
                # rospy.loginfo(f"Published {new_marker_count} combined markers ({len(left_poses)} left + {len(right_poses)} right): {stl_count} STL + {sphere_count} spheres")
    
    def create_mesh_marker(self, pose_stamped, mesh_path, marker_id, timestamp, time_step=0, total_steps=1, arm_type='left'):
        """
        为给定的pose创建一个mesh marker，支持基于时间步的颜色变化
        
        Args:
            pose_stamped: 位姿信息
            mesh_path: mesh文件路径
            marker_id: marker唯一ID
            timestamp: 时间戳
            time_step: 当前时间步 (0 到 total_steps-1)
            total_steps: 总时间步数
            arm_type: 'left' 或 'right'
        """
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = timestamp
        marker.id = marker_id
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        
        # 使用base_link坐标系下的pose
        marker.pose = pose_stamped.pose
        
        # 设置尺寸
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # 根据时间步和手臂类型设置颜色
        color = self.get_time_based_color(time_step, total_steps, arm_type)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.2 # color[3]
        
        # 设置网格资源
        marker.mesh_resource = mesh_path
        marker.mesh_use_embedded_materials = False
        
        return marker
    
    def create_sphere_marker(self, pose_stamped, marker_id, timestamp, time_step=0, total_steps=1, arm_type='left'):
        """
        为给定的pose创建一个球体marker，支持基于时间步的颜色变化
        
        Args:
            pose_stamped: 位姿信息
            marker_id: marker唯一ID
            timestamp: 时间戳
            time_step: 当前时间步 (0 到 total_steps-1)
            total_steps: 总时间步数
            arm_type: 'left' 或 'right'
        """
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = timestamp
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 使用base_link坐标系下的pose
        marker.pose = pose_stamped.pose
        
        # 设置球体尺寸
        marker.scale.x = 0.02  # 球体直径2cm
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        
        # 根据时间步和手臂类型设置颜色
        color = self.get_time_based_color(time_step, total_steps, arm_type)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        
        return marker
    
    def get_time_based_color(self, time_step, total_steps, arm_type):
        """
        根据时间步和手臂类型生成颜色
        
        Args:
            time_step: 当前时间步 (0 到 total_steps-1)
            total_steps: 总时间步数
            arm_type: 'left' 或 'right'
            
        Returns:
            (r, g, b, a) 颜色元组
        """
        # 计算时间进度 (0.0 到 1.0)
        if total_steps <= 1:
            progress = 0.0
        else:
            progress = time_step / (total_steps - 1)
        
        # 增强透明度对比：较早的时间步更透明，较晚的时间步更不透明
        base_alpha = 0.2 + 0.6 * progress  # 从0.2到0.8，增大对比度
        
        if arm_type == 'left':
            # 左手使用蓝色系渐变: 增强对比度
            # 时间步0: 深蓝 (0.0, 0.2, 1.0)
            # 时间步最后: 亮蓝青色 (0.0, 1.0, 1.0)
            r = 0.0
            g = 0.2 + 0.8 * progress
            b = 1.0
            
        else:  # right
            # 右手使用绿色系渐变: 增强对比度
            # 时间步0: 深绿 (0.0, 0.5, 0.0)
            # 时间步最后: 亮绿 (0.0, 1.0, 0.0)
            r = 0.0
            g = 0.5 + 0.5 * progress
            b = 0.0
        
        return (r, g, b, base_alpha)

    def publish_gt_marker_array(self, left_poses, right_poses):
        """
        创建并发布ground truth的MarkerArray可视化
        使用球体显示ground truth轨迹，与预测轨迹区分开
        不清空之前的markers，而是累积显示历史轨迹
        
        Args:
            left_poses: 左手ground truth poses列表
            right_poses: 右手ground truth poses列表
        """
        current_time = rospy.Time.now()
        
        # 创建ground truth MarkerArray
        gt_marker_array = MarkerArray()
        
        # 添加左手ground truth markers（使用球体）
        for i, pose in enumerate(left_poses):
            # 使用累积的marker数量作为时间步，这样颜色会随时间变化
            marker = self.create_gt_sphere_marker(pose, self.gt_marker_id_counter, current_time, 
                                                time_step=len(self.current_gt_markers) + i, total_steps=1000, arm_type='left')
            gt_marker_array.markers.append(marker)
            self.current_gt_markers.append(marker)
            self.gt_marker_id_counter += 1
        
        # 添加右手ground truth markers（使用球体）
        for i, pose in enumerate(right_poses):
            # 使用累积的marker数量作为时间步，这样颜色会随时间变化
            marker = self.create_gt_sphere_marker(pose, self.gt_marker_id_counter, current_time, 
                                                time_step=len(self.current_gt_markers) + i, total_steps=1000, arm_type='right')
            gt_marker_array.markers.append(marker)
            self.current_gt_markers.append(marker)
            self.gt_marker_id_counter += 1
        
        # 发布ground truth MarkerArray（只发布新添加的markers）
        if gt_marker_array.markers:
            self.gt_marker_pub.publish(gt_marker_array)
            new_marker_count = len(gt_marker_array.markers)
            if new_marker_count > 0:
                rospy.loginfo(f"Published {new_marker_count} new GT markers ({len(left_poses)} left + {len(right_poses)} right spheres), total GT markers: {len(self.current_gt_markers)}")

    def create_gt_sphere_marker(self, pose_stamped, marker_id, timestamp, time_step=0, total_steps=1, arm_type='left'):
        """
        为ground truth pose创建一个球体marker，使用不同的颜色和样式与预测轨迹区分
        
        Args:
            pose_stamped: 位姿信息
            marker_id: marker唯一ID
            timestamp: 时间戳
            time_step: 当前时间步 (0 到 total_steps-1)
            total_steps: 总时间步数
            arm_type: 'left' 或 'right'
        """
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = timestamp
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 使用base_link坐标系下的pose
        marker.pose = pose_stamped.pose
        
        # 设置球体尺寸（比预测轨迹稍大一些）
        marker.scale.x = 0.02  # 球体直径3cm
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        
        # 根据时间步和手臂类型设置ground truth专用颜色
        color = self.get_gt_time_based_color(time_step, total_steps, arm_type)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = color[3]
        
        return marker

    def get_gt_time_based_color(self, time_step, total_steps, arm_type):
        """
        根据时间步和手臂类型生成ground truth专用颜色
        使用更鲜明的颜色和更高的透明度来区分ground truth和预测轨迹
        
        Args:
            time_step: 当前时间步 (0 到 total_steps-1)
            total_steps: 总时间步数
            arm_type: 'left' 或 'right'
            
        Returns:
            (r, g, b, a) 颜色元组
        """
        # 计算时间进度 (0.0 到 1.0)
        if total_steps <= 1:
            progress = 0.0
        else:
            progress = time_step / (total_steps - 1)
        
        # Ground truth使用更高的透明度
        base_alpha = 0.6 + 0.3 * progress  # 从0.6到0.9，比预测轨迹更不透明
        
        if arm_type == 'left':
            # 左手ground truth使用黄色系渐变（与预测的蓝色区分）
            # 时间步0: 深黄 (0.8, 0.8, 0.0)
            # 时间步最后: 亮黄 (1.0, 1.0, 0.0)
            r = 0.8 + 0.2 * progress
            g = 0.8 + 0.2 * progress
            b = 0.0
            
        else:  # right
            # 右手ground truth使用紫色系渐变（与预测的绿色区分）
            # 时间步0: 深紫 (0.5, 0.0, 0.5)
            # 时间步最后: 亮紫 (1.0, 0.0, 1.0)
            r = 0.5 + 0.5 * progress
            g = 0.0
            b = 0.5 + 0.5 * progress
        
        return (r, g, b, base_alpha)

    def run(self):
        """运行节点"""
        rospy.loginfo("Pose transform node is running...")
        rospy.loginfo("Waiting for pose data to start visualization...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = PoseTransformNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Pose transform node shutting down")
    except Exception as e:
        rospy.logerr(f"Error in pose transform node: {str(e)}")
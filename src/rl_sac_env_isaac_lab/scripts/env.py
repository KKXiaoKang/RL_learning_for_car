# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to load and simulate a biped robot in IsaacLab.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/kuavo_robot_mpc/kuavo_locamotion.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher
import rospkg
import os
import time
import random
import numpy as np
import torch
import queue
import threading
from geometry_msgs.msg import PoseStamped

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo for loading biped robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ImuCfg # IMU设置
from isaaclab.sensors import CameraCfg, TiledCameraCfg, ContactSensorCfg, RayCasterCfg, patterns

# 导入你的机器人配置
from robot import KINOVA_ROBOTIQ

""" ROS ROBOT CONTROL """
import rospy
from kuavo_msgs.msg import jointCmd    # /joint_cmd
from kuavo_msgs.msg import sensorsData # /sensor_data_raw
from std_srvs.srv import SetBool, SetBoolResponse  
import random
import numpy as np

# Add these imports for camera publishing
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import cv2

# 在文件开头的导入部分添加新的服务类型
from kuavo_msgs.srv import resetIsaaclab, resetIsaaclabResponse
from kuavo_msgs.srv import SetTargetPoint, SetTargetPointResponse

DEBUG_FLAG = True
DECIMATION_RATE = 10
RESET_WORK_RUNNABLE = False

rospy.init_node('isaac_lab_kuavo_robot_mpc', anonymous=True) # 随机后缀

# 新增性能统计开关参数
ENABLE_PERF_LOG = rospy.get_param('enable_perf_log', True)  # 默认开启性能日志
rospy.loginfo(f"Performance logging enabled: {ENABLE_PERF_LOG}")

# 获取是否使用网格场景参数
USE_MESH_SCENE_GS_FLAG = rospy.get_param('use_mesh_scene_gs_flag', False)
rospy.loginfo(f"kuavo_locamotion.py -- USE_MESH_SCENE_GS_FLAG: {USE_MESH_SCENE_GS_FLAG}")

# 获取随机种子参数
random_seed = rospy.get_param('random_seed', int(time.time()))

# 获取网格id
MESH_ID = rospy.get_param('mesh_id', 1)
rospy.loginfo(f"kuavo_locamotion.py -- MESH_ID: {MESH_ID}")

# 获取场景名字id SLAM 3D-GS realCap 
MESH_SCENE_NAME = rospy.get_param('mesh_scene_name', 'SLAM')
rospy.loginfo(f"kuavo_locamotion.py -- MESH_SCENE_NAME: {MESH_SCENE_NAME}")

# 设置随机种子
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

rospy.loginfo(f"Using random seed: {random_seed}")

FIRST_TIME_FLAG = True
# 生成随机位置，确保两个箱子在对侧
ORIGIN_POS_X = 0.0
ORIGIN_POS_Y = 0.0
ORIGIN_POS_Z = 0.8

TARGET_POS_X = 0.0
TARGET_POS_Y = 0.0
TARGET_POS_Z = 0.8

# 在全局变量部分添加旋转四元数变量
global ORIGIN_ROT_W, ORIGIN_ROT_X, ORIGIN_ROT_Y, ORIGIN_ROT_Z

# 在全局变量部分添加新的旋转变量
global SUPPORT_ROT_W, SUPPORT_ROT_X, SUPPORT_ROT_Y, SUPPORT_ROT_Z

# 全局变量
global support_scale_tuple, original_support_height, \
        support_position_z, original_box_z, \
        scaled_support_height, height_reduction, \
        adjusted_box_z, adjusted_support_z

import sys
import termios
import tty
import select

theta_deg_global = random.uniform(-60, 60)
class keyboardlinstener(object):
    def __init__(self):
        super(keyboardlinstener,self).__init__()
        self.key_val = ""
        self.update = False

    def getKey(self, key_timeout):
        settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ' '
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

def generate_box_support_rotation():
    """
    生成box_support支架的随机旋转
    """
    global theta_deg_global
    global SUPPORT_ROT_W, SUPPORT_ROT_X, SUPPORT_ROT_Y, SUPPORT_ROT_Z
    
    # 生成随机旋转角度（-45到45度）
    theta_deg = theta_deg_global
    theta_rad = np.deg2rad(theta_deg)  # 角度取反
    
    # 基础四元数（算法零点）
    base_rot = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
    
    # 创建绕Z轴旋转的四元数（修改此处）
    cos_half = np.cos(theta_rad/2)
    sin_half = np.sin(theta_rad/2)
    z_rot = (cos_half, 0.0, 0.0, sin_half)  # 绕Z轴旋转 (w, x, y, z)
    
    # 四元数乘法（基础旋转 * Z轴旋转）
    w0, x0, y0, z0 = base_rot
    wr, xr, yr, zr = z_rot  # 使用z_rot变量
    
    SUPPORT_ROT_W = w0*wr - x0*xr - y0*yr - z0*zr
    SUPPORT_ROT_X = w0*xr + x0*wr + y0*zr - z0*yr 
    SUPPORT_ROT_Y = w0*yr - x0*zr + y0*wr + z0*xr
    SUPPORT_ROT_Z = w0*zr + x0*yr - y0*xr + z0*wr

def generate_box_height_reduction():
    """
    生成箱子的高度缩减，使用随机种子确定一致的高度
    """
    global support_scale_tuple, original_support_height, \
        support_position_z, original_box_z, \
        scaled_support_height, height_reduction, \
        adjusted_box_z, adjusted_support_z, random_seed
    
    # 使用随机种子生成一个确定的随机高度缩放比例
    # 使用random_seed的哈希值来确保即使是相同的种子也能生成不同的高度
    r = random.Random(random_seed)  # 创建一个基于种子的随机数生成器
    # 设置高度缩放范围，例如0.6到0.95之间
    scale_height_rate = r.uniform(0.7, 1.0)
    support_scale_tuple = (1.0, 1.0, scale_height_rate)
    original_support_height = 1.0
    support_position_z = 0.8
    original_box_z = 0.94057
    scaled_support_height = original_support_height * support_scale_tuple[2]
    height_reduction = original_support_height - scaled_support_height
    # Use the height_reduction variable for the box
    adjusted_box_z = original_box_z - height_reduction * 0.8  # 防止与地面碰撞

    # First, calculate an adjusted support position
    adjusted_support_z = support_position_z - height_reduction * 0.8  # 防止与地面碰撞
    
    # 可以记录使用的高度值，便于调试
    print(f"Using box height scale: {scale_height_rate:.3f} (seed: {random_seed})")

def generate_box_params(radius=2.5, height=0.8):
    """
    生成两个箱子的位置和随机旋转...
    """
    global theta_deg_global
    global ORIGIN_POS_X, ORIGIN_POS_Y, ORIGIN_POS_Z, TARGET_POS_X, TARGET_POS_Y, TARGET_POS_Z
    global ORIGIN_ROT_W, ORIGIN_ROT_X, ORIGIN_ROT_Y, ORIGIN_ROT_Z  # 新增全局变量

    # 随机生成角度（弧度）
    angle = random.uniform(0, 2 * np.pi)
        
    # 随机生成距离（在0到radius范围内）
    distance = random.uniform(1.0, radius)  # 最小距离设为1米，避免太靠近中心
        
    # 计算第一个箱子的位置
    x1 = distance * np.cos(angle)
    y1 = distance * np.sin(angle)
        
    # 计算对侧箱子的位置（相差180度）
    x2 = -x1
    y2 = -y1

    ORIGIN_POS_X = x1
    ORIGIN_POS_Y = y1
    ORIGIN_POS_Z = height

    TARGET_POS_X = x2
    TARGET_POS_Y = y2
    TARGET_POS_Z = height

    # 生成随机旋转角度（-45到45度）
    theta_deg = theta_deg_global
    theta_rad = np.deg2rad(theta_deg)
    
    # 基础四元数（算法零点）
    base_rot = (0.5, 0.5, -0.5, -0.5)  # (w, x, y, z)
    
    # 创建绕Y轴旋转的四元数
    cos_half = np.cos(theta_rad/2)
    sin_half = np.sin(theta_rad/2)
    y_rot = (cos_half, 0.0, sin_half, 0.0)  # (w, x, y, z)
    
    # 四元数乘法（基础旋转 * Y轴旋转）
    w0, x0, y0, z0 = base_rot
    wr, xr, yr, zr = y_rot
    
    ORIGIN_ROT_W = w0*wr - x0*xr - y0*yr - z0*zr
    ORIGIN_ROT_X = w0*xr + x0*wr + y0*zr - z0*yr 
    ORIGIN_ROT_Y = w0*yr - x0*zr + y0*wr + z0*xr
    ORIGIN_ROT_Z = w0*zr + x0*yr - y0*xr + z0*wr

    generate_box_support_rotation()  
    generate_box_height_reduction()  

import json

# 获取包路径
rospack = rospkg.RosPack()
ISAAC_SIM_PATH = rospack.get_path('rl_sac_env_isaac_lab')
ASSETS_PATH = os.path.join(ISAAC_SIM_PATH, 'Assets')

class KuavoRobotController():
    """
    # 45 - 机器人
    [2, 3, 7, 8, 12, 13, 
    16, 17, 20, 21, 24, 25, 
    26, 27] # 手
    zarm_l1_joint / zarm_r1_joint / zarm_l2_joint / zarm_r2_joint / zarm_l3_joint / zarm_r3_joint 
    zarm_l4_joint / zarm_r4_joint / zarm_l5_joint / zarm_r5_joint / zarm_l6_joint / zarm_r6_joint
    zarm_l7_joint / zarm_r7_joint
    
    [0,  1,  5,  6, 10, 11, 
    14, 15, 18, 19, 22, 23] # 脚 
    leg_l1_joint / leg_r1_joint / leg_l2_joint / leg_r2_joint / leg_l3_joint / leg_r3_joint 
    leg_l4_joint / leg_r4_joint / leg_l5_joint / leg_r5_joint / leg_l6_joint / leg_r6_joint

    [4, 9] # 头
    """
    def __init__(self):
        # 添加对scene的引用
        self.scene = None
        self.joint_cmd = None
        # state/cmd
        self.robot_sensor_data_pub = rospy.Publisher('/sensors_data_raw', sensorsData, queue_size=1)
        self.robot_joint_cmd_sub = rospy.Subscriber('/joint_cmd', jointCmd, self.joint_cmd_callback, queue_size=1)

        #  仿真开始标志
        self.sim_running = True
        self.sim_start_srv = rospy.Service('sim_start', SetBool, self.sim_start_callback)

        # 接收到新的命令
        self.new_cmd_received = False

        # Load joint configurations from JSON file
        config_path = os.path.join(os.path.dirname(__file__), "config", "joint_name.json")
        try:
            with open(config_path, 'r') as f:
                joint_config = json.load(f)
                self.wheel_joints = joint_config["wheel_joints"]
        except Exception as e:
            print(f"Error loading joint configuration: {e}")
            # Fallback to default values if JSON loading fails
            self.wheel_joints = []

        # Initialize empty indices lists
        self._wheel_idx = []

        # Initialize CV bridge for camera data conversion
        self.bridge = CvBridge()
        
        # Initialize camera publishers
        self.rgb_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=1)

        # 添加命令缓冲队列和锁
        self.cmd_queue = queue.Queue(maxsize=1)  # 只保留最新命令
        self.cmd_lock = threading.Lock()

        # 添加箱子位姿发布者
        self.box_origin_pub = rospy.Publisher('/box_origin_pose', PoseStamped, queue_size=1, latch=True)
        self.convert_shelves_pub = rospy.Publisher('/box/convert_shelves_pose', PoseStamped, queue_size=1, latch=True)
        self.box_support_pub = rospy.Publisher('/box_support_pose', PoseStamped, queue_size=1, latch=True)

        # Add the reset scene service
        self.reset_scene_srv = rospy.Service('/isaac_lab_reset_scene', resetIsaaclab, self.reset_scene_callback)

        # 添加机器人位姿发布者
        self.robot_pose_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=1)
        self.goal_pose_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=1)
        
        # 初始化目标点参数
        self.target_points = {
            'A': (-23.15, 11.57),
            'B': (-17.91, 11.57),
            'C': (-12.91, 11.57)
        }
        self.current_target = 'B'  # 默认目标点
        
        # 添加目标点服务
        self.target_srv = rospy.Service('/set_target_point', SetTargetPoint, self.handle_target_point)

    def sim_start_callback(self, req):
        """
        仿真启动服务的回调函数
        Args:
            req: SetBool请求，data字段为True表示启动仿真，False表示停止仿真
        Returns:
            SetBoolResponse: 服务响应
        """
        response = SetBoolResponse()
        
        self.sim_running = req.data

        if req.data:
            rospy.loginfo("Simulation started")
        else:
            rospy.loginfo("Simulation stopped")
        
        response.success = True
        response.message = "Simulation control successful"
        
        return response

    def setup_joint_indices(self, joint_names):
        """Setup joint indices based on joint names"""
        # Find indices for arm joints
        self._wheel_idx = [i for i, name in enumerate(joint_names) if name in self.wheel_joints]

    def joint_cmd_callback(self, joint_cmd):
        """处理接收到的关节力矩命令"""
        with self.cmd_lock:
            # 清空队列确保只保留最新命令
            while not self.cmd_queue.empty():
                try:
                    self.cmd_queue.get_nowait()
                except queue.Empty:
                    continue
            self.cmd_queue.put(joint_cmd)

    def update_sensor_data(self, ang_vel_b, lin_acc_b, quat_w, joint_pos, joint_vel, applied_torque, joint_acc, sim_time):
        """
        lin_vel_b = scene["imu_base"].data.lin_vel_b.tolist()  # 线速度
        ang_vel_b = scene["imu_base"].data.ang_vel_b.tolist()  # 角速度
        lin_acc_b = scene["imu_base"].data.lin_acc_b.tolist()  # 线加速度
        ang_acc_b = scene["imu_base"].data.ang_acc_b.tolist()  # 角加速度
        """
        # IMU数据组合
        sensor_data = sensorsData()

        # 使用仿真时间dt更新状态时间
        current_time = rospy.Time.from_sec(float(sim_time))
        sensor_data.header.stamp = current_time
        sensor_data.header.frame_id = "world"  # 设置适当的frame_id
        sensor_data.sensor_time = current_time

        # IMU数据
        sensor_data.imu_data.gyro.x = ang_vel_b[0]   # ang_vel
        sensor_data.imu_data.gyro.y = ang_vel_b[1]  # ang_vel
        sensor_data.imu_data.gyro.z = ang_vel_b[2]  # ang_vel
        sensor_data.imu_data.acc.x = lin_acc_b[0]  # lin_acc
        sensor_data.imu_data.acc.y = lin_acc_b[1]  # lin_acc
        sensor_data.imu_data.acc.z = lin_acc_b[2]  # lin_acc

        # sensor_data.imu_data.free_acc.x = lin_acc_b[0]  # lin_acc
        # sensor_data.imu_data.free_acc.y = lin_acc_b[1]  # lin_acc
        # sensor_data.imu_data.free_acc.z = lin_acc_b[2]  # lin_acc

        sensor_data.imu_data.quat.w = quat_w[0]  # 旋转矩阵
        sensor_data.imu_data.quat.x = quat_w[1]  # 旋转矩阵
        sensor_data.imu_data.quat.y = quat_w[2]  # 旋转矩阵
        sensor_data.imu_data.quat.z = quat_w[3]  # 旋转矩阵

        # 关节数据赋值优化
        # 初始化数组
        sensor_data.joint_data.joint_q = [0.0] * 28
        sensor_data.joint_data.joint_v = [0.0] * 28
        sensor_data.joint_data.joint_vd = [0.0] * 28
        sensor_data.joint_data.joint_torque = [0.0] * 28

        # 腿部
        for i in range(len(self._leg_idx)//2):
            sensor_data.joint_data.joint_q[i] = joint_pos[self._leg_idx[2*i]]       
            sensor_data.joint_data.joint_q[i+6] = joint_pos[self._leg_idx[2*i+1]]   

            sensor_data.joint_data.joint_v[i] = joint_vel[self._leg_idx[2*i]]
            sensor_data.joint_data.joint_v[i+6] = joint_vel[self._leg_idx[2*i+1]]

            sensor_data.joint_data.joint_torque[i] = applied_torque[self._leg_idx[2*i]]
            sensor_data.joint_data.joint_torque[i+6] = applied_torque[self._leg_idx[2*i+1]]

            sensor_data.joint_data.joint_vd[i] = joint_acc[self._leg_idx[2*i]]
            sensor_data.joint_data.joint_vd[i+6] = joint_acc[self._leg_idx[2*i+1]]

        # 手部
        for i in range(len(self._arm_idx)//2):
            sensor_data.joint_data.joint_q[12+i] = joint_pos[self._arm_idx[2*i]]
            sensor_data.joint_data.joint_q[19+i] = joint_pos[self._arm_idx[2*i+1]]

            sensor_data.joint_data.joint_v[12+i] = joint_vel[self._arm_idx[2*i]]
            sensor_data.joint_data.joint_v[19+i] = joint_vel[self._arm_idx[2*i+1]]
            
            sensor_data.joint_data.joint_torque[12+i] = applied_torque[self._arm_idx[2*i]]
            sensor_data.joint_data.joint_torque[19+i] = applied_torque[self._arm_idx[2*i+1]]

            sensor_data.joint_data.joint_vd[12+i] = joint_acc[self._arm_idx[2*i]]
            sensor_data.joint_data.joint_vd[19+i] = joint_acc[self._arm_idx[2*i+1]]

        # 头部
        sensor_data.joint_data.joint_q[26] = joint_pos[self._head_idx[0]]
        sensor_data.joint_data.joint_q[27] = joint_pos[self._head_idx[1]]

        sensor_data.joint_data.joint_v[26] = joint_vel[self._head_idx[0]]
        sensor_data.joint_data.joint_v[27] = joint_vel[self._head_idx[1]]

        sensor_data.joint_data.joint_vd[26] = joint_acc[self._head_idx[0]]
        sensor_data.joint_data.joint_vd[27] = joint_acc[self._head_idx[1]]

        sensor_data.joint_data.joint_torque[26] = applied_torque[self._head_idx[0]]
        sensor_data.joint_data.joint_torque[27] = applied_torque[self._head_idx[1]]
        
        # 发布数据
        self.robot_sensor_data_pub.publish(sensor_data)

    def publish_camera_data(self, rgb_data, depth_data, sim_time):
        """
        Convert and publish camera data as ROS topics
        
        Args:
            rgb_data: RGB image data from Isaac Sim
            depth_data: Depth image data from Isaac Sim
            sim_time: Current simulation time
        """
        if rgb_data is None or depth_data is None:
            return
            
        # Create timestamp for messages
        current_time = rospy.Time.from_sec(float(sim_time))
        
        try:
            # Convert RGB tensor to numpy array (assuming NHWC format)
            rgb_np = rgb_data.cpu().numpy()[0]  # First batch item
            
            # Convert depth tensor to numpy array
            depth_np = depth_data.cpu().numpy()[0]  # First batch item
            
            # Normalize depth for visualization if needed
            # Assuming depth values are in meters, scale for better visualization
            # You may need to adjust this based on your specific depth data format
            depth_np_normalized = (depth_np * 1000).astype(np.uint16)  # Scale to mm and convert to uint16
            
            # Convert to ROS messages using CvBridge
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_np, encoding="rgb8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_np_normalized, encoding="16UC1")
            
            # Set headers
            rgb_msg.header.stamp = current_time
            rgb_msg.header.frame_id = "camera_frame"
            depth_msg.header.stamp = current_time
            depth_msg.header.frame_id = "camera_frame"
            
            # Create camera info message
            camera_info_msg = CameraInfo()
            camera_info_msg.header.stamp = current_time
            camera_info_msg.header.frame_id = "camera_frame"
            camera_info_msg.height = rgb_np.shape[0]
            camera_info_msg.width = rgb_np.shape[1]
            
            # Publish messages
            self.rgb_pub.publish(rgb_msg)
            self.depth_pub.publish(depth_msg)
            self.camera_info_pub.publish(camera_info_msg)
            
        except Exception as e:
            rospy.logerr(f"Error publishing camera data: {e}")

    def publish_box_poses(self, sim_time):
        global adjusted_box_z, adjusted_support_z
        """发布箱子位姿到ROS话题"""
        # 发布box_origin位姿
        box_origin_msg = PoseStamped()
        box_origin_msg.header.stamp = rospy.Time.from_sec(sim_time)
        box_origin_msg.header.frame_id = "world"
        box_origin_msg.pose.position.x = ORIGIN_POS_X
        box_origin_msg.pose.position.y = ORIGIN_POS_Y
        box_origin_msg.pose.position.z = adjusted_box_z  # 根据场景配置中的高度
        box_origin_msg.pose.orientation.x = ORIGIN_ROT_X
        box_origin_msg.pose.orientation.y = ORIGIN_ROT_Y
        box_origin_msg.pose.orientation.z = ORIGIN_ROT_Z
        box_origin_msg.pose.orientation.w = ORIGIN_ROT_W
        self.box_origin_pub.publish(box_origin_msg)

        # 发布convert_shelves位姿 - 修正为-90度的位置放置货架
        convert_shelves_msg = PoseStamped()
        convert_shelves_msg.header.stamp = rospy.Time.from_sec(sim_time)
        convert_shelves_msg.header.frame_id = "world"
        convert_shelves_msg.pose.position.x = TARGET_POS_X
        convert_shelves_msg.pose.position.y = TARGET_POS_Y 
        convert_shelves_msg.pose.position.z = adjusted_support_z  # 根据场景配置中的高度
        convert_shelves_msg.pose.orientation.x = 0.0 # 0.5
        convert_shelves_msg.pose.orientation.y = 0.0 # 0.5
        convert_shelves_msg.pose.orientation.z = -0.70711 # 0.5
        convert_shelves_msg.pose.orientation.w = 0.70711  # 0.5
        self.convert_shelves_pub.publish(convert_shelves_msg)

        # 新增支架位姿发布
        box_support_msg = PoseStamped()
        box_support_msg.header.stamp = rospy.Time.from_sec(sim_time)
        box_support_msg.header.frame_id = "world"
        box_support_msg.pose.position.x = ORIGIN_POS_X
        box_support_msg.pose.position.y = ORIGIN_POS_Y
        box_support_msg.pose.position.z = adjusted_support_z
        box_support_msg.pose.orientation.x = SUPPORT_ROT_X
        box_support_msg.pose.orientation.y = SUPPORT_ROT_Y
        box_support_msg.pose.orientation.z = SUPPORT_ROT_Z
        box_support_msg.pose.orientation.w = SUPPORT_ROT_W
        self.box_support_pub.publish(box_support_msg)

    def reset_scene_callback(self, req):
        global random_seed
        response = resetIsaaclabResponse()
        
        try:
            # 设置随机种子
            if req.data != 0:
                random_seed = req.data
                random.seed(random_seed)
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(random_seed)

            # 生成随机位置
            x_range = (-24.62, 4.5)
            y_range = (-17.32, 5.35)
            new_pos = [
                random.uniform(x_range[0], x_range[1]),
                random.uniform(y_range[0], y_range[1]),
                0.2  # Z坐标保持不变
            ]

            # 获取当前机器人状态
            root_state = self.scene["robot"].data.root_state_w.clone()
            
            # 更新位置并保持原有姿态
            root_state[:, 0:3] = torch.tensor(new_pos, device=root_state.device)
            
            # 应用新状态
            self.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            self.scene["robot"].write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
            
            # 重置关节状态
            self.scene["robot"].write_joint_state_to_sim(
                self.scene["robot"].data.default_joint_pos.clone(),
                self.scene["robot"].data.default_joint_vel.clone()
            )

            response.success = True
            response.message = f"机器人已重置到位置({new_pos[0]:.2f}, {new_pos[1]:.2f})"
            
        except Exception as e:
            rospy.logerr(f"重置失败: {str(e)}")
            response.success = False
            response.message = f"错误: {str(e)}"
        
        return response

    def handle_target_point(self, req):
        if req.point_index in self.target_points:
            self.current_target = req.point_index
            return SetTargetPointResponse(True, f"Target set to {req.point_index}")
        return SetTargetPointResponse(False, "Invalid target point")

    def publish_robot_pose(self):
        """发布机器人位姿到/robot_pose话题"""
        root_state = self.scene["robot"].data.root_state_w.clone()
        pos = root_state[:, 0:3]
        quat = root_state[:, 3:7]
        
        # 创建PoseStamped消息
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        
        # 设置位姿信息
        pose_msg.pose.position.x = pos[0, 0].item()
        pose_msg.pose.position.y = pos[0, 1].item()
        pose_msg.pose.position.z = pos[0, 2].item()
        pose_msg.pose.orientation.w = quat[0, 0].item()
        pose_msg.pose.orientation.x = quat[0, 1].item()
        pose_msg.pose.orientation.y = quat[0, 2].item()
        pose_msg.pose.orientation.z = quat[0, 3].item()
        
        self.robot_pose_pub.publish(pose_msg)

    def publish_goal_pose(self):
        """发布当前目标点到/goal_pose话题"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "world"
        
        x, y = self.target_points[self.current_target]
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0  # 假设Z坐标为0
        
        self.goal_pose_pub.publish(goal_msg)

# 在类定义前添加初始化调用
generate_box_params()  # 初始化所有全局旋转变量

@configclass
class BipedSceneCfg(InteractiveSceneCfg):
    """Scene configuration with biped robot."""
    global ORIGIN_POS_X, ORIGIN_POS_Y, ORIGIN_POS_Z, TARGET_POS_X, TARGET_POS_Y, TARGET_POS_Z
    global SUPPORT_ROT_W, SUPPORT_ROT_X, SUPPORT_ROT_Y, SUPPORT_ROT_Z
    global ORIGIN_ROT_W, ORIGIN_ROT_X, ORIGIN_ROT_Y, ORIGIN_ROT_Z
    # global USE_CAMERA_FLAG_BOOL
    global MESH_ID
    global MESH_SCENE_NAME
    global support_scale_tuple, original_support_height, \
        support_position_z, original_box_z, \
        scaled_support_height, height_reduction, \
        adjusted_box_z, adjusted_support_z
    # # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane", 
    #     spawn=sim_utils.GroundPlaneCfg(
    #         color=(1.0, 1.0, 1.0)  # RGB值设为白色 (1.0, 1.0, 1.0)
    #     )
    # )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 使用你的机器人配置
    robot = KINOVA_ROBOTIQ  # 或者使用 KINOVA_ROBOTIQ_HPD
    robot.prim_path = "{ENV_REGEX_NS}/Robot"

    # # 添加IMU传感器
    # """
    #     固定的偏置为 +9.81 m/s^2
    # """
    # # imu_base = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link", gravity_bias=(0, 0, 0), debug_vis=True) # 消除了重力
    # imu_base = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link", debug_vis=True) # 没有消除重力

    """
        只针对测试场地当中的场景，只构建其中的一个转角90度部分
    """
    # 3D-GS
    mesh_scene = AssetBaseCfg(
        prim_path="/World/mesh_scene",  # 场景中mesh场景的路径
        spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_PATH, "full_warehouse.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=False,
                kinematic_enabled=False,
            max_depenetration_velocity=5.0,
        ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # 设置箱子的初始位置 (x, y, z)
            rot=(1.0, 0.0, 0.0, 0.0),  # 设置箱子的初始旋转 (w, x, y, z)
        )
    )   

    # # 根据USE_CAMERA_FLAG_BOOL条件添加相机配置
    # if USE_CAMERA_FLAG_BOOL:
    #     camera = CameraCfg(
    #         prim_path="{ENV_REGEX_NS}/Robot/camera_base/d435i_front",
    #         debug_vis=True, 
    #         update_period=0.033,  # 约30 FPS
    #         height=480, # 720
    #         width=640, # 1280
    #         data_types=["rgb", "distance_to_image_plane", "normals"],
    #         spawn=sim_utils.PinholeCameraCfg(
    #             focal_length=1.88,            # D435实际焦距约为1.88mm
    #             horizontal_aperture=3.69,     # 根据87°FOV和1.88mm焦距计算
    #             vertical_aperture=2.09,       # 根据58°FOV和1.88mm焦距计算
    #             clipping_range=(0.105, 10.0), # D435深度范围
    #         ),
    #         offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    #     )

class KeyboardListenerThread(threading.Thread):
    def __init__(self, key_listener):
        super().__init__()
        self.key_listener = key_listener
        self.current_key = ''
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def run(self):
        while not self.stopped():
            key = self.key_listener.getKey(0.1)
            with self.lock:
                if key != ' ':
                    self.current_key = key

    def get_current_key(self):
        with self.lock:
            key = self.current_key
            self.current_key = ''  # 清空已读取的按键
            return key

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, kuavo_robot: KuavoRobotController):
    """Run the simulator."""
    global FIRST_TIME_FLAG
    global DECIMATION_RATE
    global RESET_WORK_RUNNABLE
    global support_scale_tuple
    # # 初始化键盘监听线程
    # key_listener = keyboardlinstener()
    # keyboard_thread = KeyboardListenerThread(key_listener)
    # keyboard_thread.start()

    # Add timing variables
    timing_stats = {
        "sensor_data": 0.0,
        "process_commands": 0.0,
        "write_to_sim": 0.0,
        "physics_step": 0.0,
        "scene_update": 0.0,
        "total_loop": 0.0
    }
    timing_counter = 0
    timing_print_interval = 100  # Print timing stats every 100 iterations
    
    # 设置scene引用
    kuavo_robot.scene = scene
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    print("sim_dt: ", sim_dt)
    sim_time = 0.0
    count = 0

    # 控制频率
    rate = rospy.Rate(500)  # 500Hz

    body_names = scene["robot"].data.body_names   # 包含所有fix固定的joint
    joint_names = scene["robot"].data.joint_names # 只包含可活动的joint
    default_mass = scene["robot"].data.default_mass.tolist()[0] # 检查mass质量
    total_mass = 0.0
    for i in range(len(scene["robot"].data.body_names)):
        total_mass += default_mass[i]

    kuavo_robot.setup_joint_indices(joint_names)
    print("joint_names: ", joint_names) 
    print("body_names: ", body_names)
    print("total_mass: ", total_mass)

    # 设置机器人初始状态
    root_state = scene["robot"].data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    scene["robot"].write_root_pose_to_sim(root_state[:, :7])
    scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
    # set joint positions
    joint_pos = scene["robot"].data.default_joint_pos.clone()
    joint_vel = scene["robot"].data.default_joint_vel.clone()
    scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
    # clear internal buffers
    scene.reset()
    print("[INFO]: Setting initial robot state...")
    
    # Simulate physics
    while simulation_app.is_running():
        loop_start_time = time.time()

        # 重置环境功能
        if RESET_WORK_RUNNABLE:
            if scene is not None:
                # 直接尝试获取实体，捕获KeyError异常
                try:
                    box_origin = scene["box_origin"]
                    box_support = scene["box_support"] 
                    convert_shelves = scene["convert_shelves"]
                except KeyError as e:
                    available = list(scene.keys())
                    rospy.logerr(f"Scene entity {e} not found. Available entities: {available}")
                # 场景设备 
                device = scene.device
                # 更新box_origin位姿
                try:    
                    origin_positions = torch.tensor([[ORIGIN_POS_X, ORIGIN_POS_Y, 0.9]], device=device)
                    origin_orientations = torch.tensor([[ORIGIN_ROT_W, ORIGIN_ROT_X, ORIGIN_ROT_Y, ORIGIN_ROT_Z]], device=device)
                    box_origin.set_world_poses(positions=origin_positions, orientations=origin_orientations)
                except Exception as e:
                    rospy.logerr(f"Error updating box_origin: {str(e)}")
                # 更新box_support位姿
                try:
                    support_positions = torch.tensor([[ORIGIN_POS_X, ORIGIN_POS_Y, adjusted_support_z]], device=device)
                    support_orientations = torch.tensor([[SUPPORT_ROT_W, SUPPORT_ROT_X, SUPPORT_ROT_Y, SUPPORT_ROT_Z]], device=device)
                    box_support.set_world_poses(positions=support_positions, orientations=support_orientations)
                    box_support.set_local_scales(torch.tensor([[1.0, 1.0, support_scale_tuple[2]]], device=device))
                except Exception as e:
                    rospy.logerr(f"Error updating box_support: {str(e)}")
                # 重置场景
                scene.reset()
                rospy.loginfo("Successfully reset scene objects")
                # 重新复原场景状态 
                RESET_WORK_RUNNABLE = False

        # 判断是否初始化场景加载
        if not kuavo_robot.sim_running:
            rospy.sleep(1)
            rospy.loginfo("Waiting for simulation start signal...")
            continue

        # 更新传感器数据
        if not FIRST_TIME_FLAG:
            cmd_start_time = time.time()
            # 从队列获取最新命令
            current_cmd = None
            with kuavo_robot.cmd_lock:
                if not kuavo_robot.cmd_queue.empty():
                    try:
                        current_cmd = kuavo_robot.cmd_queue.get_nowait()
                    except queue.Empty:
                        pass
            
            if current_cmd is not None:
                # 使用current_cmd代替原来的joint_cmd
                full_torque_cmd = [0.0] * len(scene["robot"].data.joint_names)
                                
                # 差速轮
                full_torque_cmd[kuavo_robot._wheel_idx[0]] = current_cmd.tau[0]  # type: ignore # left_wheel_joint
                full_torque_cmd[kuavo_robot._wheel_idx[1]] = current_cmd.tau[1]  # type: ignore # right_wheel_joint

                # 将力矩命令转换为tensor并发送给机器人
                torque_tensor = torch.tensor([full_torque_cmd], device=scene["robot"].device)
                # scene["robot"].set_joint_effort_target(torque_tensor)
                scene["robot"].set_joint_velocity_target(torque_tensor)
            
            timing_stats["process_commands"] += time.time() - cmd_start_time


        # write data to sim
        write_start_time = time.time()
        scene.write_data_to_sim()
        timing_stats["write_to_sim"] += time.time() - write_start_time
        
        # perform step
        step_start_time = time.time()
        if count % DECIMATION_RATE == 0:
            sim.render()
        else:
            sim.step(False)  # 物理步往前但是不渲染
            scene["robot"].update(sim_dt)
        timing_stats["physics_step"] += time.time() - step_start_time
        
        # update sim-time
        sim_time += sim_dt
        count += 1
        
        # update buffers
        update_start_time = time.time()
        scene.update(sim_dt)
        timing_stats["scene_update"] += time.time() - update_start_time
        
        # 计算总循环时间
        timing_stats["total_loop"] += time.time() - loop_start_time
        timing_counter += 1
                    
        # 第一帧结束
        FIRST_TIME_FLAG = False

        # 数据已经用完
        kuavo_robot.new_cmd_received = False

        # # 发布箱子位姿
        # kuavo_robot.publish_box_poses(sim_time)

        # 发布位姿信息
        kuavo_robot.publish_robot_pose()
        kuavo_robot.publish_goal_pose()

        # 控制频率
        rate.sleep()

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.002, 
        device=args_cli.device,
        # physx=sim_utils.PhysxCfg(
        #     solver_type="TGS",  # CPU通常使用PGS求解器 | TGS
        # )
    )
    
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    
    # 关闭gpu动力学 
    sim._physics_context.enable_gpu_dynamics(False)
    
    # Create scene
    scene_cfg = BipedSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 创建机器人控制实例
    kuavo_robot = KuavoRobotController()

    # Reset simulation
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene, kuavo_robot)

if __name__ == "__main__":
    main()
    simulation_app.close()
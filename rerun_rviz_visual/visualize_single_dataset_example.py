#!/usr/bin/env python3
"""
Enhanced LeRobot Dataset Visualization with 3D Robot and Camera Views

This script provides comprehensive visualization of LeRobot datasets with:
- 3D robot arm visualization using URDF
- Camera pose visualization with frustums
- Real-time joint angle monitoring
- Image projection in 3D space

Usage examples:
    # Basic usage with local viewer
    python visualize_single_dataset_example.py --repo-id your-dataset --episode-index 0
    
    # With custom URDF file
    python visualize_single_dataset_example.py --repo-id your-dataset --episode-index 0 --urdf-path /path/to/robot.urdf
    
    # Save to file
    python visualize_single_dataset_example.py --repo-id your-dataset --episode-index 0 --save --output-dir ./output
"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator
import os
import xml.etree.ElementTree as ET

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Try to import trimesh for mesh processing
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logging.warning("trimesh not available. Install it with: pip install trimesh")

# Try to import rosbag (optional)
try:
    import rosbag
    ROSBAG_AVAILABLE = True
except ImportError:
    ROSBAG_AVAILABLE = False
    logging.warning("rosbag not available. Install it with: pip install rosbag")



class EpisodeSampler(torch.utils.data.Sampler):
    """Episode sampler for LeRobot datasets"""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


class DatasetWrapped():
    """
        基础数据加载类，用于加载不同的数据集

        支持 LerobotDataset 和 rosbag

        参数：
            dataset_path: 数据集路径
            dataset_mode: 数据集模式 ("lerobot", "bag")
            episode_index: 要可视化的episode索引 (仅对lerobot有效)
            root: 数据集根目录
            tolerance_s: 时间戳容忍度
    """
    def __init__(self, dataset_path: str, dataset_mode: str, episode_index: int = 0, 
                 root: Path = None, tolerance_s: float = 1e-4):
        self.dataset_path = dataset_path
        self.dataset_mode = dataset_mode
        self.episode_index = episode_index
        self.root = root
        self.tolerance_s = tolerance_s
        self.dataset = None
        self.dataloader = None
        
        self.init_dataset_loader()

    def init_dataset_loader(self):
        """初始化数据集加载器"""
        if self.dataset_mode == "lerobot":
            self.dataset = LeRobotDataset(self.dataset_path, root=self.root, tolerance_s=self.tolerance_s)
            logging.info(f"Loaded LeRobot dataset: {self.dataset_path}")
            logging.info(f"Dataset has {len(self.dataset.episode_data_index)} episodes")
        elif self.dataset_mode == "bag" and ROSBAG_AVAILABLE:
            # TODO: 实现真正的rosbag加载
            logging.warning("ROS bag loading not implemented yet")
        else:
            raise ValueError(f"Unsupported dataset mode: {self.dataset_mode}")

    def create_episode_dataloader(self, batch_size: int = 32, num_workers: int = 0):
        """创建episode数据加载器"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        
        if self.dataset_mode == "lerobot":
            episode_sampler = EpisodeSampler(self.dataset, self.episode_index)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                sampler=episode_sampler,
            )
            return self.dataloader
        else:
            raise ValueError(f"Episode dataloader not supported for mode: {self.dataset_mode}")

    def get_dataset_data(self):
        """获取数据集里面的所有数据"""
        return self.dataset

    def get_dataset_robot_data(self, state_data):
        """
        获取数据集里面关于机器人的数据
        
        Args:
            state_data: 状态数据数组
            
        Returns:
            arm_joints: 14个arm joint角度，如果提取失败返回None
        """
        # 尝试从不同的状态配置中提取arm joints
        arm_joints = None
        if len(state_data) >= 28:
            # For WBC mode: arm joints are typically at indices 6-19 (14 joints total)
            arm_joints = state_data[6:20]
        elif len(state_data) >= 21:
            # For other configurations, arm joints might be at different positions
            # Try to find 14 consecutive values that look like joint angles
            for start_idx in range(len(state_data) - 13):
                potential_joints = state_data[start_idx:start_idx+14]
                # Check if values are in reasonable joint angle range (-π to π)
                if np.all(np.abs(potential_joints) < 4.0):  # Reasonable joint angle range
                    arm_joints = potential_joints
                    break
        
        return arm_joints

    def get_dataset_image_data(self):
        """获取数据集里面关于图像的数据"""
        if self.dataset and hasattr(self.dataset, 'meta'):
            return self.dataset.meta.camera_keys
        return []

class RobotVisualizer():
    """
        机器人可视化类

        urdf_path : urdf指向的路径

        robot_mode : 机器人模式
            robot_mode = "fixed_single"      # 固定基单机械臂
            robot_mode = "fixed_dual"        # 固定基双机械臂
            robot_mode = "mobile_dual"       # 浮动基双机械臂
        
        camera_mode: 相机模式
            camera_mode = "camera_mode_a" # 只有第一人称头部相机
            camera_mode = "camera_mode_b" # 第一人称头部相机 + 腰部第一人称相机
            camera_mode = "camera_mode_c" # 第一人称头部相机 + 第三视角相机 
    """
    def __init__(self, urdf_path: str = None, robot_mode: str = "fixed_dual", camera_mode: str = "camera_mode_a"):
        self.urdf_path = urdf_path
        self.robot_mode = robot_mode
        self.camera_mode = camera_mode
        
        # URDF相关属性
        self.urdf_root = None
        self.links = {}
        self.joints = {}
        
        # 机器人关节配置
        self.joint_names_left = [f'zarm_l{i}_joint' for i in range(1, 8)]
        self.joint_names_right = [f'zarm_r{i}_joint' for i in range(1, 8)]
        self.all_joint_names = self.joint_names_left + self.joint_names_right
        
        # 头部关节角度 (可调整以匹配机器人头部姿态)
        self.head_joint_angles = np.array([0.0, np.radians(30)])  # 0° yaw, 30° pitch

        self.init_robot_visualizer()

    def init_robot_visualizer(self):
        """加载机器人可视化"""
        # 自动查找URDF文件
        if self.urdf_path is None:
            self.urdf_path = self._find_urdf_automatically()
        
        # 加载URDF
        if self.urdf_path:
            self.load_urdf(self.urdf_path)
        else:
            logging.warning("URDF not loaded - using simplified visualization")

    def _find_urdf_automatically(self):
        """自动查找URDF文件"""
        possible_urdf_paths = [
            # Primary path - user specified location
            "/home/lab/RL/src/biped_s45/urdf/biped_s45.urdf",
            # Relative paths from project root
            "./src/biped_s45/urdf/biped_s45.urdf",
            # Fallback paths
            "./gym_hil/assets/biped_s45.urdf",
            "./assets/biped_s45.urdf"
        ]
        
        for path in possible_urdf_paths:
            if os.path.exists(path):
                logging.info(f"Found URDF file at: {path}")
                return path
        return None

    def set_robot_joint_position(self, joint_angles_14, frame_index=0):
        """
        设置机器人关节位置并进行可视化
        
        Args:
            joint_angles_14: 14个关节角度数组 [left_arm_7, right_arm_7]
            frame_index: 当前帧索引
        """
        if len(joint_angles_14) < 14:
            logging.warning(f"Expected 14 joint angles, got {len(joint_angles_14)}")
            return
            
        if self.urdf_root is not None:
            self._visualize_robot_with_urdf(joint_angles_14, frame_index)
        else:
            self._visualize_robot_simplified(joint_angles_14, frame_index)

    def load_urdf(self, urdf_path: str):
        """使用xml.etree.ElementTree加载URDF文件"""
        try:
            if os.path.exists(urdf_path):
                tree = ET.parse(urdf_path)
                self.urdf_root = tree.getroot()
                
                # 解析links
                for link_elem in self.urdf_root.findall('link'):
                    link_name = link_elem.get('name')
                    self.links[link_name] = {
                        'name': link_name,
                        'visual': [],
                        'collision': []
                    }
                    
                    # 解析视觉元素
                    for visual_elem in link_elem.findall('visual'):
                        visual_data = self._parse_visual_element(visual_elem, urdf_path)
                        if visual_data:
                            self.links[link_name]['visual'].append(visual_data)
                
                # 解析joints
                for joint_elem in self.urdf_root.findall('joint'):
                    joint_name = joint_elem.get('name')
                    joint_type = joint_elem.get('type')
                    
                    parent_elem = joint_elem.find('parent')
                    child_elem = joint_elem.find('child')
                    origin_elem = joint_elem.find('origin')
                    axis_elem = joint_elem.find('axis')
                    
                    # 解析关节轴
                    axis = [0, 0, 1]  # default to Z-axis
                    if axis_elem is not None:
                        axis_str = axis_elem.get('xyz', '0 0 1')
                        axis = [float(x) for x in axis_str.split()]
                    
                    self.joints[joint_name] = {
                        'name': joint_name,
                        'type': joint_type,
                        'parent': parent_elem.get('link') if parent_elem is not None else None,
                        'child': child_elem.get('link') if child_elem is not None else None,
                        'origin': self._parse_origin(origin_elem) if origin_elem is not None else {'xyz': [0,0,0], 'rpy': [0,0,0]},
                        'axis': axis
                    }
                
                logging.info(f"Successfully loaded URDF from {urdf_path}")
                logging.info(f"Robot has {len(self.links)} links and {len(self.joints)} joints")
                return True
            else:
                logging.warning(f"URDF file not found at {urdf_path}")
                return False
        except Exception as e:
            logging.error(f"Failed to load URDF: {e}")
            return False

    def _parse_visual_element(self, visual_elem, urdf_path):
        """解析URDF视觉元素"""
        visual_data = {}
        
        # 解析几何体
        geometry_elem = visual_elem.find('geometry')
        if geometry_elem is not None:
            mesh_elem = geometry_elem.find('mesh')
            if mesh_elem is not None:
                filename = mesh_elem.get('filename')
                if filename:
                    # 转换相对路径为绝对路径
                    if not os.path.isabs(filename):
                        # 移除package://前缀
                        if filename.startswith('package://'):
                            filename = filename.replace('package://', '')
                            filename = filename.replace('kuavo_assets/models/biped_s45/', '')
                        
                        # 尝试相对于URDF目录解析
                        urdf_dir = os.path.dirname(urdf_path)
                        possible_paths = [
                            os.path.join(urdf_dir, filename),
                            os.path.join(urdf_dir, '..', filename),
                            os.path.join(urdf_dir, '..', 'meshes', os.path.basename(filename)),
                            os.path.join('/home/lab/RL/src/biped_s45/', filename),
                            os.path.join('/home/lab/RL/src/biped_s45/meshes/', os.path.basename(filename))
                        ]
                        
                        filename_found = None
                        for attempt_path in possible_paths:
                            attempt_path = os.path.normpath(attempt_path)
                            if os.path.exists(attempt_path):
                                filename_found = attempt_path
                                break
                        
                        if filename_found:
                            filename = filename_found
                        else:
                            logging.warning(f"Mesh file not found: {filename}")
                    
                    visual_data['mesh_file'] = filename
                    scale = mesh_elem.get('scale')
                    if scale:
                        visual_data['scale'] = [float(x) for x in scale.split()]
                    else:
                        visual_data['scale'] = [1.0, 1.0, 1.0]
        
        # 解析origin
        origin_elem = visual_elem.find('origin')
        if origin_elem is not None:
            visual_data['origin'] = self._parse_origin(origin_elem)
        else:
            visual_data['origin'] = {'xyz': [0,0,0], 'rpy': [0,0,0]}
        
        return visual_data if visual_data else None

    def _parse_origin(self, origin_elem):
        """解析URDF origin元素"""
        xyz = origin_elem.get('xyz', '0 0 0')
        rpy = origin_elem.get('rpy', '0 0 0')
        
        return {
            'xyz': [float(x) for x in xyz.split()],
            'rpy': [float(x) for x in rpy.split()]
        }

    def _visualize_robot_simplified(self, joint_angles_14, frame_index):
        """简化的机器人可视化(不使用URDF)"""
        if len(joint_angles_14) < 14:
            return
        
        left_joints = joint_angles_14[:7]
        right_joints = joint_angles_14[7:14]
        
        # 计算简化的正运动学
        left_positions = self._compute_simplified_fk(left_joints, 'left')
        right_positions = self._compute_simplified_fk(right_joints, 'right')
        
        # 记录机器人基座/躯干
        torso_pos = np.array([0.0, 0.0, 0.4])
        rr.log("robot_view/robot/torso", rr.Points3D([torso_pos], radii=[0.05], colors=[[100, 100, 100]]))
        
        # 记录左臂
        for i, pos in enumerate(left_positions):
            rr.log(f"robot_view/robot/left_arm/joint_{i}", 
                   rr.Points3D([pos], radii=[0.02], colors=[[255, 100, 100]]))
        
        # 记录右臂
        for i, pos in enumerate(right_positions):
            rr.log(f"robot_view/robot/right_arm/joint_{i}", 
                   rr.Points3D([pos], radii=[0.02], colors=[[100, 100, 255]]))
        
        # 记录手臂链接为线段
        if len(left_positions) > 1:
            left_line_points = np.array(left_positions)
            rr.log("robot_view/robot/left_arm/skeleton", 
                   rr.LineStrips3D([left_line_points], colors=[[255, 100, 100]], radii=[0.01]))
        
        if len(right_positions) > 1:
            right_line_points = np.array(right_positions)
            rr.log("robot_view/robot/right_arm/skeleton", 
                   rr.LineStrips3D([right_line_points], colors=[[100, 100, 255]], radii=[0.01]))
        
        # 连接躯干到肩膀
        if len(left_positions) > 0 and len(right_positions) > 0:
            torso_connections = np.array([
                [torso_pos, left_positions[0]],   # 躯干到左肩
                [torso_pos, right_positions[0]]   # 躯干到右肩
            ])
            rr.log("robot_view/robot/torso/connections", 
                   rr.LineStrips3D(torso_connections, colors=[[150, 150, 150]], radii=[0.015]))

    def _compute_simplified_fk(self, joint_angles, arm_side='left'):
        """简化的正运动学计算"""
        if arm_side == 'left':
            base_pos = np.array([0.0, 0.15, 0.4])  # 左肩偏移
        else:
            base_pos = np.array([0.0, -0.15, 0.4])  # 右肩偏移
        
        positions = [base_pos]
        current_pos = base_pos.copy()
        
        # 简化的链接向量
        link_vectors = [
            np.array([0, 0, -0.1]),   # 肩部到上臂
            np.array([0, 0, -0.15]),  # 上臂段1
            np.array([0, 0, -0.1]),   # 上臂段2
            np.array([0, 0, -0.25]),  # 前臂
            np.array([0, 0, -0.05]),  # 手腕段1
            np.array([0, 0, -0.05]),  # 手腕段2
            np.array([0, 0, -0.1]),   # 末端执行器
        ]
        
        # 应用简化变换
        for i, (angle, link_vec) in enumerate(zip(joint_angles[:7], link_vectors)):
            cos_a = np.cos(angle * 0.5)
            sin_a = np.sin(angle * 0.5)
            
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            
            rotated_link = rot_matrix @ link_vec
            current_pos += rotated_link
            positions.append(current_pos.copy())
        
        return positions

    def _visualize_robot_with_urdf(self, joint_angles_14, frame_index):
        """使用URDF和网格进行机器人可视化"""
        if not self.urdf_root:
            self._visualize_robot_simplified(joint_angles_14, frame_index)
            return
        
        try:
            # 获取关节配置
            joint_config = self._get_joint_configuration(joint_angles_14)
            if not joint_config:
                logging.warning("Could not create joint configuration")
                self._visualize_robot_simplified(joint_angles_14, frame_index)
                return
            
            # 计算正运动学
            link_transforms = self._compute_forward_kinematics(joint_config)
            
            # 记录基座链接
            rr.log("robot_view/robot/base_link", 
                   rr.Transform3D(translation=[0, 0, 0],
                                rotation=rr.Quaternion(xyzw=[0, 0, 0, 1]),
                                relation=rr.TransformRelation.ChildFromParent))
            
            # 记录基座坐标系
            base_transform = np.eye(4)
            self._log_coordinate_frame("robot_view/robot/base_link/frame", base_transform, scale=0.15)
            
            # 记录base_link网格
            if 'base_link' in self.links:
                for visual_data in self.links['base_link']['visual']:
                    if 'mesh_file' in visual_data and TRIMESH_AVAILABLE:
                        try:
                            mesh_file = visual_data['mesh_file']
                            if os.path.exists(mesh_file):
                                mesh = trimesh.load_mesh(mesh_file)
                                scale = visual_data.get('scale', [1.0, 1.0, 1.0])
                                if scale != [1.0, 1.0, 1.0]:
                                    mesh.vertices *= scale
                                
                                # 应用视觉原点变换
                                origin = visual_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                                visual_transform = np.eye(4)
                                visual_transform[:3, 3] = origin['xyz']
                                visual_transform[:3, :3] = self._rpy_to_rotation_matrix(origin['rpy'])
                                
                                # 变换网格顶点
                                vertices_homo = np.hstack([mesh.vertices, np.ones((mesh.vertices.shape[0], 1))])
                                transformed_vertices = (visual_transform @ vertices_homo.T).T[:, :3]
                                
                                rr.log("robot_view/robot/base_link/mesh",
                                       rr.Mesh3D(
                                           vertex_positions=transformed_vertices.astype(np.float32),
                                           triangle_indices=mesh.faces.astype(np.uint32)
                                       ))
                                
                                logging.info(f"Loaded base mesh: vertices={mesh.vertices.shape}, "
                                           f"faces={mesh.faces.shape}, bounds={mesh.bounds}")
                                break
                        except Exception as e:
                            logging.error(f"Failed to load base mesh: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                # 如果没有base_link网格，创建一个简单的基座可视化
                base_pos = np.array([0.0, 0.0, 0.2])
                rr.log("robot_view/robot/base_link/simple_base", 
                       rr.Points3D([base_pos], radii=[0.1], colors=[[150, 150, 150]]))
            
            # 记录手臂链接和网格
            arm_chains = {
                'left': ['zarm_l1_link', 'zarm_l2_link', 'zarm_l3_link', 'zarm_l4_link', 
                        'zarm_l5_link', 'zarm_l6_link', 'zarm_l7_link'],
                'right': ['zarm_r1_link', 'zarm_r2_link', 'zarm_r3_link', 'zarm_r4_link', 
                         'zarm_r5_link', 'zarm_r6_link', 'zarm_r7_link']
            }
            
            for arm_side, chain in arm_chains.items():
                arm_color = [255, 100, 100] if arm_side == 'left' else [100, 100, 255]
                
                for link_name in chain:
                    if link_name in link_transforms:
                        transform = link_transforms[link_name]
                        translation = transform[:3, 3]
                        
                        # 为每个链接添加简单的可视化球体
                        rr.log(f"robot_view/robot/{arm_side}_arm/{link_name}/sphere",
                               rr.Points3D([translation], 
                                         radii=[0.03], 
                                         colors=[arm_color]))
                        
                        # 记录每个关节/链接的坐标系
                        self._log_coordinate_frame(f"robot_view/robot/{arm_side}_arm/{link_name}/frame", 
                                                 transform, scale=0.15)
                        
                        # 加载和变换网格(如果可用)
                        self._load_link_mesh(link_name, transform, f"robot_view/robot/{arm_side}_arm/{link_name}")
            
            # 绘制关节之间的连接
            self._draw_arm_skeleton(arm_chains, link_transforms)
            
            # 记录关节角度为标量
            for i, angle in enumerate(joint_angles_14[:7]):
                rr.log(f"robot_view/robot/joint_angles/left_arm/{self.joint_names_left[i]}", 
                       rr.Scalars(np.degrees(angle)))
            
            for i, angle in enumerate(joint_angles_14[7:14]):
                rr.log(f"robot_view/robot/joint_angles/right_arm/{self.joint_names_right[i]}", 
                       rr.Scalars(np.degrees(angle)))
                
        except Exception as e:
            logging.error(f"Failed to visualize URDF robot: {e}")
            import traceback
            traceback.print_exc()
            # 回退到简化可视化
            self._visualize_robot_simplified(joint_angles_14, frame_index)

    def _rpy_to_rotation_matrix(self, rpy):
        """将RPY转换为3x3旋转矩阵"""
        roll, pitch, yaw = rpy
        
        # 每个轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转(ZYX顺序)
        return R_z @ R_y @ R_x

    def _get_joint_configuration(self, joint_angles_14):
        """创建关节配置字典"""
        if len(joint_angles_14) < 14:
            return {}
        
        joint_config = {}
        
        # 左臂关节
        for i, joint_name in enumerate(self.joint_names_left):
            joint_config[joint_name] = joint_angles_14[i]
        
        # 右臂关节  
        for i, joint_name in enumerate(self.joint_names_right):
            joint_config[joint_name] = joint_angles_14[i + 7]
        
        return joint_config

    def _compute_forward_kinematics(self, joint_config):
        """计算正运动学"""
        link_transforms = {}
        
        # 从基座链接开始
        link_transforms['base_link'] = np.eye(4)
        
        # 定义手臂运动链
        arm_chains = {
            'left': ['zarm_l1_joint', 'zarm_l2_joint', 'zarm_l3_joint', 'zarm_l4_joint', 
                    'zarm_l5_joint', 'zarm_l6_joint', 'zarm_l7_joint'],
            'right': ['zarm_r1_joint', 'zarm_r2_joint', 'zarm_r3_joint', 'zarm_r4_joint', 
                     'zarm_r5_joint', 'zarm_r6_joint', 'zarm_r7_joint']
        }
        
        # 处理每个手臂链
        for arm_side, chain in arm_chains.items():
            for joint_name in chain:
                if joint_name in self.joints:
                    joint_data = self.joints[joint_name]
                    parent_link = joint_data.get('parent')
                    child_link = joint_data.get('child')
                    joint_type = joint_data.get('type')
                    origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                    
                    if parent_link in link_transforms and child_link:
                        # 获取父变换
                        parent_transform = link_transforms[parent_link]
                        
                        # 创建关节变换
                        joint_transform = np.eye(4)
                        
                        # 应用关节原点平移
                        joint_transform[:3, 3] = origin['xyz']
                        
                        # 应用关节原点旋转
                        rpy = origin['rpy']
                        joint_transform[:3, :3] = self._rpy_to_rotation_matrix(rpy)
                        
                        # 如果是旋转关节且有角度数据，应用关节旋转
                        if joint_type == 'revolute' and joint_name in joint_config:
                            joint_angle = joint_config[joint_name]
                            joint_axis = joint_data.get('axis', [0, 0, 1])
                            joint_rotation = self._axis_angle_to_rotation_matrix(joint_axis, joint_angle)
                            joint_transform[:3, :3] = joint_transform[:3, :3] @ joint_rotation
                        
                        # 计算子链接变换
                        link_transforms[child_link] = parent_transform @ joint_transform
        
        return link_transforms

    def _axis_angle_to_rotation_matrix(self, axis, angle):
        """使用Rodrigues公式将轴角表示转换为3x3旋转矩阵"""
        axis = np.array(axis, dtype=float)
        # 归一化轴
        if np.linalg.norm(axis) == 0:
            return np.eye(3)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues旋转公式
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # 轴的叉积矩阵
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rodrigues公式: R = I + sin(θ)K + (1-cos(θ))K²
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        return R

    def _load_link_mesh(self, link_name, transform, entity_path):
        """加载并显示链接网格"""
        if link_name in self.links and TRIMESH_AVAILABLE:
            for visual_data in self.links[link_name]['visual']:
                if 'mesh_file' in visual_data:
                    try:
                        mesh_file = visual_data['mesh_file']
                        if os.path.exists(mesh_file):
                            mesh = trimesh.load_mesh(mesh_file)
                            scale = visual_data.get('scale', [1.0, 1.0, 1.0])
                            if scale != [1.0, 1.0, 1.0]:
                                mesh.vertices *= scale
                            
                            # 应用视觉原点变换
                            origin = visual_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                            visual_transform = np.eye(4)
                            visual_transform[:3, 3] = origin['xyz']
                            visual_transform[:3, :3] = self._rpy_to_rotation_matrix(origin['rpy'])
                            
                            # 应用链接变换和视觉变换
                            full_transform = transform @ visual_transform
                            vertices_homo = np.hstack([mesh.vertices, np.ones((mesh.vertices.shape[0], 1))])
                            transformed_vertices = (full_transform @ vertices_homo.T).T[:, :3]
                            
                            # 在世界坐标中直接记录网格
                            rr.log(f"{entity_path}/mesh",
                                   rr.Mesh3D(
                                       vertex_positions=transformed_vertices.astype(np.float32),
                                       triangle_indices=mesh.faces.astype(np.uint32)
                                   ))
                            break
                    except Exception as e:
                        logging.error(f"Failed to load mesh for {link_name}: {e}")

    def _draw_arm_skeleton(self, arm_chains, link_transforms):
        """绘制手臂骨架连接"""
        for arm_side, chain in arm_chains.items():
            arm_color = [255, 100, 100] if arm_side == 'left' else [100, 100, 255]
            positions = []
            
            # 添加基座连接点
            if arm_side == 'left':
                base_pos = np.array([-0.017499853, 0.29269999999999996, 0.4245])  # 左肩
            else:
                base_pos = np.array([-0.017499853, -0.29269999999999996, 0.4245])  # 右肩
            positions.append(base_pos)
            
            # 添加关节位置
            for link_name in chain:
                if link_name in link_transforms:
                    positions.append(link_transforms[link_name][:3, 3])
            
            if len(positions) > 1:
                positions_array = np.array(positions)
                rr.log(f"robot_view/robot/{arm_side}_arm/skeleton",
                       rr.LineStrips3D([positions_array], 
                                     colors=[arm_color], 
                                     radii=[0.02]))

    def _log_coordinate_frame(self, entity_path, transform, scale=0.1):
        """记录坐标系(TF样式)，带有X(红)，Y(绿)，Z(蓝)轴"""
        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        
        # 创建坐标系轴(X, Y, Z)
        axes = np.array([
            [scale, 0, 0],  # X轴(红)
            [0, scale, 0],  # Y轴(绿)
            [0, 0, scale]   # Z轴(蓝)
        ])
        
        # 根据旋转变换轴
        transformed_axes = rotation_matrix @ axes.T
        
        # X, Y, Z轴的颜色
        colors = [
            [255, 0, 0],    # X轴为红色
            [0, 255, 0],    # Y轴为绿色
            [0, 0, 255]     # Z轴为蓝色
        ]
        
        # 将每个轴记录为箭头
        for i, (axis, color) in enumerate(zip(transformed_axes.T, colors)):
            axis_name = ['x', 'y', 'z'][i]
            rr.log(f"{entity_path}/frame_{axis_name}",
                   rr.Arrows3D(
                       origins=[position],
                       vectors=[axis],
                       colors=[color],
                       radii=[scale * 0.05]
                   ))


class RobotUtils():
    """
        一些机器人工具类

        存放一些相机转换姿态的函数方法，一些旋转转换的方法
    """
    def __init__(self):
        pass

    @staticmethod
    def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
        """将torch tensor从CHW float32格式转换为HWC uint8 numpy数组"""
        assert chw_float32_torch.dtype == torch.float32
        assert chw_float32_torch.ndim == 3
        c, h, w = chw_float32_torch.shape
        assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
        hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
        return hwc_uint8_numpy

    @staticmethod
    def quaternion_to_rotation_matrix(quat):
        """将四元数[x, y, z, w]转换为3x3旋转矩阵"""
        x, y, z, w = quat
        
        # 归一化四元数
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm == 0:
            return np.eye(3)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # 转换为旋转矩阵
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        
        return R

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        """将旋转矩阵转换为四元数(x, y, z, w)"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qx, qy, qz, qw])  # x, y, z, w format for rerun

    @staticmethod
    def get_camera_pose_from_name(camera_name, robot_base_position=np.array([0, 0, 0.4])):
        """
        根据相机名称估算相机姿态
        
        Args:
            camera_name: 相机名称
            robot_base_position: 机器人基座位置
            
        Returns:
            tuple: (position, orientation) 其中orientation是旋转矩阵
        """
        # 默认相机参数
        default_height = 0.8  # 距离地面的高度(米)
        default_distance = 1.5  # 距离机器人的距离(米)
        
        # 解析相机名称确定位置
        camera_name_lower = camera_name.lower()
        
        if 'front' in camera_name_lower:
            # 前置相机 - 从前方看机器人
            position = robot_base_position + np.array([default_distance, 0, default_height - robot_base_position[2]])
            # 相机朝向机器人(绕Z轴旋转180度朝后看)
            orientation = np.array([
                [-1, 0, 0],  # X指向后方(远离机器人)
                [0, -1, 0],  # Y指向左侧
                [0, 0, 1]    # Z指向上方
            ])
        elif 'left' in camera_name_lower and 'wrist' in camera_name_lower:
            # 左手腕相机 - 附着在左臂上
            left_arm_offset = np.array([0.2, 0.3, 0.2])  # 估算的左臂位置
            position = robot_base_position + left_arm_offset
            # 从手腕向前看的相机
            orientation = np.array([
                [0, -1, 0],  # X指向左侧
                [0, 0, -1],  # Y指向下方
                [1, 0, 0]    # Z指向前方
            ])
        elif 'right' in camera_name_lower and 'wrist' in camera_name_lower:
            # 右手腕相机 - 附着在右臂上
            right_arm_offset = np.array([0.2, -0.3, 0.2])  # 估算的右臂位置
            position = robot_base_position + right_arm_offset
            # 从手腕向前看的相机
            orientation = np.array([
                [0, 1, 0],   # X指向右侧
                [0, 0, -1],  # Y指向下方
                [1, 0, 0]    # Z指向前方
            ])
        elif 'back' in camera_name_lower or 'rear' in camera_name_lower:
            # 后置相机 - 从后方看机器人
            position = robot_base_position + np.array([-default_distance, 0, default_height - robot_base_position[2]])
            # 相机朝向机器人
            orientation = np.array([
                [1, 0, 0],   # X指向前方(朝向机器人)
                [0, 1, 0],   # Y指向右侧
                [0, 0, 1]    # Z指向上方
            ])
        elif 'left' in camera_name_lower:
            # 左侧相机
            position = robot_base_position + np.array([0, default_distance, default_height - robot_base_position[2]])
            # 从左侧朝向机器人的相机
            orientation = np.array([
                [0, -1, 0],  # X指向右侧(朝向机器人)
                [1, 0, 0],   # Y指向前方
                [0, 0, 1]    # Z指向上方
            ])
        elif 'right' in camera_name_lower:
            # 右侧相机
            position = robot_base_position + np.array([0, -default_distance, default_height - robot_base_position[2]])
            # 从右侧朝向机器人的相机
            orientation = np.array([
                [0, 1, 0],   # X指向左侧(朝向机器人)
                [-1, 0, 0],  # Y指向后方
                [0, 0, 1]    # Z指向上方
            ])
        elif 'top' in camera_name_lower or 'overhead' in camera_name_lower:
            # 俯视相机
            position = robot_base_position + np.array([0, 0, 2.0])  # 机器人上方高处
            # 向下看的相机
            orientation = np.array([
                [1, 0, 0],   # X指向前方
                [0, 1, 0],   # Y指向右侧
                [0, 0, -1]   # Z指向下方
            ])
        else:
            # 默认前置相机位置
            position = robot_base_position + np.array([default_distance, 0, default_height - robot_base_position[2]])
            orientation = np.array([
                [-1, 0, 0],  # X指向后方
                [0, -1, 0],  # Y指向左侧
                [0, 0, 1]    # Z指向上方
            ])
        
        return position, orientation

    @staticmethod
    def log_camera_pose(entity_path, camera_position, camera_orientation, 
                       camera_name, fov_degrees=60, scale=0.1):
        """
        记录相机姿态，包括位置、方向和视野可视化
        
        Args:
            entity_path: rerun实体路径
            camera_position: 相机3D位置[x, y, z]
            camera_orientation: 四元数[x, y, z, w]或旋转矩阵
            camera_name: 相机名称用于标注
            fov_degrees: 视野角度
            scale: 可视化比例因子
        """
        # 转换为numpy数组
        position = np.array(camera_position)
        
        # 处理不同的方向格式
        if isinstance(camera_orientation, np.ndarray):
            if camera_orientation.shape == (4,):
                # 四元数格式[x, y, z, w]
                quat = camera_orientation
                rotation_matrix = RobotUtils.quaternion_to_rotation_matrix(quat)
            elif camera_orientation.shape == (3, 3):
                # 已经是旋转矩阵
                rotation_matrix = camera_orientation
                quat = RobotUtils.rotation_matrix_to_quaternion(rotation_matrix)
            else:
                raise ValueError(f"Unsupported orientation format: {camera_orientation.shape}")
        else:
            raise ValueError("camera_orientation must be a numpy array")
        
        # 记录相机位置为点
        rr.log(f"{entity_path}/position",
               rr.Points3D([position], 
                         radii=[scale * 0.5], 
                         colors=[[255, 255, 0]]))  # 相机用黄色
        
        # 记录相机的坐标系
        camera_transform = np.eye(4)
        camera_transform[:3, :3] = rotation_matrix
        camera_transform[:3, 3] = position
        RobotUtils.log_coordinate_frame(f"{entity_path}/frame", camera_transform, scale)
        
        # 记录相机视锥(视野可视化)
        RobotUtils.log_camera_frustum(entity_path, position, rotation_matrix, fov_degrees, scale)
        
        # 记录相机名称为文本
        rr.log(f"{entity_path}/label",
               rr.TextDocument(camera_name))

    @staticmethod
    def log_coordinate_frame(entity_path, transform, scale=0.1):
        """
        记录坐标系(TF样式)，带有X(红)，Y(绿)，Z(蓝)轴
        
        Args:
            entity_path: rerun实体路径
            transform: 4x4变换矩阵
            scale: 箭头的比例因子
        """
        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        
        # 创建坐标系轴(X, Y, Z)
        axes = np.array([
            [scale, 0, 0],  # X轴(红)
            [0, scale, 0],  # Y轴(绿)
            [0, 0, scale]   # Z轴(蓝)
        ])
        
        # 根据旋转变换轴
        transformed_axes = rotation_matrix @ axes.T
        
        # X, Y, Z轴的颜色
        colors = [
            [255, 0, 0],    # X轴为红色
            [0, 255, 0],    # Y轴为绿色
            [0, 0, 255]     # Z轴为蓝色
        ]
        
        # 将每个轴记录为箭头
        for i, (axis, color) in enumerate(zip(transformed_axes.T, colors)):
            axis_name = ['x', 'y', 'z'][i]
            rr.log(f"{entity_path}/frame_{axis_name}",
                   rr.Arrows3D(
                       origins=[position],
                       vectors=[axis],
                       colors=[color],
                       radii=[scale * 0.05]
                   ))

    @staticmethod
    def log_camera_frustum(entity_path, position, rotation_matrix, 
                          fov_degrees, scale, depth=0.3):
        """
        记录相机视锥以可视化视野
        
        Args:
            entity_path: rerun实体路径
            position: 相机位置
            rotation_matrix: 相机方向
            fov_degrees: 视野角度
            scale: 比例因子
            depth: 视锥的深度
        """
        # 计算视锥角点
        fov_rad = np.radians(fov_degrees)
        half_fov = fov_rad / 2
        
        # 相机坐标系中的视锥角点(相机沿+X轴看)
        frustum_depth = depth * scale * 3
        frustum_width = 2 * frustum_depth * np.tan(half_fov)
        frustum_height = frustum_width  # 假设正方形宽高比
        
        # 本地视锥角点(相机坐标系，沿+X轴看)
        local_corners = np.array([
            [0, 0, 0],  # 相机中心
            [frustum_depth, -frustum_width/2, -frustum_height/2],  # 远-底-左
            [frustum_depth, frustum_width/2, -frustum_height/2],   # 远-底-右
            [frustum_depth, frustum_width/2, frustum_height/2],    # 远-顶-右
            [frustum_depth, -frustum_width/2, frustum_height/2],   # 远-顶-左
        ])
        
        # 变换到世界坐标
        world_corners = []
        for corner in local_corners:
            world_corner = position + rotation_matrix @ corner
            world_corners.append(world_corner)
        
        world_corners = np.array(world_corners)
        
        # 定义视锥边缘(从相机中心到角点和角点之间的线)
        frustum_lines = [
            [world_corners[0], world_corners[1]],  # 中心到底-左
            [world_corners[0], world_corners[2]],  # 中心到底-右
            [world_corners[0], world_corners[3]],  # 中心到顶-右
            [world_corners[0], world_corners[4]],  # 中心到顶-左
            [world_corners[1], world_corners[2]],  # 底边
            [world_corners[2], world_corners[3]],  # 右边
            [world_corners[3], world_corners[4]],  # 顶边
            [world_corners[4], world_corners[1]],  # 左边
        ]
        
        # 记录视锥线条
        rr.log(f"{entity_path}/frustum",
               rr.LineStrips3D(frustum_lines, 
                             colors=[[255, 255, 0]], 
                             radii=[scale * 0.02]))

    @staticmethod
    def visualize_image_in_3d(entity_path, camera_pos, camera_rot, image_tensor, scale=0.3):
        """
        将相机图像作为平面在3D空间中可视化
        
        Args:
            entity_path: rerun实体路径
            camera_pos: 相机位置
            camera_rot: 相机旋转矩阵
            image_tensor: 批次中的图像张量
            scale: 图像平面的比例因子
        """
        try:
            # 将图像转换为numpy
            image_np = RobotUtils.to_hwc_uint8_numpy(image_tensor)
            h, w = image_np.shape[:2]
            
            # 在相机坐标系中创建图像平面
            # 假设图像平面在相机前方某个距离处
            plane_distance = scale * 2
            plane_width = scale
            plane_height = scale * (h / w)  # 保持宽高比
            
            # 在相机坐标中定义图像平面的角点
            # 相机沿+X轴看，所以图像平面在+plane_distance处
            local_corners = np.array([
                [plane_distance, -plane_width/2, -plane_height/2],  # 底-左
                [plane_distance, plane_width/2, -plane_height/2],   # 底-右
                [plane_distance, plane_width/2, plane_height/2],    # 顶-右
                [plane_distance, -plane_width/2, plane_height/2],   # 顶-左
            ])
            
            # 变换角点到世界坐标
            world_corners = []
            for corner in local_corners:
                world_corner = camera_pos + camera_rot @ corner
                world_corners.append(world_corner)
            
            world_corners = np.array(world_corners)
            
            # 为平面创建三角形索引(两个三角形)
            triangle_indices = np.array([
                [0, 1, 2],  # 第一个三角形
                [0, 2, 3]   # 第二个三角形
            ])
            
            # 为纹理映射创建UV坐标
            vertex_texcoords = np.array([
                [0, 1],  # 底-左
                [1, 1],  # 底-右
                [1, 0],  # 顶-右
                [0, 0]   # 顶-左
            ])
            
            # 用纹理记录图像平面网格
            rr.log(f"{entity_path}/image_plane",
                   rr.Mesh3D(
                       vertex_positions=world_corners.astype(np.float32),
                       triangle_indices=triangle_indices.astype(np.uint32),
                       vertex_texcoords=vertex_texcoords.astype(np.float32),
                       albedo_texture=image_np
                   ))
                   
        except Exception as e:
            logging.warning(f"Failed to create 3D image plane: {e}")

    @staticmethod
    def configure_robot_view():
        """为最佳机器人可视化配置rerun查看器"""
        try:
            # 设置主要的3D视图聚焦于机器人
            rr.log("robot_view", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
            
            # 为机器人观察配置一个好的视角
            # 将相机定位到从好角度看机器人
            rr.log("robot_view/camera", 
                   rr.ViewCoordinates(
                       xyz=rr.ViewCoordinates.RUB  # 右，上，后
                   ))
            
            # 设置合理的视距和角度
            # 这将视图定位为从前-右-上看机器人
            eye_pos = [2.0, 1.5, 1.5]  # 相机位置
            target_pos = [0.0, 0.0, 0.4]  # 看向机器人基座
            up_vector = [0.0, 0.0, 1.0]  # Z向上
            
            # 注意：Rerun 0.24.1没有通过API直接控制相机的功能
            # 用户需要在GUI中手动调整视图以获得最佳机器人观察效果
            logging.info("Robot view configured. Adjust camera position in Rerun viewer for optimal robot observation.")
            logging.info(f"Suggested view: Eye={eye_pos}, Target={target_pos}, Up={up_vector}")
            
        except Exception as e:
            logging.warning(f"Failed to configure robot view: {e}")


class DatasetVisualizer():
    """
    数据集可视化主类，整合所有功能
    """
    def __init__(self, dataset_path: str, dataset_mode: str = "lerobot", 
                 episode_index: int = 0, urdf_path: str = None,
                 robot_mode: str = "fixed_dual", camera_mode: str = "camera_mode_a"):
        
        # 初始化数据集
        self.dataset_wrapper = DatasetWrapped(dataset_path, dataset_mode, episode_index)
        
        # 初始化机器人可视化器
        self.robot_visualizer = RobotVisualizer(urdf_path, robot_mode, camera_mode)
        
        # 初始化工具类
        self.robot_utils = RobotUtils()
        
        # 可视化参数
        self.batch_size = 32
        self.num_workers = 0
        
        logging.info(f"DatasetVisualizer initialized for {dataset_path}, episode {episode_index}")

    def visualize_camera_poses(self, camera_keys, batch, batch_index):
        """在3D空间中可视化相机姿态"""
        # 机器人基座位置(可以根据实际机器人状态调整)
        robot_base_pos = np.array([0, 0, 0.4])
        
        # 使用头部关节角度配置
        head_joint_angles = self.robot_visualizer.head_joint_angles
        
        for camera_key in camera_keys:
            try:
                # 从URDF或名称估算获取相机姿态
                camera_pos, camera_rot = self._get_camera_pose_from_urdf(camera_key, robot_base_pos, head_joint_angles)
                
                # 为可视化创建清晰的相机名称
                camera_name = camera_key.replace('observation.images.', '').replace('.', '_')
                
                # 记录带有视锥的相机姿态
                RobotUtils.log_camera_pose(
                    f"robot_view/cameras/{camera_name}",
                    camera_pos,
                    camera_rot,
                    camera_name,
                    fov_degrees=60,
                    scale=0.2
                )
                
                # 添加图像平面可视化(将图像投影到3D空间)
                if camera_key in batch:
                    RobotUtils.visualize_image_in_3d(f"robot_view/cameras/{camera_name}", 
                                                   camera_pos, camera_rot, 
                                                   batch[camera_key][batch_index], scale=0.3)
                                             
            except Exception as e:
                logging.warning(f"Failed to visualize camera pose for {camera_key}: {e}")

    def _get_camera_pose_from_urdf(self, camera_name, robot_base_position, head_joint_angles=None):
        """从URDF运动链提取相机姿态"""
        if not self.robot_visualizer or not self.robot_visualizer.urdf_root:
            return RobotUtils.get_camera_pose_from_name(camera_name, robot_base_position)
        
        # 定义从相机键到URDF链接名称的映射
        camera_link_mapping = {
            'observation.images.front': 'camera',  # 头部相机
            'front': 'camera',
            'camera': 'camera',
            'head': 'camera',
            'observation.images.torso': 'torso-camera',  # 躯干相机
            'torso': 'torso-camera',
            'observation.images.waist': 'waist_camera',  # 腰部相机
            'waist': 'waist_camera',
        }
        
        # 找到对应的URDF链接
        urdf_link_name = None
        camera_name_clean = camera_name.lower().replace('observation.images.', '').replace('.', '_')
        
        for key, link_name in camera_link_mapping.items():
            if key in camera_name.lower() or camera_name_clean == key:
                urdf_link_name = link_name
                break
        
        if urdf_link_name and urdf_link_name in self.robot_visualizer.links:
            # 计算正运动学获取相机姿态
            camera_transform = self._compute_camera_forward_kinematics(urdf_link_name, head_joint_angles)
            if camera_transform is not None:
                position = camera_transform[:3, 3]
                orientation = camera_transform[:3, :3]
                return position, orientation
        
        # 如果URDF解析失败，回退到基于名称的估算
        return RobotUtils.get_camera_pose_from_name(camera_name, robot_base_position)

    def _compute_camera_forward_kinematics(self, camera_link_name, head_joint_angles=None):
        """基于URDF结构计算相机链接的正运动学"""
        try:
            # 从base_link变换开始
            current_transform = np.eye(4)
            
            # 如果未提供头部关节角度，设置默认值
            if head_joint_angles is None:
                head_joint_angles = self.robot_visualizer.head_joint_angles
            
            if camera_link_name == 'camera':
                # 头部相机: base_link -> zhead_1_joint -> zhead_1_link -> zhead_2_joint -> zhead_2_link -> camera_joint -> camera
                
                # 1. zhead_1_joint变换 (base_link到zhead_1_link)
                if 'zhead_1_joint' in self.robot_visualizer.joints:
                    joint_data = self.robot_visualizer.joints['zhead_1_joint']
                    origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                    
                    joint_transform = np.eye(4)
                    joint_transform[:3, 3] = origin['xyz']
                    joint_transform[:3, :3] = self.robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])
                    
                    # 使用实际头部关节角度添加Z轴旋转
                    joint_angle = head_joint_angles[0] if len(head_joint_angles) > 0 else 0.0
                    if joint_data.get('type') == 'revolute':
                        joint_axis = joint_data.get('axis', [0, 0, 1])  # Z轴
                        joint_rotation = self.robot_visualizer._axis_angle_to_rotation_matrix(joint_axis, joint_angle)
                        joint_transform[:3, :3] = joint_transform[:3, :3] @ joint_rotation
                    
                    current_transform = current_transform @ joint_transform
                
                # 2. zhead_2_joint变换 (zhead_1_link到zhead_2_link)  
                if 'zhead_2_joint' in self.robot_visualizer.joints:
                    joint_data = self.robot_visualizer.joints['zhead_2_joint']
                    origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                    
                    joint_transform = np.eye(4)
                    joint_transform[:3, 3] = origin['xyz']
                    joint_transform[:3, :3] = self.robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])
                    
                    # 使用实际头部关节角度添加Y轴旋转
                    joint_angle = head_joint_angles[1] if len(head_joint_angles) > 1 else np.radians(30)
                    if joint_data.get('type') == 'revolute':
                        joint_axis = joint_data.get('axis', [0, 1, 0])  # Y轴
                        joint_rotation = self.robot_visualizer._axis_angle_to_rotation_matrix(joint_axis, joint_angle)
                        joint_transform[:3, :3] = joint_transform[:3, :3] @ joint_rotation
                    
                    current_transform = current_transform @ joint_transform
                
                # 3. camera固定关节变换 (zhead_2_link到camera)
                if 'camera' in self.robot_visualizer.joints:
                    joint_data = self.robot_visualizer.joints['camera']
                    origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                    
                    joint_transform = np.eye(4)
                    joint_transform[:3, 3] = origin['xyz']
                    joint_transform[:3, :3] = self.robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])
                    
                    current_transform = current_transform @ joint_transform
            
            elif camera_link_name == 'torso-camera':
                # 躯干相机: base_link -> torso-camera (直接固定关节)
                if 'torso-camera' in self.robot_visualizer.joints:
                    joint_data = self.robot_visualizer.joints['torso-camera']
                    origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                    
                    joint_transform = np.eye(4)
                    joint_transform[:3, 3] = origin['xyz']
                    joint_transform[:3, :3] = self.robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])
                    
                    current_transform = current_transform @ joint_transform
            
            elif camera_link_name == 'waist_camera':
                # 腰部相机: base_link -> waist_camera (直接固定关节)
                if 'waist_camera' in self.robot_visualizer.joints:
                    joint_data = self.robot_visualizer.joints['waist_camera']
                    origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                    
                    joint_transform = np.eye(4)
                    joint_transform[:3, 3] = origin['xyz']
                    joint_transform[:3, :3] = self.robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])
                    
                    current_transform = current_transform @ joint_transform
            
            else:
                logging.warning(f"Unknown camera link: {camera_link_name}")
                return None
            
            return current_transform
                
        except Exception as e:
            logging.warning(f"Failed to compute camera forward kinematics for {camera_link_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def visualize_dataset(self, mode: str = "local", web_port: int = 9090, ws_port: int = 9087, 
                         save: bool = False, output_dir: Path = None):
        """
        可视化数据集的主要方法
        
        Args:
            mode: 可视化模式 ("local", "distant")
            web_port: web端口 (distant模式使用)
            ws_port: websocket端口 (distant模式使用)
            save: 是否保存为.rrd文件
            output_dir: 输出目录 (save=True时使用)
        """
        if save:
            assert output_dir is not None, (
                "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
            )

        dataset = self.dataset_wrapper.get_dataset_data()
        repo_id = dataset.repo_id if hasattr(dataset, 'repo_id') else "unknown_dataset"
        episode_index = self.dataset_wrapper.episode_index

        logging.info("Loading dataloader")
        dataloader = self.dataset_wrapper.create_episode_dataloader(self.batch_size, self.num_workers)

        logging.info("Starting Rerun")

        if mode not in ["local", "distant"]:
            raise ValueError(f"Invalid mode: {mode}")

        spawn_local_viewer = mode == "local" and not save
        rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

        # 手动调用python垃圾收集器，避免在dataloader迭代时卡住
        gc.collect()
        
        # 配置机器人中心视图设置
        RobotUtils.configure_robot_view()

        if mode == "distant":
            rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

        logging.info("Logging to Rerun")

        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            # 遍历批次
            for i in range(len(batch["index"])):
                rr.set_time("frame_index", sequence=batch["frame_index"][i].item())
                rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

                # 显示每个相机图像
                camera_keys = self.dataset_wrapper.get_dataset_image_data()
                for key in camera_keys:
                    rr.log(key, rr.Image(RobotUtils.to_hwc_uint8_numpy(batch[key][i])))
                
                # 在3D空间中可视化相机姿态
                self.visualize_camera_poses(camera_keys, batch, i)

                # 显示动作空间的每个维度(例如执行器命令)
                if "action" in batch:
                    for dim_idx, val in enumerate(batch["action"][i]):
                        rr.log(f"action/{dim_idx}", rr.Scalars(val.item()))

                # 显示观察状态空间的每个维度(例如关节空间中的agent位置)
                if "observation.state" in batch:
                    for dim_idx, val in enumerate(batch["observation.state"][i]):
                        rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))
                    
                    # 提取并可视化机器人手臂关节
                    state_data = batch["observation.state"][i].numpy()
                    arm_joints = self.dataset_wrapper.get_dataset_robot_data(state_data)
                    
                    if arm_joints is not None and len(arm_joints) == 14:
                        try:
                            self.robot_visualizer.set_robot_joint_position(arm_joints, batch["frame_index"][i].item())
                        except Exception as e:
                            logging.warning(f"Failed to visualize robot at frame {batch['frame_index'][i].item()}: {e}")
                    else:
                        # 记录状态结构以供调试
                        if i == 0:  # 只在每批次记录一次以避免垃圾信息
                            logging.info(f"State shape: {state_data.shape}, cannot extract 14 arm joints for robot visualization")

                if "next.done" in batch:
                    rr.log("next.done", rr.Scalars(batch["next.done"][i].item()))

                if "next.reward" in batch:
                    rr.log("next.reward", rr.Scalars(batch["next.reward"][i].item()))

                if "next.success" in batch:
                    rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

        if mode == "local" and save:
            # 本地保存.rrd
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            repo_id_str = repo_id.replace("/", "_")
            rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
            rr.save(rrd_path)
            logging.info(f"Saved visualization to {rrd_path}")
            return rrd_path

        elif mode == "distant":
            # 阻止进程退出，因为它正在为websocket连接提供服务
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Ctrl-C received. Exiting.")

        return None


def main():
    """主函数，解析命令行参数并运行可视化"""
    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset with 3D robot and camera visualization")

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode to visualize (default: 0).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save` is set.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "distant"],
        help="Mode of viewing between 'local' or 'distant'.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save a .rrd file in the directory provided by `--output-dir`.",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Tolerance in seconds used to ensure data timestamps respect the dataset fps value.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help="Path to URDF file for accurate robot visualization with meshes.",
    )
    parser.add_argument(
        "--robot-mode",
        type=str,
        default="fixed_dual",
        choices=["fixed_single", "fixed_dual", "mobile_dual"],
        help="Robot mode for visualization.",
    )
    parser.add_argument(
        "--camera-mode",
        type=str,
        default="camera_mode_a",
        choices=["camera_mode_a", "camera_mode_b", "camera_mode_c"],
        help="Camera mode for visualization.",
    )

    args = parser.parse_args()

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    try:
        # 创建数据集可视化器
        visualizer = DatasetVisualizer(
            dataset_path=args.repo_id,
            dataset_mode="lerobot",
            episode_index=args.episode_index,
            urdf_path=args.urdf_path,
            robot_mode=args.robot_mode,
            camera_mode=args.camera_mode
        )

        # 运行可视化
        result_path = visualizer.visualize_dataset(
            mode=args.mode,
            web_port=args.web_port,
            ws_port=args.ws_port,
            save=args.save,
            output_dir=args.output_dir
        )

        if result_path:
            print(f"Visualization saved to: {result_path}")

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


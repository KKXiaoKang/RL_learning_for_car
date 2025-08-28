"""
Robot System Module

This module provides a comprehensive framework for robotics data processing and visualization.
It includes robot abstraction, camera systems, joint management, and dataset wrappers.

Main Components:
- Robotic: Main robot class with camera, joint, and model management
- Camera/DepthCamera: Camera system abstractions
- Joint: Joint management system
- DatasetWrapper: Virtual base class for dataset handling
"""

from .robotic import Robotic, create_dual_arm_robot, create_kuavo_robot
from .camera import Camera, DepthCamera, CameraIntrinsics, CameraPose
from .joint import Joint, JointCollection, JointLimits, JointAxis
from .dataset_wrapper import DatasetWrapper, LeRobotDatasetWrapper
from .models import ModelMode, TorsoConfig

__all__ = [
    'Robotic',
    'create_dual_arm_robot',
    'create_kuavo_robot',
    'Camera', 
    'DepthCamera',
    'CameraIntrinsics',
    'CameraPose',
    'Joint',
    'JointCollection',
    'JointLimits',
    'JointAxis',
    'DatasetWrapper',
    'LeRobotDatasetWrapper',
    'ModelMode',
    'TorsoConfig'
]

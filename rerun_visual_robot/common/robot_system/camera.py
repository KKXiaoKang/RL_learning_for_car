"""
Camera System Classes

This module defines camera abstractions for robot vision systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length in x
    fy: float  # Focal length in y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int  # Image width
    height: int  # Image height
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get 3x3 camera projection matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])


@dataclass 
class CameraPose:
    """Camera pose in 3D space"""
    position: np.ndarray  # 3D position [x, y, z]
    orientation: np.ndarray  # 3x3 rotation matrix or quaternion [x, y, z, w]
    
    def get_transform_matrix(self) -> np.ndarray:
        """
        Get 4x4 transformation matrix
        
        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        transform[:3, 3] = self.position
        
        if self.orientation.shape == (4,):
            # Convert quaternion to rotation matrix
            transform[:3, :3] = self._quaternion_to_rotation_matrix(self.orientation)
        elif self.orientation.shape == (3, 3):
            # Already a rotation matrix
            transform[:3, :3] = self.orientation
        else:
            raise ValueError(f"Invalid orientation shape: {self.orientation.shape}")
        
        return transform
    
    def get_rotation_matrix(self) -> np.ndarray:
        """
        Get rotation matrix from orientation
        
        Returns:
            3x3 rotation matrix
        """
        if self.orientation.shape == (4,):
            # Convert quaternion to rotation matrix
            return self._quaternion_to_rotation_matrix(self.orientation)
        elif self.orientation.shape == (3, 3):
            # Already a rotation matrix
            return self.orientation
        else:
            raise ValueError(f"Invalid orientation shape: {self.orientation.shape}")
    
    def _quaternion_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [x, y, z, w] to rotation matrix"""
        x, y, z, w = quat
        
        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm == 0:
            return np.eye(3)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Convert to rotation matrix
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])


class Camera(ABC):
    """Abstract base class for camera systems"""
    
    def __init__(
        self,
        name: str,
        intrinsics: Optional[CameraIntrinsics] = None,
        pose: Optional[CameraPose] = None,
        **kwargs
    ):
        self.name = name
        self.intrinsics = intrinsics
        self.pose = pose
        self.properties = kwargs
    
    @abstractmethod
    def capture_image(self) -> Union[np.ndarray, torch.Tensor]:
        """Capture image from camera"""
        pass
    
    @abstractmethod
    def get_image_from_data(self, data: Dict[str, Any]) -> Union[np.ndarray, torch.Tensor]:
        """Extract image from dataset data"""
        pass
    
    def update_pose(self, pose: CameraPose):
        """Update camera pose"""
        self.pose = pose
    
    def update_intrinsics(self, intrinsics: CameraIntrinsics):
        """Update camera intrinsics"""
        self.intrinsics = intrinsics
    
    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates
        
        Args:
            points_3d: Nx3 array of 3D points
            
        Returns:
            Nx2 array of 2D image coordinates
        """
        if self.intrinsics is None or self.pose is None:
            raise ValueError("Camera intrinsics and pose must be set for projection")
        
        # Transform points to camera coordinate system
        transform = self.pose.get_transform_matrix()
        camera_transform = np.linalg.inv(transform)
        
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        
        # Transform to camera coordinates
        points_camera = (camera_transform @ points_homo.T).T[:, :3]
        
        # Project to image plane
        projection_matrix = self.intrinsics.get_projection_matrix()
        points_2d_homo = (projection_matrix @ points_camera.T).T
        
        # Convert from homogeneous to 2D coordinates
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        return points_2d
    
    def get_field_of_view(self) -> Tuple[float, float]:
        """
        Get horizontal and vertical field of view in radians
        
        Returns:
            (horizontal_fov, vertical_fov) in radians
        """
        if self.intrinsics is None:
            return (np.radians(60), np.radians(45))  # Default values
        
        h_fov = 2 * np.arctan(self.intrinsics.width / (2 * self.intrinsics.fx))
        v_fov = 2 * np.arctan(self.intrinsics.height / (2 * self.intrinsics.fy))
        
        return h_fov, v_fov


class RGBCamera(Camera):
    """RGB camera implementation"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.image_format = "RGB"
    
    def capture_image(self) -> Union[np.ndarray, torch.Tensor]:
        """Capture RGB image from camera"""
        # Placeholder implementation
        raise NotImplementedError("Real-time capture not implemented")
    
    def get_image_from_data(self, data: Dict[str, Any]) -> Union[np.ndarray, torch.Tensor]:
        """Extract RGB image from dataset data"""
        # Look for image data with this camera's name
        possible_keys = [
            self.name,
            f"observation.images.{self.name}",
            f"images.{self.name}",
            f"camera.{self.name}"
        ]
        
        for key in possible_keys:
            if key in data:
                return data[key]
        
        raise KeyError(f"No image data found for camera '{self.name}' in keys: {list(data.keys())}")


class DepthCamera(Camera):
    """Depth camera implementation"""
    
    def __init__(self, name: str, depth_scale: float = 1000.0, **kwargs):
        super().__init__(name, **kwargs)
        self.depth_scale = depth_scale  # Scale factor for depth values
        self.image_format = "DEPTH"
    
    def capture_image(self) -> Union[np.ndarray, torch.Tensor]:
        """Capture depth image from camera"""
        # Placeholder implementation
        raise NotImplementedError("Real-time capture not implemented")
    
    def get_image_from_data(self, data: Dict[str, Any]) -> Union[np.ndarray, torch.Tensor]:
        """Extract depth image from dataset data"""
        # Look for depth data with this camera's name
        possible_keys = [
            f"{self.name}_depth",
            f"depth_{self.name}",
            f"observation.images.{self.name}_depth",
            f"depth.{self.name}"
        ]
        
        for key in possible_keys:
            if key in data:
                return data[key]
        
        raise KeyError(f"No depth data found for camera '{self.name}' in keys: {list(data.keys())}")
    
    def depth_to_pointcloud(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert depth image to 3D point cloud
        
        Args:
            depth_image: 2D depth image
            
        Returns:
            Nx3 array of 3D points
        """
        if self.intrinsics is None:
            raise ValueError("Camera intrinsics required for point cloud generation")
        
        height, width = depth_image.shape
        
        # Create pixel coordinate arrays
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth to real-world scale
        z = depth_image / self.depth_scale
        
        # Convert to 3D points in camera coordinate system
        x = (u - self.intrinsics.cx) * z / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * z / self.intrinsics.fy
        
        # Stack and reshape to Nx3
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Filter out invalid depth values
        valid_mask = points[:, 2] > 0
        return points[valid_mask]


class CameraSystem:
    """Manager for multiple cameras"""
    
    def __init__(self):
        self.cameras: Dict[str, Camera] = {}
    
    def add_camera(self, camera: Camera):
        """Add a camera to the system"""
        self.cameras[camera.name] = camera
    
    def get_camera(self, name: str) -> Optional[Camera]:
        """Get camera by name"""
        return self.cameras.get(name)
    
    def get_all_cameras(self) -> Dict[str, Camera]:
        """Get all cameras"""
        return self.cameras.copy()
    
    def get_rgb_cameras(self) -> Dict[str, RGBCamera]:
        """Get all RGB cameras"""
        return {name: cam for name, cam in self.cameras.items() 
                if isinstance(cam, RGBCamera)}
    
    def get_depth_cameras(self) -> Dict[str, DepthCamera]:
        """Get all depth cameras"""
        return {name: cam for name, cam in self.cameras.items() 
                if isinstance(cam, DepthCamera)}
    
    def extract_all_images(self, data: Dict[str, Any]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Extract images from all cameras in the system"""
        images = {}
        for name, camera in self.cameras.items():
            try:
                images[name] = camera.get_image_from_data(data)
            except KeyError:
                # Camera data not available in this dataset
                continue
        return images

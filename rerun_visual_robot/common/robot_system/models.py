"""
Robot Model Configuration

This module defines robot model modes and configuration classes.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


class ModelMode(Enum):
    """Robot model base modes"""
    FIXED_BASE = "fixed_base"
    FLOATING_BASE = "floating_base"


@dataclass
class TorsoConfig:
    """Configuration for robot torso and base link positioning"""
    
    # Base link position in world coordinates
    base_link_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Torso position relative to base link
    torso_position: Tuple[float, float, float] = (0.0, 0.0, 0.4)
    
    # Torso orientation (roll, pitch, yaw in radians)
    torso_orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Model mode
    model_mode: ModelMode = ModelMode.FIXED_BASE
    
    def get_base_link_transform(self) -> np.ndarray:
        """
        Get 4x4 transformation matrix for base link
        
        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        transform[:3, 3] = self.base_link_position
        
        # Apply torso orientation if needed
        if any(self.torso_orientation):
            roll, pitch, yaw = self.torso_orientation
            
            # Rotation matrices for each axis
            cos_r, sin_r = np.cos(roll), np.sin(roll)
            cos_p, sin_p = np.cos(pitch), np.sin(pitch)
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            
            # Combined rotation matrix (ZYX order)
            R_x = np.array([
                [1, 0, 0],
                [0, cos_r, -sin_r],
                [0, sin_r, cos_r]
            ])
            
            R_y = np.array([
                [cos_p, 0, sin_p],
                [0, 1, 0],
                [-sin_p, 0, cos_p]
            ])
            
            R_z = np.array([
                [cos_y, -sin_y, 0],
                [sin_y, cos_y, 0],
                [0, 0, 1]
            ])
            
            transform[:3, :3] = R_z @ R_y @ R_x
        
        return transform
    
    def get_torso_transform(self) -> np.ndarray:
        """
        Get 4x4 transformation matrix for torso relative to base link
        
        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        transform[:3, 3] = self.torso_position
        return transform
    
    def get_world_torso_transform(self) -> np.ndarray:
        """
        Get 4x4 transformation matrix for torso in world coordinates
        
        Returns:
            4x4 transformation matrix
        """
        base_transform = self.get_base_link_transform()
        torso_transform = self.get_torso_transform()
        return base_transform @ torso_transform

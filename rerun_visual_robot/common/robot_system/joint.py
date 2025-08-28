"""
Joint System Classes

This module defines joint management for robot kinematics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from enum import Enum


class JointType(Enum):
    """Types of robot joints"""
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CONTINUOUS = "continuous"
    FIXED = "fixed"


@dataclass
class JointLimits:
    """Joint angle/position limits"""
    lower: float
    upper: float
    velocity: Optional[float] = None
    effort: Optional[float] = None


@dataclass
class JointAxis:
    """Joint rotation/translation axis"""
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array and normalize"""
        axis = np.array([self.x, self.y, self.z])
        norm = np.linalg.norm(axis)
        if norm == 0:
            return np.array([0, 0, 1])  # Default to Z-axis
        return axis / norm


class Joint:
    """Individual robot joint representation"""
    
    def __init__(
        self,
        name: str,
        joint_type: JointType = JointType.REVOLUTE,
        axis: Optional[JointAxis] = None,
        limits: Optional[JointLimits] = None,
        parent_link: Optional[str] = None,
        child_link: Optional[str] = None,
        origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.name = name
        self.joint_type = joint_type
        self.axis = axis or JointAxis()
        self.limits = limits
        self.parent_link = parent_link
        self.child_link = child_link
        self.origin_xyz = origin_xyz
        self.origin_rpy = origin_rpy
        
        # Current state
        self.position = 0.0
        self.velocity = 0.0
        self.effort = 0.0
    
    def update_state(self, position: float, velocity: float = 0.0, effort: float = 0.0):
        """Update joint state"""
        self.position = position
        self.velocity = velocity
        self.effort = effort
    
    def is_within_limits(self, position: float) -> bool:
        """Check if position is within joint limits"""
        if self.limits is None:
            return True
        return self.limits.lower <= position <= self.limits.upper
    
    def clamp_to_limits(self, position: float) -> float:
        """Clamp position to joint limits"""
        if self.limits is None:
            return position
        return np.clip(position, self.limits.lower, self.limits.upper)
    
    def get_origin_transform(self) -> np.ndarray:
        """Get 4x4 transformation matrix for joint origin"""
        transform = np.eye(4)
        transform[:3, 3] = self.origin_xyz
        
        # Apply rotation from RPY
        roll, pitch, yaw = self.origin_rpy
        
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        # Rotation matrices
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
    
    def get_joint_transform(self, position: Optional[float] = None) -> np.ndarray:
        """
        Get joint transformation for given position
        
        Args:
            position: Joint position (uses current position if None)
            
        Returns:
            4x4 transformation matrix
        """
        if position is None:
            position = self.position
        
        transform = np.eye(4)
        
        if self.joint_type == JointType.REVOLUTE:
            # Rotation around joint axis
            axis = self.axis.to_numpy()
            angle = position
            
            # Rodrigues' rotation formula
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            # Cross product matrix for axis
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            
            # R = I + sin(θ)K + (1-cos(θ))K²
            rotation = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
            transform[:3, :3] = rotation
            
        elif self.joint_type == JointType.PRISMATIC:
            # Translation along joint axis
            axis = self.axis.to_numpy()
            transform[:3, 3] = axis * position
        
        return transform


class JointCollection:
    """Collection of robot joints organized by chains/groups"""
    
    def __init__(self):
        self.joints: Dict[str, Joint] = {}
        self.joint_chains: Dict[str, List[str]] = {}
        self.joint_groups: Dict[str, List[str]] = {}
    
    def add_joint(self, joint: Joint):
        """Add a joint to the collection"""
        self.joints[joint.name] = joint
    
    def get_joint(self, name: str) -> Optional[Joint]:
        """Get joint by name"""
        return self.joints.get(name)
    
    def add_joint_chain(self, chain_name: str, joint_names: List[str]):
        """Define a kinematic chain of joints"""
        # Validate that all joints exist
        for joint_name in joint_names:
            if joint_name not in self.joints:
                raise ValueError(f"Joint '{joint_name}' not found in collection")
        self.joint_chains[chain_name] = joint_names
    
    def add_joint_group(self, group_name: str, joint_names: List[str]):
        """Define a group of joints (not necessarily in kinematic order)"""
        for joint_name in joint_names:
            if joint_name not in self.joints:
                raise ValueError(f"Joint '{joint_name}' not found in collection")
        self.joint_groups[group_name] = joint_names
    
    def get_chain_positions(self, chain_name: str) -> np.ndarray:
        """Get positions for all joints in a chain"""
        if chain_name not in self.joint_chains:
            raise ValueError(f"Chain '{chain_name}' not found")
        
        positions = []
        for joint_name in self.joint_chains[chain_name]:
            positions.append(self.joints[joint_name].position)
        return np.array(positions)
    
    def set_chain_positions(self, chain_name: str, positions: Union[List[float], np.ndarray]):
        """Set positions for all joints in a chain"""
        if chain_name not in self.joint_chains:
            raise ValueError(f"Chain '{chain_name}' not found")
        
        joint_names = self.joint_chains[chain_name]
        if len(positions) != len(joint_names):
            raise ValueError(f"Position array length {len(positions)} doesn't match chain length {len(joint_names)}")
        
        for joint_name, position in zip(joint_names, positions):
            self.joints[joint_name].update_state(position)
    
    def get_group_positions(self, group_name: str) -> np.ndarray:
        """Get positions for all joints in a group"""
        if group_name not in self.joint_groups:
            raise ValueError(f"Group '{group_name}' not found")
        
        positions = []
        for joint_name in self.joint_groups[group_name]:
            positions.append(self.joints[joint_name].position)
        return np.array(positions)
    
    def set_group_positions(self, group_name: str, positions: Union[List[float], np.ndarray]):
        """Set positions for all joints in a group"""
        if group_name not in self.joint_groups:
            raise ValueError(f"Group '{group_name}' not found")
        
        joint_names = self.joint_groups[group_name]
        if len(positions) != len(joint_names):
            raise ValueError(f"Position array length {len(positions)} doesn't match group length {len(joint_names)}")
        
        for joint_name, position in zip(joint_names, positions):
            self.joints[joint_name].update_state(position)
    
    def compute_forward_kinematics(self, chain_name: str, base_transform: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute forward kinematics for a joint chain
        
        Args:
            chain_name: Name of the joint chain
            base_transform: Base transformation matrix (identity if None)
            
        Returns:
            Dictionary mapping joint names to 4x4 transformation matrices
        """
        if chain_name not in self.joint_chains:
            raise ValueError(f"Chain '{chain_name}' not found")
        
        if base_transform is None:
            base_transform = np.eye(4)
        
        transforms = {}
        current_transform = base_transform.copy()
        
        for joint_name in self.joint_chains[chain_name]:
            joint = self.joints[joint_name]
            
            # Apply joint origin transform
            origin_transform = joint.get_origin_transform()
            current_transform = current_transform @ origin_transform
            
            # Apply joint motion transform
            joint_transform = joint.get_joint_transform()
            current_transform = current_transform @ joint_transform
            
            transforms[joint_name] = current_transform.copy()
        
        return transforms
    
    def extract_from_state_vector(self, state_vector: np.ndarray, joint_mapping: Dict[str, int]):
        """
        Extract joint positions from a state vector
        
        Args:
            state_vector: Full robot state vector
            joint_mapping: Dictionary mapping joint names to indices in state vector
        """
        for joint_name, index in joint_mapping.items():
            if joint_name in self.joints and index < len(state_vector):
                self.joints[joint_name].update_state(state_vector[index])


class ArmJointSystem(JointCollection):
    """Specialized joint system for dual-arm robots"""
    
    def __init__(self):
        super().__init__()
        self._setup_arm_chains()
    
    def _setup_arm_chains(self):
        """Setup standard left and right arm chains"""
        # Will be populated when joints are added
        pass
    
    def add_arm_joints(self, arm_side: str, joint_names: List[str]):
        """
        Add joints for left or right arm
        
        Args:
            arm_side: 'left' or 'right'
            joint_names: List of joint names for this arm
        """
        if arm_side not in ['left', 'right']:
            raise ValueError("arm_side must be 'left' or 'right'")
        
        # Add to chains
        chain_name = f"{arm_side}_arm"
        self.add_joint_chain(chain_name, joint_names)
        
        # Also add to groups
        self.add_joint_group(chain_name, joint_names)
    
    def get_arm_positions(self, arm_side: str) -> np.ndarray:
        """Get positions for left or right arm"""
        chain_name = f"{arm_side}_arm"
        return self.get_chain_positions(chain_name)
    
    def set_arm_positions(self, arm_side: str, positions: Union[List[float], np.ndarray]):
        """Set positions for left or right arm"""
        chain_name = f"{arm_side}_arm"
        self.set_chain_positions(chain_name, positions)
    
    def extract_arm_joints_from_state(self, state_vector: np.ndarray, 
                                    left_indices: List[int], right_indices: List[int]):
        """
        Extract arm joint positions from state vector
        
        Args:
            state_vector: Full robot state vector
            left_indices: Indices for left arm joints in state vector
            right_indices: Indices for right arm joints in state vector
        """
        if 'left_arm' in self.joint_chains:
            left_positions = state_vector[left_indices]
            self.set_arm_positions('left', left_positions)
        
        if 'right_arm' in self.joint_chains:
            right_positions = state_vector[right_indices]
            self.set_arm_positions('right', right_positions)
    
    def get_dual_arm_positions(self) -> np.ndarray:
        """Get positions for both arms concatenated [left_arm, right_arm]"""
        positions = []
        
        if 'left_arm' in self.joint_chains:
            positions.extend(self.get_arm_positions('left'))
        
        if 'right_arm' in self.joint_chains:
            positions.extend(self.get_arm_positions('right'))
        
        return np.array(positions)

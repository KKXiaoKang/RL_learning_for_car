"""
Dataset Wrapper Classes

This module provides virtual base classes for dataset handling and specific implementations
for robot data extraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class DatasetWrapper(ABC):
    """
    Virtual base class for dataset wrappers
    
    This class defines the interface for extracting different types of data
    from robotics datasets.
    """
    
    def __init__(self, dataset_path: Union[str, Path], **kwargs):
        self.dataset_path = dataset_path
        self.dataset = None
        self.properties = kwargs
    
    @abstractmethod
    def load_dataset(self) -> Any:
        """Load the dataset"""
        pass
    
    @abstractmethod
    def extract_robot_joint_positions(self, data_item: Dict[str, Any]) -> np.ndarray:
        """
        Extract robot joint positions from data item
        
        Args:
            data_item: Single data item from dataset
            
        Returns:
            Array of joint positions
        """
        pass
    
    @abstractmethod
    def extract_camera_images(self, data_item: Dict[str, Any]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Extract camera images from data item
        
        Args:
            data_item: Single data item from dataset
            
        Returns:
            Dictionary mapping camera names to images
        """
        pass
    
    @abstractmethod
    def extract_robot_world_position(self, data_item: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract robot world position
        
        Args:
            data_item: Single data item from dataset
            
        Returns:
            Robot position in world coordinates [x, y, z] or None if not available
        """
        pass
    
    @abstractmethod
    def extract_target_world_position(self, data_item: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract target object world position (e.g., box position)
        
        Args:
            data_item: Single data item from dataset
            
        Returns:
            Target position in world coordinates [x, y, z] or None if not available
        """
        pass
    
    @abstractmethod
    def get_episode_data(self, episode_index: int) -> List[Dict[str, Any]]:
        """
        Get all data items for a specific episode
        
        Args:
            episode_index: Index of the episode
            
        Returns:
            List of data items for the episode
        """
        pass
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        return {
            'dataset_path': str(self.dataset_path),
            'properties': self.properties
        }


class LeRobotDatasetWrapper(DatasetWrapper):
    """
    Implementation of DatasetWrapper for LeRobotDataset
    
    This class handles extraction of robot data from LeRobot format datasets.
    """
    
    def __init__(
        self, 
        repo_id: str,
        root: Optional[Union[str, Path]] = None,
        arm_joint_indices: Optional[Tuple[int, int]] = None,
        robot_base_position_indices: Optional[Tuple[int, int]] = None,
        target_position_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LeRobotDatasetWrapper
        
        Args:
            repo_id: Repository ID for the dataset
            root: Root directory for dataset storage
            arm_joint_indices: (start, end) indices for arm joints in observation.state
            robot_base_position_indices: (start, end) indices for robot base position
            target_position_key: Key for target object position in data
            **kwargs: Additional parameters
        """
        super().__init__(repo_id, **kwargs)
        self.repo_id = repo_id
        self.root = root
        
        # Default arm joint indices for dual-arm robot (14 joints: 7 left + 7 right)
        self.arm_joint_indices = arm_joint_indices or (6, 20)  # Indices 6-19 for arm joints
        
        # Robot base position indices (if available in state)
        self.robot_base_position_indices = robot_base_position_indices
        
        # Target position key
        self.target_position_key = target_position_key
        
        self.dataset: Optional[LeRobotDataset] = None
    
    def load_dataset(self) -> LeRobotDataset:
        """Load the LeRobotDataset"""
        if self.dataset is None:
            self.dataset = LeRobotDataset(
                self.repo_id, 
                root=self.root,
                **self.properties
            )
        return self.dataset
    
    def extract_robot_joint_positions(self, data_item: Dict[str, Any]) -> np.ndarray:
        """
        Extract robot joint positions from observation.state
        
        Args:
            data_item: Single data item from LeRobotDataset
            
        Returns:
            Array of joint positions (typically 14 joints for dual-arm)
        """
        if "observation.state" not in data_item:
            raise KeyError("observation.state not found in data item")
        
        state = data_item["observation.state"]
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        
        # Extract arm joints using specified indices
        start_idx, end_idx = self.arm_joint_indices
        if len(state) < end_idx:
            raise ValueError(f"State vector length {len(state)} is less than required end index {end_idx}")
        
        arm_joints = state[start_idx:end_idx]
        return arm_joints
    
    def extract_camera_images(self, data_item: Dict[str, Any]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Extract camera images from data item
        
        Args:
            data_item: Single data item from LeRobotDataset
            
        Returns:
            Dictionary mapping camera names to images
        """
        images = {}
        
        # Look for image keys in the data
        for key, value in data_item.items():
            if "observation.images" in key:
                camera_name = key.replace("observation.images.", "")
                images[camera_name] = value
            elif key.startswith("images."):
                camera_name = key.replace("images.", "")
                images[camera_name] = value
        
        return images
    
    def extract_robot_world_position(self, data_item: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract robot world position from observation.state
        
        Args:
            data_item: Single data item from LeRobotDataset
            
        Returns:
            Robot position in world coordinates [x, y, z] or None if not available
        """
        if "observation.state" not in data_item:
            return None
        
        if self.robot_base_position_indices is None:
            # Try to extract from first 3 elements (common convention)
            state = data_item["observation.state"]
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            
            if len(state) >= 3:
                return state[:3]  # Assume first 3 are x, y, z
            return None
        
        state = data_item["observation.state"]
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        
        start_idx, end_idx = self.robot_base_position_indices
        if len(state) >= end_idx:
            return state[start_idx:end_idx]
        
        return None
    
    def extract_target_world_position(self, data_item: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract target object world position
        
        Args:
            data_item: Single data item from LeRobotDataset
            
        Returns:
            Target position in world coordinates [x, y, z] or None if not available
        """
        if self.target_position_key is None:
            # Try common target position keys
            possible_keys = [
                "target_position",
                "object_position", 
                "box_position",
                "goal_position",
                "observation.target_position",
                "observation.object_position"
            ]
            
            for key in possible_keys:
                if key in data_item:
                    position = data_item[key]
                    if isinstance(position, torch.Tensor):
                        position = position.numpy()
                    return position[:3] if len(position) >= 3 else position
            
            return None
        
        if self.target_position_key in data_item:
            position = data_item[self.target_position_key]
            if isinstance(position, torch.Tensor):
                position = position.numpy()
            return position[:3] if len(position) >= 3 else position
        
        return None
    
    def get_episode_data(self, episode_index: int) -> List[Dict[str, Any]]:
        """
        Get all data items for a specific episode
        
        Args:
            episode_index: Index of the episode
            
        Returns:
            List of data items for the episode
        """
        if self.dataset is None:
            self.load_dataset()
        
        # Get episode boundaries
        from_idx = self.dataset.episode_data_index["from"][episode_index].item()
        to_idx = self.dataset.episode_data_index["to"][episode_index].item()
        
        episode_data = []
        for frame_idx in range(from_idx, to_idx):
            data_item = self.dataset[frame_idx]
            episode_data.append(data_item)
        
        return episode_data
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get detailed information about the LeRobotDataset"""
        info = super().get_dataset_info()
        
        if self.dataset is not None:
            info.update({
                'repo_id': self.repo_id,
                'num_episodes': len(self.dataset.episode_data_index["from"]),
                'total_frames': len(self.dataset),
                'camera_keys': list(self.dataset.meta.camera_keys) if hasattr(self.dataset.meta, 'camera_keys') else [],
                'fps': getattr(self.dataset.meta, 'fps', None),
                'arm_joint_indices': self.arm_joint_indices,
                'robot_base_position_indices': self.robot_base_position_indices,
                'target_position_key': self.target_position_key
            })
        
        return info
    
    def get_episode_info(self, episode_index: int) -> Dict[str, Any]:
        """
        Get information about a specific episode
        
        Args:
            episode_index: Index of the episode
            
        Returns:
            Dictionary with episode information
        """
        if self.dataset is None:
            self.load_dataset()
        
        if episode_index >= len(self.dataset.episode_data_index["from"]):
            raise IndexError(f"Episode index {episode_index} out of range")
        
        from_idx = self.dataset.episode_data_index["from"][episode_index].item()
        to_idx = self.dataset.episode_data_index["to"][episode_index].item()
        
        return {
            'episode_index': episode_index,
            'from_frame': from_idx,
            'to_frame': to_idx,
            'num_frames': to_idx - from_idx,
            'duration_seconds': (to_idx - from_idx) / getattr(self.dataset.meta, 'fps', 30)
        }
    
    def extract_arm_joints_separately(self, data_item: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left and right arm joints separately
        
        Args:
            data_item: Single data item from LeRobotDataset
            
        Returns:
            Tuple of (left_arm_joints, right_arm_joints)
        """
        arm_joints = self.extract_robot_joint_positions(data_item)
        
        if len(arm_joints) >= 14:
            # Assume first 7 are left arm, next 7 are right arm
            left_arm = arm_joints[:7]
            right_arm = arm_joints[7:14]
            return left_arm, right_arm
        else:
            raise ValueError(f"Expected at least 14 arm joints, got {len(arm_joints)}")
    
    def extract_action_data(self, data_item: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract action data from data item
        
        Args:
            data_item: Single data item from LeRobotDataset
            
        Returns:
            Action array or None if not available
        """
        if "action" not in data_item:
            return None
        
        action = data_item["action"]
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        
        return action

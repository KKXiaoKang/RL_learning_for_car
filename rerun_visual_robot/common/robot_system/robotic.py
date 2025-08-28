"""
Robotic Class

Main robot abstraction that integrates camera systems, joint management, 
model configuration, and torso settings.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import torch
from pathlib import Path

from .camera import Camera, RGBCamera, DepthCamera, CameraSystem, CameraIntrinsics, CameraPose
from .joint import Joint, JointCollection, ArmJointSystem, JointType, JointLimits, JointAxis
from .models import ModelMode, TorsoConfig
from .dataset_wrapper import DatasetWrapper, LeRobotDatasetWrapper


class Robotic:
    """
    Main robot class that abstracts camera systems, joints, model configuration,
    and torso settings for robotics applications.
    """
    
    def __init__(
        self,
        name: str,
        model_mode: ModelMode = ModelMode.FIXED_BASE,
        torso_config: Optional[TorsoConfig] = None,
        urdf_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize robot instance
        
        Args:
            name: Robot name/identifier
            model_mode: Robot base model mode (fixed or floating)
            torso_config: Torso configuration settings
            urdf_path: Path to URDF file for robot description
            **kwargs: Additional robot properties
        """
        self.name = name
        self.model_mode = model_mode
        self.torso_config = torso_config or TorsoConfig()
        self.urdf_path = Path(urdf_path) if urdf_path else None
        self.properties = kwargs
        
        # Initialize subsystems
        self.camera_system = CameraSystem()
        self.joint_system = ArmJointSystem()  # Specialized for dual-arm robots
        
        # Dataset wrapper for data loading
        self.dataset_wrapper: Optional[DatasetWrapper] = None
        
        # Current robot state
        self.current_state = {
            'joint_positions': {},
            'world_position': None,
            'target_position': None,
            'timestamp': 0.0
        }
    
    # === Camera System Methods ===
    
    def add_camera(
        self, 
        name: str, 
        camera_type: str = "RGB",
        intrinsics: Optional[CameraIntrinsics] = None,
        pose: Optional[CameraPose] = None,
        **camera_kwargs
    ) -> Camera:
        """
        Add a camera to the robot
        
        Args:
            name: Camera name
            camera_type: "RGB" or "DEPTH"
            intrinsics: Camera intrinsic parameters
            pose: Camera pose in robot coordinate system
            **camera_kwargs: Additional camera parameters
            
        Returns:
            Created camera instance
        """
        if camera_type.upper() == "RGB":
            camera = RGBCamera(name, intrinsics=intrinsics, pose=pose, **camera_kwargs)
        elif camera_type.upper() == "DEPTH":
            camera = DepthCamera(name, intrinsics=intrinsics, pose=pose, **camera_kwargs)
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")
        
        self.camera_system.add_camera(camera)
        return camera
    
    def get_camera(self, name: str) -> Optional[Camera]:
        """Get camera by name"""
        return self.camera_system.get_camera(name)
    
    def get_all_cameras(self) -> Dict[str, Camera]:
        """Get all cameras"""
        return self.camera_system.get_all_cameras()
    
    def get_rgb_cameras(self) -> Dict[str, RGBCamera]:
        """Get all RGB cameras"""
        return self.camera_system.get_rgb_cameras()
    
    def get_depth_cameras(self) -> Dict[str, DepthCamera]:
        """Get all depth cameras"""
        return self.camera_system.get_depth_cameras()
    
    # === Joint System Methods ===
    
    def add_joint(
        self,
        name: str,
        joint_type: JointType = JointType.REVOLUTE,
        axis: Optional[JointAxis] = None,
        limits: Optional[JointLimits] = None,
        parent_link: Optional[str] = None,
        child_link: Optional[str] = None,
        origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Joint:
        """
        Add a joint to the robot
        
        Args:
            name: Joint name
            joint_type: Type of joint
            axis: Joint rotation/translation axis
            limits: Joint limits
            parent_link: Parent link name
            child_link: Child link name
            origin_xyz: Joint origin position
            origin_rpy: Joint origin orientation
            
        Returns:
            Created joint instance
        """
        joint = Joint(
            name=name,
            joint_type=joint_type,
            axis=axis,
            limits=limits,
            parent_link=parent_link,
            child_link=child_link,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy
        )
        
        self.joint_system.add_joint(joint)
        return joint
    
    def setup_dual_arm_joints(
        self,
        left_arm_names: List[str],
        right_arm_names: List[str],
        joint_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Setup dual-arm joint configuration
        
        Args:
            left_arm_names: List of left arm joint names
            right_arm_names: List of right arm joint names
            joint_configs: Optional dictionary of joint-specific configurations
        """
        # Add left arm joints
        for joint_name in left_arm_names:
            config = joint_configs.get(joint_name, {}) if joint_configs else {}
            self.add_joint(joint_name, **config)
        
        # Add right arm joints
        for joint_name in right_arm_names:
            config = joint_configs.get(joint_name, {}) if joint_configs else {}
            self.add_joint(joint_name, **config)
        
        # Setup arm chains
        self.joint_system.add_arm_joints('left', left_arm_names)
        self.joint_system.add_arm_joints('right', right_arm_names)
    
    def get_joint(self, name: str) -> Optional[Joint]:
        """Get joint by name"""
        return self.joint_system.get_joint(name)
    
    def get_arm_positions(self, arm_side: str) -> np.ndarray:
        """Get joint positions for specified arm"""
        return self.joint_system.get_arm_positions(arm_side)
    
    def set_arm_positions(self, arm_side: str, positions: Union[List[float], np.ndarray]):
        """Set joint positions for specified arm"""
        self.joint_system.set_arm_positions(arm_side, positions)
        
        # Update current state
        arm_name = f"{arm_side}_arm"
        self.current_state['joint_positions'][arm_name] = np.array(positions)
    
    def get_dual_arm_positions(self) -> np.ndarray:
        """Get positions for both arms concatenated"""
        return self.joint_system.get_dual_arm_positions()
    
    def set_dual_arm_positions(self, positions: Union[List[float], np.ndarray]):
        """Set positions for both arms from concatenated array"""
        positions = np.array(positions)
        if len(positions) >= 14:
            self.set_arm_positions('left', positions[:7])
            self.set_arm_positions('right', positions[7:14])
        else:
            raise ValueError(f"Expected at least 14 joint positions, got {len(positions)}")
    
    # === Model and Transform Methods ===
    
    def set_model_mode(self, mode: ModelMode):
        """Set robot model mode"""
        self.model_mode = mode
        self.torso_config.model_mode = mode
    
    def update_torso_config(self, **config_updates):
        """Update torso configuration parameters"""
        for key, value in config_updates.items():
            if hasattr(self.torso_config, key):
                setattr(self.torso_config, key, value)
    
    def get_base_link_transform(self) -> np.ndarray:
        """Get base link transformation matrix"""
        return self.torso_config.get_base_link_transform()
    
    def get_torso_transform(self) -> np.ndarray:
        """Get torso transformation matrix"""
        return self.torso_config.get_torso_transform()
    
    def get_world_torso_transform(self) -> np.ndarray:
        """Get torso transformation in world coordinates"""
        return self.torso_config.get_world_torso_transform()
    
    def compute_forward_kinematics(self, arm_side: str) -> Dict[str, np.ndarray]:
        """
        Compute forward kinematics for specified arm
        
        Args:
            arm_side: 'left' or 'right'
            
        Returns:
            Dictionary mapping joint names to transformation matrices
        """
        base_transform = self.get_world_torso_transform()
        chain_name = f"{arm_side}_arm"
        return self.joint_system.compute_forward_kinematics(chain_name, base_transform)
    
    # === Dataset Integration Methods ===
    
    def set_dataset_wrapper(self, wrapper: DatasetWrapper):
        """Set dataset wrapper for data loading"""
        self.dataset_wrapper = wrapper
    
    def create_lerobot_dataset_wrapper(
        self,
        repo_id: str,
        root: Optional[Union[str, Path]] = None,
        arm_joint_indices: Optional[Tuple[int, int]] = None,
        robot_base_position_indices: Optional[Tuple[int, int]] = None,
        target_position_key: Optional[str] = None,
        **kwargs
    ) -> LeRobotDatasetWrapper:
        """
        Create and set LeRobotDatasetWrapper
        
        Args:
            repo_id: Repository ID for the dataset
            root: Root directory for dataset storage
            arm_joint_indices: (start, end) indices for arm joints
            robot_base_position_indices: (start, end) indices for robot base position
            target_position_key: Key for target object position
            **kwargs: Additional parameters
            
        Returns:
            Created dataset wrapper
        """
        wrapper = LeRobotDatasetWrapper(
            repo_id=repo_id,
            root=root,
            arm_joint_indices=arm_joint_indices,
            robot_base_position_indices=robot_base_position_indices,
            target_position_key=target_position_key,
            **kwargs
        )
        
        self.set_dataset_wrapper(wrapper)
        return wrapper
    
    def load_state_from_data(self, data_item: Dict[str, Any]):
        """
        Load robot state from dataset data item
        
        Args:
            data_item: Data item from dataset
        """
        if self.dataset_wrapper is None:
            raise ValueError("Dataset wrapper not set. Call set_dataset_wrapper() first.")
        
        # Extract joint positions
        try:
            joint_positions = self.dataset_wrapper.extract_robot_joint_positions(data_item)
            if len(joint_positions) >= 14:
                self.set_dual_arm_positions(joint_positions)
        except (KeyError, ValueError) as e:
            print(f"Warning: Could not extract joint positions: {e}")
        
        # Extract world position
        try:
            world_position = self.dataset_wrapper.extract_robot_world_position(data_item)
            if world_position is not None:
                self.current_state['world_position'] = world_position
                # Update torso base link position
                self.update_torso_config(base_link_position=tuple(world_position[:3]))
        except Exception as e:
            print(f"Warning: Could not extract world position: {e}")
        
        # Extract target position
        try:
            target_position = self.dataset_wrapper.extract_target_world_position(data_item)
            if target_position is not None:
                self.current_state['target_position'] = target_position
        except Exception as e:
            print(f"Warning: Could not extract target position: {e}")
        
        # Update timestamp
        if 'timestamp' in data_item:
            timestamp = data_item['timestamp']
            if isinstance(timestamp, torch.Tensor):
                timestamp = timestamp.item()
            self.current_state['timestamp'] = timestamp
    
    def extract_camera_images(self, data_item: Dict[str, Any]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Extract camera images from dataset data item
        
        Args:
            data_item: Data item from dataset
            
        Returns:
            Dictionary mapping camera names to images
        """
        if self.dataset_wrapper is None:
            raise ValueError("Dataset wrapper not set. Call set_dataset_wrapper() first.")
        
        return self.dataset_wrapper.extract_camera_images(data_item)
    
    def estimate_camera_pose_from_name(self, camera_name: str, robot_base_position: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate camera pose based on camera name and robot state
        
        Args:
            camera_name: Name of the camera
            robot_base_position: Base position of the robot
            
        Returns:
            Tuple of (position, rotation_matrix)
        """
        if robot_base_position is None:
            if self.current_state['world_position'] is not None:
                robot_base_position = self.current_state['world_position'][:3]
            else:
                robot_base_position = np.array([0, 0, 0.4])  # Default robot base height
        
        # Default camera parameters
        default_height = 0.8  # meters above ground
        default_distance = 1.5  # meters from robot
        
        # Parse camera name to determine position
        camera_name_lower = camera_name.lower()
        
        if 'front' in camera_name_lower:
            # Front camera - looking at robot from front
            position = robot_base_position + np.array([default_distance, 0, default_height - robot_base_position[2]])
            # Camera looking towards robot (rotate 180 degrees around Z to look backwards)
            orientation = np.array([
                [-1, 0, 0],  # X points backwards (away from robot)
                [0, -1, 0],  # Y points left
                [0, 0, 1]    # Z points up
            ])
        elif 'left' in camera_name_lower and 'wrist' in camera_name_lower:
            # Left wrist camera - attached to left arm
            # Try to get actual left arm end-effector position
            try:
                left_fk = self.compute_forward_kinematics('left')
                if left_fk:
                    # Get the last joint position as end-effector
                    left_eef_transform = list(left_fk.values())[-1]
                    position = left_eef_transform[:3, 3]
                else:
                    left_arm_offset = np.array([0.2, 0.3, 0.2])  # Estimated left arm position
                    position = robot_base_position + left_arm_offset
            except:
                left_arm_offset = np.array([0.2, 0.3, 0.2])  # Estimated left arm position
                position = robot_base_position + left_arm_offset
            
            # Camera looking forward from wrist
            orientation = np.array([
                [0, -1, 0],  # X points left
                [0, 0, -1],  # Y points down
                [1, 0, 0]    # Z points forward
            ])
        elif 'right' in camera_name_lower and 'wrist' in camera_name_lower:
            # Right wrist camera - attached to right arm
            try:
                right_fk = self.compute_forward_kinematics('right')
                if right_fk:
                    # Get the last joint position as end-effector
                    right_eef_transform = list(right_fk.values())[-1]
                    position = right_eef_transform[:3, 3]
                else:
                    right_arm_offset = np.array([0.2, -0.3, 0.2])  # Estimated right arm position
                    position = robot_base_position + right_arm_offset
            except:
                right_arm_offset = np.array([0.2, -0.3, 0.2])  # Estimated right arm position
                position = robot_base_position + right_arm_offset
            
            # Camera looking forward from wrist
            orientation = np.array([
                [0, 1, 0],   # X points right
                [0, 0, -1],  # Y points down
                [1, 0, 0]    # Z points forward
            ])
        elif 'back' in camera_name_lower or 'rear' in camera_name_lower:
            # Back camera - looking at robot from behind
            position = robot_base_position + np.array([-default_distance, 0, default_height - robot_base_position[2]])
            # Camera looking towards robot
            orientation = np.array([
                [1, 0, 0],   # X points forward (towards robot)
                [0, 1, 0],   # Y points right
                [0, 0, 1]    # Z points up
            ])
        elif 'left' in camera_name_lower:
            # Left side camera
            position = robot_base_position + np.array([0, default_distance, default_height - robot_base_position[2]])
            # Camera looking towards robot from left side
            orientation = np.array([
                [0, -1, 0],  # X points right (towards robot)
                [1, 0, 0],   # Y points forward
                [0, 0, 1]    # Z points up
            ])
        elif 'right' in camera_name_lower:
            # Right side camera
            position = robot_base_position + np.array([0, -default_distance, default_height - robot_base_position[2]])
            # Camera looking towards robot from right side
            orientation = np.array([
                [0, 1, 0],   # X points left (towards robot)
                [-1, 0, 0],  # Y points backward
                [0, 0, 1]    # Z points up
            ])
        elif 'top' in camera_name_lower or 'overhead' in camera_name_lower:
            # Top-down camera
            position = robot_base_position + np.array([0, 0, 2.0])  # High above robot
            # Camera looking down
            orientation = np.array([
                [1, 0, 0],   # X points forward
                [0, 1, 0],   # Y points right
                [0, 0, -1]   # Z points down
            ])
        else:
            # Default front camera position
            position = robot_base_position + np.array([default_distance, 0, default_height - robot_base_position[2]])
            orientation = np.array([
                [-1, 0, 0],  # X points backwards
                [0, -1, 0],  # Y points left
                [0, 0, 1]    # Z points up
            ])
        
        return position, orientation
    
    def update_camera_poses_from_robot_state(self):
        """
        Update camera poses based on current robot state and camera names
        """
        robot_base_position = None
        if self.current_state['world_position'] is not None:
            robot_base_position = self.current_state['world_position'][:3]
        
        for camera_name, camera in self.camera_system.cameras.items():
            # Estimate pose from camera name
            position, rotation_matrix = self.estimate_camera_pose_from_name(camera_name, robot_base_position)
            
            # Update camera pose
            from .camera import CameraPose
            camera.pose = CameraPose(position=position, orientation=rotation_matrix)
    
    def visualize_camera_poses_in_rerun(self, entity_base_path: str = "robot_view/cameras"):
        """
        Visualize camera poses in Rerun
        
        Args:
            entity_base_path: Base path for camera entities in Rerun
        """
        try:
            import rerun as rr
            
            for camera_name, camera in self.camera_system.cameras.items():
                if camera.pose:
                    # Log camera position
                    rr.log(f"{entity_base_path}/{camera_name}/position",
                           rr.Points3D([camera.pose.position], radii=[0.03], colors=[[255, 255, 0]]))
                    
                    # Log camera coordinate frame
                    self._log_camera_coordinate_frame(f"{entity_base_path}/{camera_name}/frame", 
                                                    camera.pose.position, camera.pose.get_rotation_matrix())
                    
                    # Log camera frustum
                    self._log_camera_frustum(f"{entity_base_path}/{camera_name}/frustum",
                                           camera.pose.position, camera.pose.get_rotation_matrix())
        except ImportError:
            logger.warning("Rerun not available for camera visualization")
    
    def _log_camera_coordinate_frame(self, entity_path: str, position: np.ndarray, 
                                   rotation_matrix: np.ndarray, scale: float = 0.1):
        """Log camera coordinate frame in Rerun"""
        try:
            import rerun as rr
            
            # Create coordinate frame axes
            axes = np.array([
                [scale, 0, 0],  # X-axis (red)
                [0, scale, 0],  # Y-axis (green)
                [0, 0, scale]   # Z-axis (blue)
            ])
            
            # Transform axes according to the rotation
            transformed_axes = rotation_matrix @ axes.T
            
            # Colors for X, Y, Z axes
            colors = [
                [255, 0, 0],    # Red for X
                [0, 255, 0],    # Green for Y
                [0, 0, 255]     # Blue for Z
            ]
            
            # Log each axis as an arrow
            for i, (axis, color) in enumerate(zip(transformed_axes.T, colors)):
                axis_name = ['x', 'y', 'z'][i]
                rr.log(f"{entity_path}_{axis_name}",
                       rr.Arrows3D(
                           origins=[position],
                           vectors=[axis],
                           colors=[color],
                           radii=[scale * 0.05]
                       ))
        except ImportError:
            pass
    
    def _log_camera_frustum(self, entity_path: str, position: np.ndarray, 
                          rotation_matrix: np.ndarray, fov_degrees: float = 60, depth: float = 0.3):
        """Log camera field of view frustum in Rerun"""
        try:
            import rerun as rr
            
            # Calculate frustum corners
            fov_rad = np.radians(fov_degrees)
            half_fov = fov_rad / 2
            
            frustum_width = 2 * depth * np.tan(half_fov)
            frustum_height = frustum_width  # Assuming square aspect ratio
            
            # Local frustum corners (camera coordinate system, looking along +X axis)
            local_corners = np.array([
                [0, 0, 0],  # Camera center
                [depth, -frustum_width/2, -frustum_height/2],  # Far-bottom-left
                [depth, frustum_width/2, -frustum_height/2],   # Far-bottom-right
                [depth, frustum_width/2, frustum_height/2],    # Far-top-right
                [depth, -frustum_width/2, frustum_height/2],   # Far-top-left
            ])
            
            # Transform to world coordinates
            world_corners = []
            for corner in local_corners:
                world_corner = position + rotation_matrix @ corner
                world_corners.append(world_corner)
            
            world_corners = np.array(world_corners)
            
            # Define frustum edges
            frustum_lines = [
                [world_corners[0], world_corners[1]],  # Center to corners
                [world_corners[0], world_corners[2]],
                [world_corners[0], world_corners[3]],
                [world_corners[0], world_corners[4]],
                [world_corners[1], world_corners[2]],  # Far plane edges
                [world_corners[2], world_corners[3]],
                [world_corners[3], world_corners[4]],
                [world_corners[4], world_corners[1]],
            ]
            
            # Log frustum lines
            rr.log(entity_path,
                   rr.LineStrips3D(frustum_lines, 
                                 colors=[[255, 255, 0]], 
                                 radii=[0.002]))
        except ImportError:
            pass
    
    def get_episode_data(self, episode_index: int) -> List[Dict[str, Any]]:
        """
        Get all data for a specific episode
        
        Args:
            episode_index: Episode index
            
        Returns:
            List of data items for the episode
        """
        if self.dataset_wrapper is None:
            raise ValueError("Dataset wrapper not set. Call set_dataset_wrapper() first.")
        
        return self.dataset_wrapper.get_episode_data(episode_index)
    
    # === Utility Methods ===
    
    def get_robot_info(self) -> Dict[str, Any]:
        """Get comprehensive robot information"""
        info = {
            'name': self.name,
            'model_mode': self.model_mode,
            'torso_config': {
                'base_link_position': self.torso_config.base_link_position,
                'torso_position': self.torso_config.torso_position,
                'torso_orientation': self.torso_config.torso_orientation,
            },
            'urdf_path': str(self.urdf_path) if self.urdf_path else None,
            'cameras': {name: {
                'type': type(cam).__name__,
                'format': getattr(cam, 'image_format', 'UNKNOWN')
            } for name, cam in self.camera_system.get_all_cameras().items()},
            'joints': {
                'total_joints': len(self.joint_system.joints),
                'joint_chains': list(self.joint_system.joint_chains.keys()),
                'joint_groups': list(self.joint_system.joint_groups.keys())
            },
            'current_state': self.current_state.copy(),
            'properties': self.properties
        }
        
        if self.dataset_wrapper:
            info['dataset_info'] = self.dataset_wrapper.get_dataset_info()
        
        return info
    
    def reset_state(self):
        """Reset robot to initial state"""
        # Reset joint positions
        for joint in self.joint_system.joints.values():
            joint.update_state(0.0)
        
        # Reset current state
        self.current_state = {
            'joint_positions': {},
            'world_position': None,
            'target_position': None,
            'timestamp': 0.0
        }
    
    def validate_configuration(self) -> List[str]:
        """
        Validate robot configuration and return any issues
        
        Returns:
            List of validation issues (empty if all good)
        """
        issues = []
        
        # Check if dual-arm setup is complete
        if 'left_arm' not in self.joint_system.joint_chains:
            issues.append("Left arm joint chain not configured")
        
        if 'right_arm' not in self.joint_system.joint_chains:
            issues.append("Right arm joint chain not configured")
        
        # Check camera setup
        if len(self.camera_system.cameras) == 0:
            issues.append("No cameras configured")
        
        # Check URDF path
        if self.urdf_path and not self.urdf_path.exists():
            issues.append(f"URDF file not found: {self.urdf_path}")
        
        return issues


# Factory functions for common robot configurations

def create_dual_arm_robot(
    name: str = "DualArmRobot",
    urdf_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Robotic:
    """
    Create a standard dual-arm robot configuration
    
    Args:
        name: Robot name
        urdf_path: Path to URDF file
        **kwargs: Additional robot parameters
        
    Returns:
        Configured Robotic instance
    """
    robot = Robotic(name=name, urdf_path=urdf_path, **kwargs)
    
    # Setup standard dual-arm joint names
    left_arm_joints = [f'zarm_l{i}_joint' for i in range(1, 8)]
    right_arm_joints = [f'zarm_r{i}_joint' for i in range(1, 8)]
    
    robot.setup_dual_arm_joints(left_arm_joints, right_arm_joints)
    
    # Add common cameras
    robot.add_camera('front', camera_type='RGB')
    
    return robot


def create_kuavo_robot(
    urdf_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Robotic:
    """
    Create Kuavo robot configuration
    
    Args:
        urdf_path: Path to URDF file (auto-detected if None)
        **kwargs: Additional robot parameters
        
    Returns:
        Configured Kuavo robot instance
    """
    # Auto-detect URDF path if not provided
    if urdf_path is None:
        possible_paths = [
            "/home/lab/RL/src/biped_s45/urdf/biped_s45.urdf",
            "./src/biped_s45/urdf/biped_s45.urdf",
            "./gym_hil/assets/biped_s45.urdf"
        ]
        for path in possible_paths:
            if Path(path).exists():
                urdf_path = path
                break
    
    # Create robot with Kuavo-specific configuration
    torso_config = TorsoConfig(
        base_link_position=(0.0, 0.0, 0.0),
        torso_position=(0.0, 0.0, 0.4),
        model_mode=ModelMode.FLOATING_BASE
    )
    
    robot = Robotic(
        name="Kuavo",
        model_mode=ModelMode.FLOATING_BASE,
        torso_config=torso_config,
        urdf_path=urdf_path,
        **kwargs
    )
    
    # Setup Kuavo arm joints
    left_arm_joints = [f'zarm_l{i}_joint' for i in range(1, 8)]
    right_arm_joints = [f'zarm_r{i}_joint' for i in range(1, 8)]
    robot.setup_dual_arm_joints(left_arm_joints, right_arm_joints)
    
    # Add Kuavo cameras
    robot.add_camera('front', camera_type='RGB')
    robot.add_camera('torso', camera_type='RGB')
    robot.add_camera('waist', camera_type='RGB')
    
    return robot

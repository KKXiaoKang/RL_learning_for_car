"""
Robotic-Rerun Adapter

This module provides adapters and utilities to integrate the Robotic class
with Rerun visualization, bridging the gap between robot data structures
and 3D visualization.
"""

import numpy as np
import rerun as rr
import torch
from typing import Dict, Any, Optional, List, Tuple
import logging

from common.robot_system import Robotic, Camera, Joint

logger = logging.getLogger(__name__)


class RoboticRerunAdapter:
    """
    Adapter class that integrates Robotic instances with Rerun visualization
    """
    
    def __init__(self, robot: Robotic, enable_urdf_visualization: bool = True):
        """
        Initialize adapter
        
        Args:
            robot: Robotic instance to adapt
            enable_urdf_visualization: Whether to use URDF-based visualization
        """
        self.robot = robot
        self.enable_urdf_visualization = enable_urdf_visualization
        
        # Import the original robot visualizer for URDF support
        try:
            from visualize_dataset_robotics import RobotVisualizer, initialize_robot_visualizer, _robot_visualizer
            self.robot_visualizer = _robot_visualizer
            if self.robot_visualizer is None and self.robot.urdf_path:
                initialize_robot_visualizer(str(self.robot.urdf_path))
                from visualize_dataset_robotics import _robot_visualizer
                self.robot_visualizer = _robot_visualizer
        except ImportError:
            logger.warning("Could not import URDF visualization. Using simplified visualization.")
            self.robot_visualizer = None
            self.enable_urdf_visualization = False
    
    def log_robot_state(self, frame_index: int):
        """
        Log complete robot state to Rerun
        
        Args:
            frame_index: Current frame index
        """
        # Log basic robot information
        self._log_robot_base_state()
        
        # Log joint states
        self._log_joint_states()
        
        # Log robot visualization (URDF or simplified)
        self._log_robot_visualization(frame_index)
        
        # Log forward kinematics
        self._log_forward_kinematics()
        
        # Log target positions
        self._log_target_positions()
    
    def log_camera_system(self, camera_images: Dict[str, Any]):
        """
        Log camera images and poses using robot's camera system
        
        Args:
            camera_images: Dictionary of camera images from dataset
        """
        for camera_name, image_data in camera_images.items():
            # Log camera image
            self._log_camera_image(camera_name, image_data)
            
            # Log camera pose if available
            camera = self.robot.get_camera(camera_name)
            if camera and camera.pose:
                self._log_camera_pose(camera_name, camera)
    
    def _log_robot_base_state(self):
        """Log robot base position and orientation"""
        try:
            # Robot world position
            if self.robot.current_state['world_position'] is not None:
                world_pos = self.robot.current_state['world_position'][:3]
                rr.log("robot_view/robot/base_position", 
                       rr.Points3D([world_pos], radii=[0.05], colors=[[100, 255, 100]]))
                
                # Log base coordinate frame
                base_transform = self.robot.get_world_torso_transform()
                self._log_coordinate_frame("robot_view/robot/base_frame", base_transform)
            
            # Robot torso position
            torso_transform = self.robot.get_torso_transform()
            torso_pos = torso_transform[:3, 3]
            rr.log("robot_view/robot/torso_position",
                   rr.Points3D([torso_pos], radii=[0.04], colors=[[150, 150, 255]]))
                   
        except Exception as e:
            logger.warning(f"Failed to log robot base state: {e}")
    
    def _log_joint_states(self):
        """Log individual joint states as scalars"""
        try:
            # Left arm joints
            left_positions = self.robot.get_arm_positions('left')
            for i, position in enumerate(left_positions):
                joint_name = f"zarm_l{i+1}_joint"
                rr.log(f"robot_view/robot/joint_angles/left_arm/{joint_name}", 
                       rr.Scalars(np.degrees(position)))
            
            # Right arm joints
            right_positions = self.robot.get_arm_positions('right')
            for i, position in enumerate(right_positions):
                joint_name = f"zarm_r{i+1}_joint"
                rr.log(f"robot_view/robot/joint_angles/right_arm/{joint_name}", 
                       rr.Scalars(np.degrees(position)))
                       
        except Exception as e:
            logger.warning(f"Failed to log joint states: {e}")
    
    def _log_robot_visualization(self, frame_index: int):
        """Log robot 3D visualization"""
        try:
            # Use URDF visualization if available
            if self.enable_urdf_visualization and self.robot_visualizer:
                dual_arm_positions = self.robot.get_dual_arm_positions()
                if len(dual_arm_positions) >= 14:
                    self.robot_visualizer.log_robot_with_urdf(dual_arm_positions, frame_index)
            else:
                # Use simplified visualization
                self._log_simplified_robot_visualization()
                
        except Exception as e:
            logger.warning(f"Failed to log robot visualization: {e}")
    
    def _log_forward_kinematics(self):
        """Log forward kinematics results"""
        try:
            # Compute and log left arm forward kinematics
            left_fk = self.robot.compute_forward_kinematics('left')
            self._log_arm_kinematics('left', left_fk, [255, 100, 100])
            
            # Compute and log right arm forward kinematics
            right_fk = self.robot.compute_forward_kinematics('right')
            self._log_arm_kinematics('right', right_fk, [100, 100, 255])
            
        except Exception as e:
            logger.warning(f"Failed to log forward kinematics: {e}")
    
    def _log_arm_kinematics(self, arm_side: str, fk_results: Dict[str, np.ndarray], color: List[int]):
        """Log forward kinematics for one arm"""
        positions = []
        
        for joint_name, transform in fk_results.items():
            position = transform[:3, 3]
            positions.append(position)
            
            # Log individual joint position
            rr.log(f"robot_view/robot/{arm_side}_arm/{joint_name}/position",
                   rr.Points3D([position], radii=[0.02], colors=[color]))
            
            # Log joint coordinate frame
            self._log_coordinate_frame(f"robot_view/robot/{arm_side}_arm/{joint_name}/frame", 
                                     transform, scale=0.05)
        
        # Log arm skeleton
        if len(positions) > 1:
            rr.log(f"robot_view/robot/{arm_side}_arm/skeleton",
                   rr.LineStrips3D([positions], colors=[color], radii=[0.01]))
    
    def _log_target_positions(self):
        """Log target and goal positions"""
        try:
            # Target object position
            if self.robot.current_state['target_position'] is not None:
                target_pos = self.robot.current_state['target_position'][:3]
                rr.log("robot_view/robot/target_position",
                       rr.Points3D([target_pos], radii=[0.03], colors=[[255, 100, 100]]))
                
                # Draw line from robot to target
                if self.robot.current_state['world_position'] is not None:
                    robot_pos = self.robot.current_state['world_position'][:3]
                    rr.log("robot_view/robot/target_connection",
                           rr.LineStrips3D([[robot_pos, target_pos]], 
                                         colors=[[255, 255, 0]], radii=[0.005]))
                                         
        except Exception as e:
            logger.warning(f"Failed to log target positions: {e}")
    
    def _log_camera_image(self, camera_name: str, image_data: Any):
        """Log camera image to rerun"""
        try:
            if isinstance(image_data, torch.Tensor):
                # Convert tensor to numpy
                if image_data.ndim == 3:  # CHW format
                    image_np = (image_data * 255).type(torch.uint8).permute(1, 2, 0).numpy()
                else:
                    image_np = image_data.numpy()
            else:
                image_np = image_data
            
            rr.log(f"robot_view/cameras/{camera_name}/image", rr.Image(image_np))
            
        except Exception as e:
            logger.warning(f"Failed to log camera image for {camera_name}: {e}")
    
    def _log_camera_pose(self, camera_name: str, camera: Camera):
        """Log camera pose visualization"""
        try:
            if camera.pose:
                position = camera.pose.position
                rotation_matrix = camera.pose.get_rotation_matrix()
                
                # Log camera position
                rr.log(f"robot_view/cameras/{camera_name}/position",
                       rr.Points3D([position], radii=[0.03], colors=[[255, 255, 0]]))
                
                # Log camera coordinate frame
                camera_transform = np.eye(4)
                camera_transform[:3, :3] = rotation_matrix
                camera_transform[:3, 3] = position
                self._log_coordinate_frame(f"robot_view/cameras/{camera_name}/frame", 
                                         camera_transform, scale=0.1)
                
                # Log camera frustum
                self._log_camera_frustum(camera_name, position, rotation_matrix)
                
        except Exception as e:
            logger.warning(f"Failed to log camera pose for {camera_name}: {e}")
    
    def _log_camera_frustum(self, camera_name: str, position: np.ndarray, 
                           rotation_matrix: np.ndarray, fov_degrees: float = 60, depth: float = 0.3):
        """Log camera field of view frustum"""
        try:
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
            rr.log(f"robot_view/cameras/{camera_name}/frustum",
                   rr.LineStrips3D(frustum_lines, 
                                 colors=[[255, 255, 0]], 
                                 radii=[0.002]))
                                 
        except Exception as e:
            logger.warning(f"Failed to log camera frustum for {camera_name}: {e}")
    
    def _log_coordinate_frame(self, entity_path: str, transform: np.ndarray, scale: float = 0.1):
        """Log coordinate frame (TF-style) with X(red), Y(green), Z(blue) axes"""
        try:
            position = transform[:3, 3]
            rotation_matrix = transform[:3, :3]
            
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
                       
        except Exception as e:
            logger.warning(f"Failed to log coordinate frame: {e}")
    
    def _log_simplified_robot_visualization(self):
        """Fallback simplified robot visualization"""
        try:
            # Log robot torso
            torso_transform = self.robot.get_torso_transform()
            torso_pos = torso_transform[:3, 3]
            rr.log("robot_view/robot/torso",
                   rr.Points3D([torso_pos], radii=[0.05], colors=[[100, 100, 100]]))
            
            # Log simplified arm positions
            left_positions = self.robot.get_arm_positions('left')
            right_positions = self.robot.get_arm_positions('right')
            
            if len(left_positions) >= 7:
                # Simplified left arm visualization
                left_base = torso_pos + np.array([0, 0.15, 0])
                rr.log("robot_view/robot/left_arm/base",
                       rr.Points3D([left_base], radii=[0.03], colors=[[255, 100, 100]]))
            
            if len(right_positions) >= 7:
                # Simplified right arm visualization
                right_base = torso_pos + np.array([0, -0.15, 0])
                rr.log("robot_view/robot/right_arm/base",
                       rr.Points3D([right_base], radii=[0.03], colors=[[100, 100, 255]]))
                       
        except Exception as e:
            logger.warning(f"Failed to log simplified robot visualization: {e}")
    
    def get_robot_info_summary(self) -> str:
        """Get formatted robot information summary"""
        info = self.robot.get_robot_info()
        
        summary = f"""
Robot Information:
- Name: {info['name']}
- Model Mode: {info['model_mode']}
- URDF Path: {info['urdf_path']}
- Cameras: {len(info['cameras'])} ({', '.join(info['cameras'].keys())})
- Joints: {info['joints']['total_joints']} total
- Joint Chains: {', '.join(info['joints']['joint_chains'])}
- Current State: {info['current_state']['timestamp']}s
"""
        
        if 'dataset_info' in info:
            dataset_info = info['dataset_info']
            summary += f"""
Dataset Information:
- Repository: {dataset_info.get('repo_id', 'N/A')}
- Episodes: {dataset_info.get('num_episodes', 'N/A')}
- Total Frames: {dataset_info.get('total_frames', 'N/A')}
- FPS: {dataset_info.get('fps', 'N/A')}
- Cameras: {len(dataset_info.get('camera_keys', []))}
"""
        
        return summary.strip()


def create_adapter_for_dataset(repo_id: str, root: Optional[str] = None) -> RoboticRerunAdapter:
    """
    Convenience function to create a RoboticRerunAdapter for a dataset
    
    Args:
        repo_id: Dataset repository ID
        root: Root directory for dataset
        
    Returns:
        Configured RoboticRerunAdapter
    """
    from common.robot_system import create_kuavo_robot
    
    # Create robot
    robot = create_kuavo_robot()
    
    # Setup dataset wrapper
    robot.create_lerobot_dataset_wrapper(
        repo_id=repo_id,
        root=root,
        arm_joint_indices=(6, 20),
        robot_base_position_indices=(0, 3),
        target_position_key="observation.object_position"
    )
    
    # Create adapter
    adapter = RoboticRerunAdapter(robot, enable_urdf_visualization=True)
    
    return adapter

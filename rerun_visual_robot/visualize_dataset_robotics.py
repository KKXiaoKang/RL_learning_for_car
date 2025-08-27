#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

This enhanced version includes 3D robot visualization using rerun-sdk 0.24.1 for robotics datasets.
The robot visualization automatically extracts 14 arm joint angles from observation.state and 
displays a 3D representation of the robot's dual-arm configuration.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Robot Visualization Features:
- 3D visualization of dual-arm robot configuration with URDF support
- Real-time joint angle monitoring with degree conversion
- Mesh-based robot visualization (if urdfpy and trimesh are available)
- Forward kinematics computation for arm poses
- Color-coded left (red) and right (blue) arms
- Torso and shoulder connections visualization
- TF-style coordinate frame visualization for each joint (X=red, Y=green, Z=blue)
- Camera pose visualization with position, orientation, and field of view frustums
- 3D image plane projection showing camera images in 3D space
- Robot-centric view configuration for optimal visualization
- Automatic fallback to simplified visualization if URDF/mesh loading fails

Installation Requirements:
For full URDF and mesh support, install additional dependencies:
```bash
pip install urdfpy trimesh
```

Examples:

- Visualize robotics data stored on a local machine (with robot 3D view):
```
local$ python lerobot/scripts/visualize_dataset_robotics.py \
    --repo-id your-robotics-dataset \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python lerobot/scripts/visualize_dataset_robotics.py \
    --repo-id your-robotics-dataset \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/your_robotics_dataset_episode_0.rrd .
local$ rerun your_robotics_dataset_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ python lerobot/scripts/visualize_dataset_robotics.py \
    --repo-id your-robotics-dataset \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

Robot Joint Configuration:
The script expects 14 arm joints in observation.state at indices [6:20]:
- Joints 0-6: Left arm (zarm_l1_joint to zarm_l7_joint)
- Joints 7-13: Right arm (zarm_r1_joint to zarm_r7_joint)
Joint angles should be in radians.

URDF and Mesh Loading:
The script automatically searches for URDF files in these locations:
- /home/lab/RL/src/biped_s45/urdf/biped_s45.urdf (primary location)
- ./src/biped_s45/urdf/biped_s45.urdf (relative from project root)
- ./gym_hil/assets/biped_s45.urdf (fallback location)
- Or specify custom path via --urdf-path argument

Visualization Modes:
1. Full URDF mode: Uses accurate robot geometry and meshes
2. Simplified mode: Uses basic geometric shapes as fallback

Camera Visualization:
The script automatically detects and visualizes camera poses based on camera names:
- 'front': Front-facing camera positioned in front of robot
- 'left_wrist'/'right_wrist': Wrist-mounted cameras attached to robot arms
- 'left'/'right': Side cameras positioned at robot sides
- 'back'/'rear': Rear cameras positioned behind robot
- 'top'/'overhead': Top-down cameras positioned above robot

Camera Coordinate System:
- Cameras look along the +X axis (forward direction)
- Y axis points to the right, Z axis points up
- This follows robotics convention where X is forward

Each camera visualization includes:
- Yellow position marker
- TF-style coordinate frame (X=red, Y=green, Z=blue)
- Field of view frustum (yellow wireframe) extending along +X axis
- 3D projected image plane showing actual camera image

Interactive Features:
- Joint coordinate frames can be toggled in the Rerun viewer
- Camera frustums help understand spatial relationships
- Image planes show what each camera sees in 3D context
- Robot-centric view provides optimal observation angle

Head Joint Configuration:
The script uses fixed head joint angles since they're not available in observation state.
You can adjust the HEAD_JOINT_ANGLES global variable to match your robot's head pose:
- HEAD_JOINT_ANGLES[0]: zhead_1_joint (yaw, left-right head rotation in radians)
- HEAD_JOINT_ANGLES[1]: zhead_2_joint (pitch, up-down head rotation in radians)
Default: [0.0, np.radians(30)] = 0° yaw, 30° pitch (looking slightly down)

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


# Robot visualization constants and functions
class RobotVisualizer:
    """
    Robot visualizer with URDF and mesh support for rerun-sdk 0.24.1
    Uses xml.etree.ElementTree for URDF parsing instead of urdfpy
    """
    
    def __init__(self, urdf_path: str = None):
        self.urdf_root = None
        self.urdf_path = urdf_path
        self.joint_names_left = [f'zarm_l{i}_joint' for i in range(1, 8)]
        self.joint_names_right = [f'zarm_r{i}_joint' for i in range(1, 8)]
        self.all_joint_names = self.joint_names_left + self.joint_names_right
        self.links = {}
        self.joints = {}
        
        # Try to load URDF if available
        if urdf_path:
            self.load_urdf(urdf_path)
        else:
            logging.warning("URDF not loaded - using simplified visualization")
    
    def load_urdf(self, urdf_path: str):
        """Load URDF file using xml.etree.ElementTree"""
        try:
            if os.path.exists(urdf_path):
                tree = ET.parse(urdf_path)
                self.urdf_root = tree.getroot()
                
                # Parse links
                for link_elem in self.urdf_root.findall('link'):
                    link_name = link_elem.get('name')
                    self.links[link_name] = {
                        'name': link_name,
                        'visual': [],
                        'collision': []
                    }
                    
                    # Parse visual elements
                    for visual_elem in link_elem.findall('visual'):
                        visual_data = self._parse_visual_element(visual_elem, urdf_path)
                        if visual_data:
                            self.links[link_name]['visual'].append(visual_data)
                
                # Parse joints
                for joint_elem in self.urdf_root.findall('joint'):
                    joint_name = joint_elem.get('name')
                    joint_type = joint_elem.get('type')
                    
                    parent_elem = joint_elem.find('parent')
                    child_elem = joint_elem.find('child')
                    origin_elem = joint_elem.find('origin')
                    axis_elem = joint_elem.find('axis')
                    
                    # Parse joint axis
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
                
                # Log arm joint details for debugging
                left_joints = [j for j in self.joints.keys() if 'zarm_l' in j]
                right_joints = [j for j in self.joints.keys() if 'zarm_r' in j]
                logging.info(f"Left arm joints found: {sorted(left_joints)}")
                logging.info(f"Right arm joints found: {sorted(right_joints)}")
                
                # Log joint details
                for joint_name in sorted(self.joints.keys()):
                    if 'zarm' in joint_name:
                        joint_data = self.joints[joint_name]
                        logging.info(f"Joint {joint_name}: axis={joint_data.get('axis')}, "
                                   f"parent={joint_data.get('parent')}, child={joint_data.get('child')}")
                
                return True
            else:
                logging.warning(f"URDF file not found at {urdf_path}")
                return False
        except Exception as e:
            logging.error(f"Failed to load URDF: {e}")
            return False
    
    def _parse_visual_element(self, visual_elem, urdf_path):
        """Parse visual element from URDF"""
        visual_data = {}
        
        # Parse geometry
        geometry_elem = visual_elem.find('geometry')
        if geometry_elem is not None:
            mesh_elem = geometry_elem.find('mesh')
            if mesh_elem is not None:
                filename = mesh_elem.get('filename')
                if filename:
                    # Convert relative paths to absolute paths
                    if not os.path.isabs(filename):
                        # Remove package:// prefix if present
                        if filename.startswith('package://'):
                            filename = filename.replace('package://', '')
                            filename = filename.replace('kuavo_assets/models/biped_s45/', '')
                        
                        # Try to resolve relative to URDF directory first
                        urdf_dir = os.path.dirname(urdf_path)
                        possible_paths = [
                            # Direct relative path
                            os.path.join(urdf_dir, filename),
                            # Parent directory relative
                            os.path.join(urdf_dir, '..', filename),
                            # In meshes subdirectory
                            os.path.join(urdf_dir, '..', 'meshes', os.path.basename(filename)),
                            # Multiple levels up
                            os.path.join(urdf_dir, '..', '..', 'meshes', os.path.basename(filename)),
                            # Try in src directory structure
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
                            logging.info(f"Found mesh file: {filename}")
                        else:
                            logging.warning(f"Mesh file not found, tried: {possible_paths}")
                            filename = possible_paths[0]  # fallback
                    
                    visual_data['mesh_file'] = filename
                    scale = mesh_elem.get('scale')
                    if scale:
                        visual_data['scale'] = [float(x) for x in scale.split()]
                    else:
                        visual_data['scale'] = [1.0, 1.0, 1.0]
            
            # Parse other geometry types
            box_elem = geometry_elem.find('box')
            if box_elem is not None:
                size = box_elem.get('size')
                if size:
                    visual_data['box_size'] = [float(x) for x in size.split()]
            
            cylinder_elem = geometry_elem.find('cylinder')
            if cylinder_elem is not None:
                radius = cylinder_elem.get('radius')
                length = cylinder_elem.get('length')
                if radius and length:
                    visual_data['cylinder'] = {'radius': float(radius), 'length': float(length)}
        
        # Parse origin
        origin_elem = visual_elem.find('origin')
        if origin_elem is not None:
            visual_data['origin'] = self._parse_origin(origin_elem)
        else:
            visual_data['origin'] = {'xyz': [0,0,0], 'rpy': [0,0,0]}
        
        return visual_data if visual_data else None
    
    def _parse_origin(self, origin_elem):
        """Parse origin element from URDF"""
        xyz = origin_elem.get('xyz', '0 0 0')
        rpy = origin_elem.get('rpy', '0 0 0')
        
        return {
            'xyz': [float(x) for x in xyz.split()],
            'rpy': [float(x) for x in rpy.split()]
        }
    
    def get_joint_configuration(self, joint_angles_14):
        """
        Create joint configuration dictionary for URDF
        
        Args:
            joint_angles_14: numpy array of 14 joint angles [left_arm_7, right_arm_7]
        
        Returns:
            Dictionary mapping joint names to angles
        """
        if len(joint_angles_14) < 14:
            return {}
        
        joint_config = {}
        
        # Left arm joints
        for i, joint_name in enumerate(self.joint_names_left):
            joint_config[joint_name] = joint_angles_14[i]
        
        # Right arm joints  
        for i, joint_name in enumerate(self.joint_names_right):
            joint_config[joint_name] = joint_angles_14[i + 7]
        
        return joint_config
    
    def _compute_forward_kinematics(self, joint_config):
        """
        Compute forward kinematics for the robot using joint configuration
        
        Args:
            joint_config: Dictionary mapping joint names to angles
            
        Returns:
            Dictionary mapping link names to 4x4 transform matrices
        """
        link_transforms = {}
        
        # Start with base link at identity
        link_transforms['base_link'] = np.eye(4)
        
        # Build kinematic chain by traversing joints in correct order
        # Define the kinematic chains for arms
        arm_chains = {
            'left': ['zarm_l1_joint', 'zarm_l2_joint', 'zarm_l3_joint', 'zarm_l4_joint', 
                    'zarm_l5_joint', 'zarm_l6_joint', 'zarm_l7_joint'],
            'right': ['zarm_r1_joint', 'zarm_r2_joint', 'zarm_r3_joint', 'zarm_r4_joint', 
                     'zarm_r5_joint', 'zarm_r6_joint', 'zarm_r7_joint']
        }
        
        # Process each arm chain
        for arm_side, chain in arm_chains.items():
            for joint_name in chain:
                if joint_name in self.joints:
                    joint_data = self.joints[joint_name]
                    parent_link = joint_data.get('parent')
                    child_link = joint_data.get('child')
                    joint_type = joint_data.get('type')
                    origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                    
                    if parent_link in link_transforms and child_link:
                        # Get parent transform
                        parent_transform = link_transforms[parent_link]
                        
                        # Create joint transform
                        joint_transform = np.eye(4)
                        
                        # Apply joint origin translation
                        joint_transform[:3, 3] = origin['xyz']
                        
                        # Apply joint origin rotation
                        rpy = origin['rpy']
                        joint_transform[:3, :3] = self._rpy_to_rotation_matrix(rpy)
                        
                        # Apply joint rotation if it's a revolute joint and we have angle data
                        if joint_type == 'revolute' and joint_name in joint_config:
                            joint_angle = joint_config[joint_name]
                            
                            # Get the joint axis from URDF (stored during parsing)
                            joint_axis = joint_data.get('axis', [0, 0, 1])  # default to Z-axis
                            
                            # Create rotation matrix around the specified axis
                            joint_rotation = self._axis_angle_to_rotation_matrix(joint_axis, joint_angle)
                            joint_transform[:3, :3] = joint_transform[:3, :3] @ joint_rotation
                        
                        # Compute child link transform
                        link_transforms[child_link] = parent_transform @ joint_transform
        
        return link_transforms
    
    def _rpy_to_rotation_matrix(self, rpy):
        """Convert RPY to 3x3 rotation matrix"""
        roll, pitch, yaw = rpy
        
        # Rotation matrices for each axis
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
        
        # Combined rotation (ZYX order)
        return R_z @ R_y @ R_x
    
    def _log_coordinate_frame(self, entity_path, transform, scale=0.1):
        """
        Log a coordinate frame (TF-style) with X(red), Y(green), Z(blue) axes
        
        Args:
            entity_path: rerun entity path for the coordinate frame
            transform: 4x4 transformation matrix
            scale: scale factor for the arrows
        """
        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        
        # Create coordinate frame axes (X, Y, Z)
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
            rr.log(f"{entity_path}/frame_{axis_name}",
                   rr.Arrows3D(
                       origins=[position],
                       vectors=[axis],
                       colors=[color],
                       radii=[scale * 0.05]
                   ))
    
    def _log_camera_pose(self, entity_path, camera_position, camera_orientation, 
                        camera_name, fov_degrees=60, scale=0.1):
        """
        Log camera pose with position, orientation and field of view visualization
        
        Args:
            entity_path: rerun entity path for the camera
            camera_position: 3D position of the camera [x, y, z]
            camera_orientation: quaternion [x, y, z, w] or rotation matrix
            camera_name: name of the camera for labeling
            fov_degrees: field of view in degrees
            scale: scale factor for visualization
        """
        # Convert to numpy arrays
        position = np.array(camera_position)
        
        # Handle different orientation formats
        if isinstance(camera_orientation, np.ndarray):
            if camera_orientation.shape == (4,):
                # Quaternion format [x, y, z, w]
                quat = camera_orientation
                rotation_matrix = self._quaternion_to_rotation_matrix(quat)
            elif camera_orientation.shape == (3, 3):
                # Already a rotation matrix
                rotation_matrix = camera_orientation
                quat = self._rotation_matrix_to_quaternion(rotation_matrix)
            else:
                raise ValueError(f"Unsupported orientation format: {camera_orientation.shape}")
        else:
            raise ValueError("camera_orientation must be a numpy array")
        
        # Log camera position as a point
        rr.log(f"{entity_path}/position",
               rr.Points3D([position], 
                         radii=[scale * 0.5], 
                         colors=[[255, 255, 0]]))  # Yellow for camera
        
        # Log coordinate frame for camera
        camera_transform = np.eye(4)
        camera_transform[:3, :3] = rotation_matrix
        camera_transform[:3, 3] = position
        self._log_coordinate_frame(f"{entity_path}/frame", camera_transform, scale)
        
        # Log camera frustum (field of view visualization)
        self._log_camera_frustum(entity_path, position, rotation_matrix, fov_degrees, scale)
        
        # Log camera name as text
        rr.log(f"{entity_path}/label",
               rr.TextDocument(camera_name))
    
    def _log_camera_frustum(self, entity_path, position, rotation_matrix, 
                           fov_degrees, scale, depth=0.3):
        """
        Log camera frustum to visualize field of view
        
        Args:
            entity_path: rerun entity path
            position: camera position
            rotation_matrix: camera orientation
            fov_degrees: field of view in degrees
            scale: scale factor
            depth: depth of the frustum
        """
        # Calculate frustum corners
        fov_rad = np.radians(fov_degrees)
        half_fov = fov_rad / 2
        
        # Frustum corners in camera coordinate system (camera looks along +X axis)
        frustum_depth = depth * scale * 3
        frustum_width = 2 * frustum_depth * np.tan(half_fov)
        frustum_height = frustum_width  # Assuming square aspect ratio
        
        # Local frustum corners (camera coordinate system, looking along +X axis)
        local_corners = np.array([
            [0, 0, 0],  # Camera center
            [frustum_depth, -frustum_width/2, -frustum_height/2],  # Far-bottom-left
            [frustum_depth, frustum_width/2, -frustum_height/2],   # Far-bottom-right
            [frustum_depth, frustum_width/2, frustum_height/2],    # Far-top-right
            [frustum_depth, -frustum_width/2, frustum_height/2],   # Far-top-left
        ])
        
        # Transform to world coordinates
        world_corners = []
        for corner in local_corners:
            world_corner = position + rotation_matrix @ corner
            world_corners.append(world_corner)
        
        world_corners = np.array(world_corners)
        
        # Define frustum edges (lines from camera center to corners and between corners)
        frustum_lines = [
            [world_corners[0], world_corners[1]],  # Center to bottom-left
            [world_corners[0], world_corners[2]],  # Center to bottom-right
            [world_corners[0], world_corners[3]],  # Center to top-right
            [world_corners[0], world_corners[4]],  # Center to top-left
            [world_corners[1], world_corners[2]],  # Bottom edge
            [world_corners[2], world_corners[3]],  # Right edge
            [world_corners[3], world_corners[4]],  # Top edge
            [world_corners[4], world_corners[1]],  # Left edge
        ]
        
        # Log frustum lines
        rr.log(f"{entity_path}/frustum",
               rr.LineStrips3D(frustum_lines, 
                             colors=[[255, 255, 0]], 
                             radii=[scale * 0.02]))
    
    def _quaternion_to_rotation_matrix(self, quat):
        """
        Convert quaternion [x, y, z, w] to 3x3 rotation matrix
        """
        x, y, z, w = quat
        
        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm == 0:
            return np.eye(3)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        
        return R

    def log_robot_with_urdf(self, joint_angles_14, frame_index):
        """
        Log robot visualization using URDF and meshes with forward kinematics
        
        Args:
            joint_angles_14: numpy array of 14 joint angles [left_arm_7, right_arm_7]
            frame_index: current frame index for timeline
        """
        if not self.urdf_root:
            self.log_robot_simplified(joint_angles_14, frame_index)
            return
        
        try:
            # Get joint configuration
            joint_config = self.get_joint_configuration(joint_angles_14)
            if not joint_config:
                logging.warning("Could not create joint configuration")
                self.log_robot_simplified(joint_angles_14, frame_index)
                return
            
            # Compute forward kinematics
            link_transforms = self._compute_forward_kinematics(joint_config)
            
            # Log base link
            rr.log("robot_view/robot/base_link", 
                   rr.Transform3D(translation=[0, 0, 0],
                                rotation=rr.Quaternion(xyzw=[0, 0, 0, 1]),
                                relation=rr.TransformRelation.ChildFromParent))
            
            # Log base coordinate frame
            base_transform = np.eye(4)
            self._log_coordinate_frame("robot_view/robot/base_link/frame", base_transform, scale=0.15)
            
            # Log a simple test mesh first
            try:
                # Create a simple test cube at the robot base for debugging
                if TRIMESH_AVAILABLE:
                    test_cube = trimesh.creation.box([0.2, 0.2, 0.2])
                    test_cube.vertices += [0, 0, 0.1]  # move up slightly
                    rr.log("robot_view/robot/debug/test_cube",
                           rr.Mesh3D(
                               vertex_positions=test_cube.vertices.astype(np.float32),
                               triangle_indices=test_cube.faces.astype(np.uint32)
                           ))
                    logging.info("Added debug test cube at robot base")
            except Exception as e:
                logging.warning(f"Failed to create test cube: {e}")
            
            # Log base mesh if available
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
                                
                                # Apply visual origin transform
                                origin = visual_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                                visual_transform = np.eye(4)
                                visual_transform[:3, 3] = origin['xyz']
                                visual_transform[:3, :3] = self._rpy_to_rotation_matrix(origin['rpy'])
                                
                                # Transform mesh vertices
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
            
            # Log arm links and meshes
            arm_chains = {
                'left': ['zarm_l1_link', 'zarm_l2_link', 'zarm_l3_link', 'zarm_l4_link', 
                        'zarm_l5_link', 'zarm_l6_link', 'zarm_l7_link'],
                'right': ['zarm_r1_link', 'zarm_r2_link', 'zarm_r3_link', 'zarm_r4_link', 
                         'zarm_r5_link', 'zarm_r6_link', 'zarm_r7_link']
            }
            
            for arm_side, chain in arm_chains.items():
                arm_color = [255, 100, 100] if arm_side == 'left' else [100, 100, 255]
                
                for i, link_name in enumerate(chain):
                    if link_name in link_transforms:
                        transform = link_transforms[link_name]
                        translation = transform[:3, 3]
                        rotation_matrix = transform[:3, :3]
                        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
                        
                        # Add a simple visualization sphere for each link
                        rr.log(f"robot_view/robot/{arm_side}_arm/{link_name}/sphere",
                               rr.Points3D([translation], 
                                         radii=[0.03], 
                                         colors=[arm_color]))
                        
                        # Log coordinate frame for each joint/link
                        self._log_coordinate_frame(f"robot_view/robot/{arm_side}_arm/{link_name}/frame", 
                                                 transform, scale=0.15)
                        
                        # Load and transform mesh if available
                        if link_name in self.links:
                            for visual_data in self.links[link_name]['visual']:
                                if 'mesh_file' in visual_data and TRIMESH_AVAILABLE:
                                    try:
                                        mesh_file = visual_data['mesh_file']
                                        if os.path.exists(mesh_file):
                                            mesh = trimesh.load_mesh(mesh_file)
                                            scale = visual_data.get('scale', [1.0, 1.0, 1.0])
                                            if scale != [1.0, 1.0, 1.0]:
                                                mesh.vertices *= scale
                                            
                                            # Apply visual origin transform
                                            origin = visual_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                                            visual_transform = np.eye(4)
                                            visual_transform[:3, 3] = origin['xyz']
                                            visual_transform[:3, :3] = self._rpy_to_rotation_matrix(origin['rpy'])
                                            
                                            # Apply link transform and visual transform
                                            full_transform = transform @ visual_transform
                                            vertices_homo = np.hstack([mesh.vertices, np.ones((mesh.vertices.shape[0], 1))])
                                            transformed_vertices = (full_transform @ vertices_homo.T).T[:, :3]
                                            
                                            # Log mesh directly in world coordinates
                                            rr.log(f"robot_view/robot/{arm_side}_arm/{link_name}/mesh",
                                                   rr.Mesh3D(
                                                       vertex_positions=transformed_vertices.astype(np.float32),
                                                       triangle_indices=mesh.faces.astype(np.uint32)
                                                   ))
                                            
                                            # Debug: log mesh info
                                            logging.info(f"Loaded mesh for {link_name}: "
                                                       f"vertices={mesh.vertices.shape}, "
                                                       f"faces={mesh.faces.shape}, "
                                                       f"bounds={mesh.bounds}, "
                                                       f"center={translation}")
                                            break
                                    except Exception as e:
                                        logging.error(f"Failed to load mesh for {link_name}: {e}")
                                        import traceback
                                        traceback.print_exc()
            
            # Draw connections between joints
            for arm_side, chain in arm_chains.items():
                arm_color = [255, 100, 100] if arm_side == 'left' else [100, 100, 255]
                positions = []
                
                # Add base connection point
                if arm_side == 'left':
                    base_pos = np.array([-0.017499853, 0.29269999999999996, 0.4245])  # left shoulder
                else:
                    base_pos = np.array([-0.017499853, -0.29269999999999996, 0.4245])  # right shoulder
                positions.append(base_pos)
                
                # Add joint positions
                for link_name in chain:
                    if link_name in link_transforms:
                        positions.append(link_transforms[link_name][:3, 3])
                
                if len(positions) > 1:
                    positions_array = np.array(positions)
                    rr.log(f"robot_view/robot/{arm_side}_arm/skeleton",
                           rr.LineStrips3D([positions_array], 
                                         colors=[arm_color], 
                                         radii=[0.02]))
            
            # Log joint angles as scalars
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
            # Fallback to simplified visualization
            self.log_robot_simplified(joint_angles_14, frame_index)
    
    def log_robot_simplified(self, joint_angles_14, frame_index):
        """
        Fallback simplified robot visualization without URDF
        
        Args:
            joint_angles_14: numpy array of 14 joint angles [left_arm_7, right_arm_7]
            frame_index: current frame index for timeline
        """
        if len(joint_angles_14) < 14:
            return
        
        left_joints = joint_angles_14[:7]
        right_joints = joint_angles_14[7:14]
        
        # Compute simplified forward kinematics for both arms
        left_positions = self._compute_simplified_fk(left_joints, 'left')
        right_positions = self._compute_simplified_fk(right_joints, 'right')
        
        # Log robot base/torso
        torso_pos = np.array([0.0, 0.0, 0.4])
        rr.log("robot_view/robot/torso", rr.Points3D([torso_pos], radii=[0.05], colors=[[100, 100, 100]]))
        
        # Log left arm
        for i, pos in enumerate(left_positions):
            rr.log(f"robot_view/robot/left_arm/joint_{i}", 
                   rr.Points3D([pos], radii=[0.02], colors=[[255, 100, 100]]))
        
        # Log right arm
        for i, pos in enumerate(right_positions):
            rr.log(f"robot_view/robot/right_arm/joint_{i}", 
                   rr.Points3D([pos], radii=[0.02], colors=[[100, 100, 255]]))
        
        # Log arm links as line segments
        if len(left_positions) > 1:
            left_line_points = np.array(left_positions)
            rr.log("robot_view/robot/left_arm/skeleton", 
                   rr.LineStrips3D([left_line_points], colors=[[255, 100, 100]], radii=[0.01]))
        
        if len(right_positions) > 1:
            right_line_points = np.array(right_positions)
            rr.log("robot_view/robot/right_arm/skeleton", 
                   rr.LineStrips3D([right_line_points], colors=[[100, 100, 255]], radii=[0.01]))
        
        # Connect torso to shoulders
        if len(left_positions) > 0 and len(right_positions) > 0:
            torso_connections = np.array([
                [torso_pos, left_positions[0]],   # Torso to left shoulder
                [torso_pos, right_positions[0]]   # Torso to right shoulder
            ])
            rr.log("robot_view/robot/torso/connections", 
                   rr.LineStrips3D(torso_connections, colors=[[150, 150, 150]], radii=[0.015]))
        
        # Log joint angles as scalar plots
        for i, angle in enumerate(left_joints):
            rr.log(f"robot_view/robot/joint_angles/left_arm/joint_{i+1}", rr.Scalars(np.degrees(angle)))
        
        for i, angle in enumerate(right_joints):
            rr.log(f"robot_view/robot/joint_angles/right_arm/joint_{i+1}", rr.Scalars(np.degrees(angle)))
        
        # Log end-effector positions
        if len(left_positions) > 0:
            left_eef_pos = left_positions[-1]
            rr.log("robot_view/robot/left_arm/end_effector", 
                   rr.Points3D([left_eef_pos], radii=[0.03], colors=[[255, 0, 0]]))
        
        if len(right_positions) > 0:
            right_eef_pos = right_positions[-1]
            rr.log("robot_view/robot/right_arm/end_effector", 
                   rr.Points3D([right_eef_pos], radii=[0.03], colors=[[0, 0, 255]]))
    
    def _compute_simplified_fk(self, joint_angles, arm_side='left'):
        """Simplified forward kinematics computation"""
        # Base position (torso)
        if arm_side == 'left':
            base_pos = np.array([0.0, 0.15, 0.4])  # Left shoulder offset
        else:
            base_pos = np.array([0.0, -0.15, 0.4])  # Right shoulder offset
        
        positions = [base_pos]
        current_pos = base_pos.copy()
        
        # Simplified link vectors for visualization
        link_vectors = [
            np.array([0, 0, -0.1]),   # shoulder to upper arm
            np.array([0, 0, -0.15]),  # upper arm segment 1
            np.array([0, 0, -0.1]),   # upper arm segment 2
            np.array([0, 0, -0.25]),  # forearm
            np.array([0, 0, -0.05]),  # wrist segment 1
            np.array([0, 0, -0.05]),  # wrist segment 2
            np.array([0, 0, -0.1]),   # end effector
        ]
        
        # Apply simplified transformations
        for i, (angle, link_vec) in enumerate(zip(joint_angles[:7], link_vectors)):
            # Very simplified transformation - just rotate around Z axis
            cos_a = np.cos(angle * 0.5)  # Scale down rotation for visualization
            sin_a = np.sin(angle * 0.5)
            
            # Simple rotation matrix around Z axis
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            
            # Apply rotation and translation
            rotated_link = rot_matrix @ link_vec
            current_pos += rotated_link
            positions.append(current_pos.copy())
        
        return positions
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion (x, y, z, w)"""
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
    
    def _rpy_to_quaternion(self, rpy):
        """Convert RPY (roll, pitch, yaw) to quaternion (x, y, z, w)"""
        roll, pitch, yaw = rpy
        
        # Convert to quaternion
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw])  # x, y, z, w format for rerun
    
    def _log_box_mesh(self, entity_path, size):
        """Log a box mesh to rerun"""
        if TRIMESH_AVAILABLE:
            try:
                box = trimesh.creation.box(size)
                rr.log(entity_path, rr.Mesh3D(
                    vertex_positions=box.vertices,
                    triangle_indices=box.faces
                ))
            except:
                # Fallback to simple point
                rr.log(entity_path, rr.Points3D([[0, 0, 0]], radii=[max(size) / 2]))
    
    def _log_cylinder_mesh(self, entity_path, radius, length):
        """Log a cylinder mesh to rerun"""
        if TRIMESH_AVAILABLE:
            try:
                cylinder = trimesh.creation.cylinder(radius=radius, height=length)
                rr.log(entity_path, rr.Mesh3D(
                    vertex_positions=cylinder.vertices,
                    triangle_indices=cylinder.faces
                ))
            except:
                # Fallback to simple point
                rr.log(entity_path, rr.Points3D([[0, 0, 0]], radii=[radius]))
    
    def _axis_angle_to_rotation_matrix(self, axis, angle):
        """
        Convert axis-angle representation to 3x3 rotation matrix using Rodrigues' formula
        
        Args:
            axis: 3D vector representing rotation axis [x, y, z]
            angle: rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        axis = np.array(axis, dtype=float)
        # Normalize the axis
        if np.linalg.norm(axis) == 0:
            return np.eye(3)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues' rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Cross product matrix for axis
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        return R


# Global robot visualizer instance
_robot_visualizer = None

# Global head joint angles configuration
# You can adjust these values to match your robot's head pose:
# [zhead_1_joint (yaw, left-right), zhead_2_joint (pitch, up-down)]
HEAD_JOINT_ANGLES = np.array([0.0, np.radians(30)])  # 0° yaw, 30° pitch (looking slightly down)


def initialize_robot_visualizer(urdf_path: str = None):
    """Initialize the robot visualizer with optional URDF path"""
    global _robot_visualizer
    
    # Try to find URDF file automatically if not provided
    if urdf_path is None:
        # Look for URDF in common locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_urdf_paths = [
            # Primary path - user specified location
            "/home/lab/RL/src/biped_s45/urdf/biped_s45.urdf",
            # Relative paths from project root
            os.path.join(script_dir, "..", "..", "..", "src", "biped_s45", "urdf", "biped_s45.urdf"),
            "./src/biped_s45/urdf/biped_s45.urdf",
            # Fallback paths
            os.path.join(script_dir, "..", "..", "gym_hil", "assets", "biped_s45.urdf"),
            os.path.join(script_dir, "..", "..", "..", "gym_hil", "assets", "biped_s45.urdf"),
            "./gym_hil/assets/biped_s45.urdf",
            "./assets/biped_s45.urdf"
        ]
        
        for path in possible_urdf_paths:
            if os.path.exists(path):
                urdf_path = path
                logging.info(f"Found URDF file at: {path}")
                break
    
    _robot_visualizer = RobotVisualizer(urdf_path)
    logging.info(f"Robot visualizer initialized with URDF: {urdf_path}")


def log_robot_visualization(joint_angles_14, frame_index):
    """
    Log robot visualization to rerun using URDF if available
    
    Args:
        joint_angles_14: numpy array of 14 joint angles [left_arm_7, right_arm_7]
        frame_index: current frame index for timeline
    """
    global _robot_visualizer
    
    if _robot_visualizer is None:
        initialize_robot_visualizer()
    
    if _robot_visualizer.urdf_root is not None:
        _robot_visualizer.log_robot_with_urdf(joint_angles_14, frame_index)
    else:
        _robot_visualizer.log_robot_simplified(joint_angles_14, frame_index)


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def _get_camera_pose_from_urdf(camera_name, robot_visualizer, robot_base_position=np.array([0, 0, 0.4]), head_joint_angles=None):
    """
    Extract camera pose from URDF kinematic chain
    
    Args:
        camera_name: name of the camera (e.g., 'observation.images.front')
        robot_visualizer: RobotVisualizer instance with loaded URDF
        robot_base_position: base position of the robot
        head_joint_angles: optional array of head joint angles [zhead_1, zhead_2]
        
    Returns:
        tuple: (position, orientation) where orientation is a rotation matrix
    """
    if not robot_visualizer or not robot_visualizer.urdf_root:
        return _get_camera_pose_from_name_fallback(camera_name, robot_base_position)
    
    # Define mapping from camera keys to URDF link names
    camera_link_mapping = {
        'observation.images.front': 'camera',  # Head camera
        'front': 'camera',
        'camera': 'camera',
        'head': 'camera',
        'observation.images.torso': 'torso-camera',  # Torso camera
        'torso': 'torso-camera',
        'observation.images.waist': 'waist_camera',  # Waist camera
        'waist': 'waist_camera',
    }
    
    # Find the corresponding URDF link
    urdf_link_name = None
    camera_name_clean = camera_name.lower().replace('observation.images.', '').replace('.', '_')
    
    for key, link_name in camera_link_mapping.items():
        if key in camera_name.lower() or camera_name_clean == key:
            urdf_link_name = link_name
            break
    
    if urdf_link_name and urdf_link_name in robot_visualizer.links:
        # Compute forward kinematics to get camera pose
        camera_transform = _compute_camera_forward_kinematics(urdf_link_name, robot_visualizer, head_joint_angles)
        if camera_transform is not None:
            position = camera_transform[:3, 3]
            orientation = camera_transform[:3, :3]
            return position, orientation
    
    # Fallback to name-based estimation if URDF parsing fails
    return _get_camera_pose_from_name_fallback(camera_name, robot_base_position)


def _compute_camera_forward_kinematics(camera_link_name, robot_visualizer, head_joint_angles=None):
    """
    Compute forward kinematics for camera link based on URDF structure
    
    Args:
        camera_link_name: name of the camera link in URDF
        robot_visualizer: RobotVisualizer instance
        head_joint_angles: array of head joint angles [zhead_1, zhead_2] or None for default
        
    Returns:
        4x4 transformation matrix or None if failed
    """
    try:
        # Start with base_link transform
        current_transform = np.eye(4)
        
        # Set default head joint angles if not provided
        if head_joint_angles is None:
            head_joint_angles = HEAD_JOINT_ANGLES
        
        if camera_link_name == 'camera':
            # Head camera: base_link -> zhead_1_joint -> zhead_1_link -> zhead_2_joint -> zhead_2_link -> camera_joint -> camera
            
            # 1. zhead_1_joint transform (base_link to zhead_1_link)
            if 'zhead_1_joint' in robot_visualizer.joints:
                joint_data = robot_visualizer.joints['zhead_1_joint']
                origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                
                joint_transform = np.eye(4)
                joint_transform[:3, 3] = origin['xyz']  # xyz="-0.018499853 0.0 0.6014"
                joint_transform[:3, :3] = robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])
                
                # Add rotation around Z-axis using actual head joint angle
                joint_angle = head_joint_angles[0] if len(head_joint_angles) > 0 else 0.0
                if joint_data.get('type') == 'revolute':
                    joint_axis = joint_data.get('axis', [0, 0, 1])  # Z-axis
                    joint_rotation = robot_visualizer._axis_angle_to_rotation_matrix(joint_axis, joint_angle)
                    joint_transform[:3, :3] = joint_transform[:3, :3] @ joint_rotation
                
                current_transform = current_transform @ joint_transform
            
            # 2. zhead_2_joint transform (zhead_1_link to zhead_2_link)  
            if 'zhead_2_joint' in robot_visualizer.joints:
                joint_data = robot_visualizer.joints['zhead_2_joint']
                origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                
                joint_transform = np.eye(4)
                joint_transform[:3, 3] = origin['xyz']  # xyz="0 0 0"
                joint_transform[:3, :3] = robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])
                
                # Add rotation around Y-axis using actual head joint angle
                joint_angle = head_joint_angles[1] if len(head_joint_angles) > 1 else np.radians(30)
                if joint_data.get('type') == 'revolute':
                    joint_axis = joint_data.get('axis', [0, 1, 0])  # Y-axis
                    joint_rotation = robot_visualizer._axis_angle_to_rotation_matrix(joint_axis, joint_angle)
                    joint_transform[:3, :3] = joint_transform[:3, :3] @ joint_rotation
                
                current_transform = current_transform @ joint_transform
            
            # 3. camera fixed joint transform (zhead_2_link to camera)
            if 'camera' in robot_visualizer.joints:
                joint_data = robot_visualizer.joints['camera']
                origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                
                joint_transform = np.eye(4)
                joint_transform[:3, 3] = origin['xyz']  # xyz="0.0967509784707853 0.0175003248712456 0.125953265112721"
                joint_transform[:3, :3] = robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])  # rpy="0 0.488692190558413 0"
                
                current_transform = current_transform @ joint_transform
        
        elif camera_link_name == 'torso-camera':
            # Torso camera: base_link -> torso-camera (direct fixed joint)
            if 'torso-camera' in robot_visualizer.joints:
                joint_data = robot_visualizer.joints['torso-camera']
                origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                
                joint_transform = np.eye(4)
                joint_transform[:3, 3] = origin['xyz']  # xyz="0.115535332475413 0.0175000000005085 0.2883441440894"
                joint_transform[:3, :3] = robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])  # rpy="0 0.732666 0"
                
                current_transform = current_transform @ joint_transform
        
        elif camera_link_name == 'waist_camera':
            # Waist camera: base_link -> waist_camera (direct fixed joint)
            if 'waist_camera' in robot_visualizer.joints:
                joint_data = robot_visualizer.joints['waist_camera']
                origin = joint_data.get('origin', {'xyz': [0,0,0], 'rpy': [0,0,0]})
                
                joint_transform = np.eye(4)
                joint_transform[:3, 3] = origin['xyz']  # xyz="0.168717483101422 0 0.01355599662743"
                joint_transform[:3, :3] = robot_visualizer._rpy_to_rotation_matrix(origin['rpy'])  # rpy="0 0 0"
                
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


def _get_camera_pose_from_name_fallback(camera_name, robot_base_position=np.array([0, 0, 0.4])):
    """
    Fallback method to estimate camera pose based on camera name
    
    Args:
        camera_name: name of the camera
        robot_base_position: base position of the robot
        
    Returns:
        tuple: (position, orientation) where orientation is a rotation matrix
    """
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
        # Position relative to left end-effector (estimated)
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


def _visualize_camera_poses(camera_keys, batch, batch_index):
    """
    Visualize camera poses in 3D space
    
    Args:
        camera_keys: list of camera keys from dataset
        batch: current batch data
        batch_index: index within the batch
    """
    global _robot_visualizer
    
    if _robot_visualizer is None:
        initialize_robot_visualizer()
    
    # Robot base position (can be adjusted based on actual robot state)
    robot_base_pos = np.array([0, 0, 0.4])
    
    # Use global head joint angles configuration
    head_joint_angles = HEAD_JOINT_ANGLES
    
    for camera_key in camera_keys:
        try:
            # Get camera pose from URDF or fallback to name estimation
            camera_pos, camera_rot = _get_camera_pose_from_urdf(camera_key, _robot_visualizer, robot_base_pos, head_joint_angles)
            
            # Create clean camera name for visualization
            camera_name = camera_key.replace('observation.images.', '').replace('.', '_')
            
            # Log camera pose with frustum
            _robot_visualizer._log_camera_pose(
                f"robot_view/cameras/{camera_name}",
                camera_pos,
                camera_rot,
                camera_name,
                fov_degrees=60,
                scale=0.2
            )
            
            # Add image plane visualization (project image into 3D space)
            if camera_key in batch:
                _visualize_image_in_3d(f"robot_view/cameras/{camera_name}", camera_pos, camera_rot, 
                                     batch[camera_key][batch_index], scale=0.3)
                                     
        except Exception as e:
            logging.warning(f"Failed to visualize camera pose for {camera_key}: {e}")


def _visualize_image_in_3d(entity_path, camera_pos, camera_rot, image_tensor, scale=0.3):
    """
    Visualize camera image as a plane in 3D space
    
    Args:
        entity_path: rerun entity path
        camera_pos: camera position
        camera_rot: camera rotation matrix
        image_tensor: image tensor from batch
        scale: scale factor for the image plane
    """
    try:
        # Convert image to numpy
        image_np = to_hwc_uint8_numpy(image_tensor)
        h, w = image_np.shape[:2]
        
        # Create image plane in camera coordinate system
        # Assuming the image plane is at some distance in front of the camera
        plane_distance = scale * 2
        plane_width = scale
        plane_height = scale * (h / w)  # Maintain aspect ratio
        
        # Define corners of the image plane in camera coordinates
        # Camera looks along +X axis, so image plane is at +plane_distance
        local_corners = np.array([
            [plane_distance, -plane_width/2, -plane_height/2],  # Bottom-left
            [plane_distance, plane_width/2, -plane_height/2],   # Bottom-right
            [plane_distance, plane_width/2, plane_height/2],    # Top-right
            [plane_distance, -plane_width/2, plane_height/2],   # Top-left
        ])
        
        # Transform corners to world coordinates
        world_corners = []
        for corner in local_corners:
            world_corner = camera_pos + camera_rot @ corner
            world_corners.append(world_corner)
        
        world_corners = np.array(world_corners)
        
        # Create triangle indices for the plane (two triangles)
        triangle_indices = np.array([
            [0, 1, 2],  # First triangle
            [0, 2, 3]   # Second triangle
        ])
        
        # Create UV coordinates for texture mapping
        vertex_texcoords = np.array([
            [0, 1],  # Bottom-left
            [1, 1],  # Bottom-right
            [1, 0],  # Top-right
            [0, 0]   # Top-left
        ])
        
        # Log the image plane mesh with texture
        rr.log(f"{entity_path}/image_plane",
               rr.Mesh3D(
                   vertex_positions=world_corners.astype(np.float32),
                   triangle_indices=triangle_indices.astype(np.uint32),
                   vertex_texcoords=vertex_texcoords.astype(np.float32),
                   albedo_texture=image_np
               ))
               
    except Exception as e:
        logging.warning(f"Failed to create 3D image plane: {e}")


def _configure_robot_view():
    """
    Configure rerun viewer for optimal robot visualization
    """
    try:
        # Set up the main 3D view focused on the robot
        rr.log("robot_view", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
        
        # Configure a good viewing angle for robot observation
        # Position camera to look at robot from a good angle
        rr.log("robot_view/camera", 
               rr.ViewCoordinates(
                   xyz=rr.ViewCoordinates.RUB  # Right, Up, Back
               ))
        
        # Set a reasonable view distance and angle
        # This positions the view to look at the robot from front-right-above
        eye_pos = [2.0, 1.5, 1.5]  # Camera position
        target_pos = [0.0, 0.0, 0.4]  # Look at robot base
        up_vector = [0.0, 0.0, 1.0]  # Z is up
        
        # Note: Rerun 0.24.1 doesn't have direct camera control via API
        # Users will need to manually adjust the view in the GUI
        logging.info("Robot view configured. Adjust camera position in Rerun viewer for optimal robot observation.")
        logging.info(f"Suggested view: Eye={eye_pos}, Target={target_pos}, Up={up_vector}")
        
    except Exception as e:
        logging.warning(f"Failed to configure robot view: {e}")


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()
    
    # Configure robot-centric view settings
    _configure_robot_view()

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["frame_index"][i].item())
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            # display each camera image
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))
            
            # Visualize camera poses in 3D space
            _visualize_camera_poses(dataset.meta.camera_keys, batch, i)

            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalars(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))
                
                # Extract and visualize robot arm joints
                state_data = batch["observation.state"][i].numpy()
                
                # Try to extract arm joints based on different possible state configurations
                arm_joints = None
                if len(state_data) >= 28:
                    # For WBC mode: arm joints are typically at indices 14-27 (14 joints total)
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
                
                if arm_joints is not None and len(arm_joints) == 14:
                    try:
                        log_robot_visualization(arm_joints, batch["frame_index"][i].item())
                    except Exception as e:
                        logging.warning(f"Failed to visualize robot at frame {batch['frame_index'][i].item()}: {e}")
                else:
                    # Log a message about state structure for debugging
                    if i == 0:  # Only log once per batch to avoid spam
                        logging.info(f"State shape: {state_data.shape}, cannot extract 14 arm joints for robot visualization")

            if "next.done" in batch:
                rr.log("next.done", rr.Scalars(batch["next.done"][i].item()))

            if "next.reward" in batch:
                rr.log("next.reward", rr.Scalars(batch["next.reward"][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
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
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    parser.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help=(
            "Path to URDF file for accurate robot visualization with meshes. "
            "If not provided, the script will search automatically in common locations. "
            "Requires urdfpy and trimesh packages for full functionality."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")
    urdf_path = kwargs.pop("urdf_path")

    # Initialize robot visualizer with URDF path if provided
    if urdf_path:
        initialize_robot_visualizer(urdf_path)
        logging.info(f"Initialized robot visualizer with URDF: {urdf_path}")

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, **kwargs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Robot System Usage Example

This script demonstrates how to use the modular robot system for
loading and processing robotics datasets.
"""

import numpy as np
from pathlib import Path
import sys

# Add lerobot to path
sys.path.append(str(Path(__file__).parent.parent))

from lerobot.common.robot_system import (
    Robotic, 
    create_kuavo_robot, 
    create_dual_arm_robot,
    CameraIntrinsics,
    CameraPose,
    JointLimits,
    JointAxis,
    ModelMode,
    TorsoConfig
)


def basic_robot_creation_example():
    """Example 1: Basic robot creation and configuration"""
    print("=== Basic Robot Creation Example ===")
    
    # Create a basic dual-arm robot
    robot = create_dual_arm_robot(name="MyRobot")
    
    # Add a depth camera
    robot.add_camera('depth_front', camera_type='DEPTH', depth_scale=1000.0)
    
    # Set some joint positions
    left_positions = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    right_positions = np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7])
    
    robot.set_arm_positions('left', left_positions)
    robot.set_arm_positions('right', right_positions)
    
    # Get robot info
    info = robot.get_robot_info()
    print(f"Robot name: {info['name']}")
    print(f"Model mode: {info['model_mode']}")
    print(f"Cameras: {list(info['cameras'].keys())}")
    print(f"Joint chains: {info['joints']['joint_chains']}")
    print()


def kuavo_robot_example():
    """Example 2: Kuavo robot configuration"""
    print("=== Kuavo Robot Example ===")
    
    # Create Kuavo robot (will auto-detect URDF)
    robot = create_kuavo_robot()
    
    # Configure camera with intrinsics
    intrinsics = CameraIntrinsics(
        fx=500.0, fy=500.0, 
        cx=320.0, cy=240.0,
        width=640, height=480
    )
    
    # Configure camera pose (front camera)
    position = np.array([0.1, 0.0, 0.6])  # In front of robot head
    orientation = np.array([
        [-1, 0, 0],  # X points back (camera looking at robot)
        [0, -1, 0],  # Y points left
        [0, 0, 1]    # Z points up
    ])
    pose = CameraPose(position=position, orientation=orientation)
    
    # Update front camera with intrinsics and pose
    front_camera = robot.get_camera('front')
    if front_camera:
        front_camera.update_intrinsics(intrinsics)
        front_camera.update_pose(pose)
    
    # Compute forward kinematics
    left_fk = robot.compute_forward_kinematics('left')
    print(f"Left arm forward kinematics computed for {len(left_fk)} joints")
    
    # Check configuration
    issues = robot.validate_configuration()
    if issues:
        print(f"Configuration issues: {issues}")
    else:
        print("Robot configuration is valid!")
    print()


def dataset_integration_example():
    """Example 3: Dataset integration"""
    print("=== Dataset Integration Example ===")
    
    # Create robot
    robot = create_kuavo_robot()
    
    # Create dataset wrapper (this would work with actual dataset)
    try:
        wrapper = robot.create_lerobot_dataset_wrapper(
            repo_id="your-dataset-repo-id",
            arm_joint_indices=(6, 20),  # Indices 6-19 for 14 arm joints
            robot_base_position_indices=(0, 3),  # First 3 elements for robot position
            target_position_key="target_position"
        )
        print("Dataset wrapper created successfully")
        
        # This would load actual data:
        # dataset = wrapper.load_dataset()
        # episode_data = robot.get_episode_data(0)
        
    except Exception as e:
        print(f"Dataset integration example (expected to fail without real dataset): {e}")
    
    print()


def advanced_configuration_example():
    """Example 4: Advanced robot configuration"""
    print("=== Advanced Configuration Example ===")
    
    # Create robot with custom torso configuration
    torso_config = TorsoConfig(
        base_link_position=(0.0, 0.0, 0.0),
        torso_position=(0.0, 0.0, 0.45),  # Slightly higher torso
        torso_orientation=(0.0, np.radians(5), 0.0),  # 5 degree pitch
        model_mode=ModelMode.FLOATING_BASE
    )
    
    robot = Robotic(
        name="AdvancedRobot",
        torso_config=torso_config,
        urdf_path="/path/to/robot.urdf"
    )
    
    # Add joints with specific configurations
    joint_configs = {}
    for i in range(1, 8):
        # Left arm joints
        left_joint_name = f'zarm_l{i}_joint'
        joint_configs[left_joint_name] = {
            'limits': JointLimits(lower=-np.pi, upper=np.pi, velocity=2.0, effort=100.0),
            'axis': JointAxis(0, 0, 1)  # Z-axis rotation
        }
        
        # Right arm joints
        right_joint_name = f'zarm_r{i}_joint'
        joint_configs[right_joint_name] = {
            'limits': JointLimits(lower=-np.pi, upper=np.pi, velocity=2.0, effort=100.0),
            'axis': JointAxis(0, 0, 1)  # Z-axis rotation
        }
    
    # Setup arms with configurations
    left_joints = [f'zarm_l{i}_joint' for i in range(1, 8)]
    right_joints = [f'zarm_r{i}_joint' for i in range(1, 8)]
    
    robot.setup_dual_arm_joints(left_joints, right_joints, joint_configs)
    
    # Add multiple cameras with different configurations
    # RGB camera with high resolution
    robot.add_camera(
        'high_res_front', 
        camera_type='RGB',
        intrinsics=CameraIntrinsics(fx=800, fy=800, cx=512, cy=384, width=1024, height=768)
    )
    
    # Depth camera for perception
    robot.add_camera(
        'perception_depth',
        camera_type='DEPTH',
        depth_scale=1000.0,
        intrinsics=CameraIntrinsics(fx=400, fy=400, cx=320, cy=240, width=640, height=480)
    )
    
    print(f"Advanced robot created with {len(robot.get_all_cameras())} cameras")
    print(f"Joint system has {len(robot.joint_system.joints)} joints")
    print(f"Torso position: {robot.torso_config.torso_position}")
    print()


def data_processing_example():
    """Example 5: Data processing workflow"""
    print("=== Data Processing Workflow Example ===")
    
    robot = create_kuavo_robot()
    
    # Simulate loading data from dataset
    simulated_data_item = {
        'observation.state': np.random.randn(28),  # 28-dimensional state
        'observation.images.front': np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),
        'observation.images.torso': np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),
        'action': np.random.randn(14),  # 14-dimensional action
        'timestamp': 1234567890.0
    }
    
    # Create a mock dataset wrapper for this example
    class MockDatasetWrapper:
        def extract_robot_joint_positions(self, data_item):
            return data_item['observation.state'][6:20]  # Extract arm joints
        
        def extract_robot_world_position(self, data_item):
            return data_item['observation.state'][:3]  # First 3 elements
        
        def extract_target_world_position(self, data_item):
            return np.array([1.0, 0.5, 0.3])  # Mock target position
        
        def extract_camera_images(self, data_item):
            images = {}
            for key, value in data_item.items():
                if 'observation.images' in key:
                    camera_name = key.replace('observation.images.', '')
                    images[camera_name] = value
            return images
        
        def get_dataset_info(self):
            return {'mock': True}
    
    # Set mock wrapper
    robot.dataset_wrapper = MockDatasetWrapper()
    
    # Process the data
    robot.load_state_from_data(simulated_data_item)
    
    # Extract images
    images = robot.extract_camera_images(simulated_data_item)
    
    print(f"Processed data item:")
    print(f"  - Joint positions shape: {robot.get_dual_arm_positions().shape}")
    print(f"  - World position: {robot.current_state['world_position']}")
    print(f"  - Target position: {robot.current_state['target_position']}")
    print(f"  - Images extracted: {list(images.keys())}")
    print(f"  - Timestamp: {robot.current_state['timestamp']}")
    print()


def main():
    """Run all examples"""
    print("Robot System Usage Examples")
    print("=" * 50)
    
    basic_robot_creation_example()
    kuavo_robot_example()
    dataset_integration_example()
    advanced_configuration_example()
    data_processing_example()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()

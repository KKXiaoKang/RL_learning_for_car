#!/usr/bin/env python3
"""
Test script for the new current robot pose integration feature.
This script tests the functionality without requiring the actual robot hardware.
"""

import json
import numpy as np
from robotic_bezier_action_record_tool import BezierTrajectoryGenerator

def test_without_ros():
    """Test the tool without ROS (basic functionality)."""
    print("=" * 60)
    print("Testing Bézier Tool WITHOUT ROS")
    print("=" * 60)
    
    try:
        # Initialize without ROS
        generator = BezierTrajectoryGenerator(
            enable_ros=False, 
            debug=True, 
            use_current_robot_pose=False
        )
        
        print("✓ Generator initialized successfully without ROS")
        
        # Test basic trajectory generation
        trajectory_data = generator.get_trajectory_data()
        left_positions = trajectory_data['left_hand']['positions']
        right_positions = trajectory_data['right_hand']['positions']
        
        print(f"✓ Generated trajectories:")
        print(f"  Left hand points: {len(left_positions)}")
        print(f"  Right hand points: {len(right_positions)}")
        
        # Test action calculation
        left_actions, right_actions = generator.calculate_trajectory_actions()
        print(f"✓ Calculated actions:")
        print(f"  Left actions shape: {left_actions.shape}")
        print(f"  Right actions shape: {right_actions.shape}")
        print(f"  Actions in range [-1,1]: {np.all(np.abs(np.concatenate([left_actions, right_actions])) <= 1.0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during non-ROS testing: {e}")
        return False

def test_keypoint_structure():
    """Test that the key-point JSON structure is valid for the new functionality."""
    print("\n" + "=" * 60)
    print("Testing Key-Point Structure")
    print("=" * 60)
    
    try:
        # Load and examine the key-point structure
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        keypoint_file = os.path.join(script_dir, "key_point.json")
        
        with open(keypoint_file, 'r') as f:
            data = json.load(f)
        
        key_points = data['key_points']
        keyframes = key_points['keyframes']
        
        print(f"✓ Loaded key-points file with {len(keyframes)} keyframes")
        
        # Check for frame_id 0
        initial_frame = None
        for frame in keyframes:
            if frame.get('frame_id', -1) == 0:
                initial_frame = frame
                break
        
        if initial_frame:
            print("✓ Found initial keyframe (frame_id: 0)")
            print(f"  Left hand position: {initial_frame['left_hand']['position']}")
            print(f"  Left hand quaternion: {initial_frame['left_hand']['quaternion']}")
            print(f"  Right hand position: {initial_frame['right_hand']['position']}")
            print(f"  Right hand quaternion: {initial_frame['right_hand']['quaternion']}")
        else:
            print("✗ No initial keyframe (frame_id: 0) found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during key-point structure testing: {e}")
        return False

def simulate_ros_pose_update():
    """Simulate what would happen when updating the initial pose from ROS data."""
    print("\n" + "=" * 60)
    print("Simulating ROS Pose Update")
    print("=" * 60)
    
    try:
        # Create a generator instance
        generator = BezierTrajectoryGenerator(
            enable_ros=False, 
            debug=True, 
            use_current_robot_pose=False
        )
        
        # Get original key-points
        original_keypoints = generator.key_points.copy()
        original_frame_0 = None
        for frame in original_keypoints['keyframes']:
            if frame.get('frame_id', -1) == 0:
                original_frame_0 = frame.copy()
                break
        
        if not original_frame_0:
            print("✗ No frame_id 0 found in original key-points")
            return False
        
        print("Original frame_id 0 poses:")
        print(f"  Left: {original_frame_0['left_hand']['position']}")
        print(f"  Right: {original_frame_0['right_hand']['position']}")
        
        # Simulate new robot poses (as if received from ROS topics)
        simulated_left_pose = [0.4, 0.35, 0.1]  # Different from original
        simulated_left_quat = [0.0, -0.70711, 0.0, 0.70711]
        simulated_right_pose = [0.4, -0.35, 0.1]  # Different from original
        simulated_right_quat = [0.0, -0.70711, 0.0, 0.70711]
        
        # Manually update frame_id 0 (simulating what would happen with ROS)
        for frame in generator.key_points['keyframes']:
            if frame.get('frame_id', -1) == 0:
                frame['left_hand']['position'] = simulated_left_pose
                frame['left_hand']['quaternion'] = simulated_left_quat
                frame['right_hand']['position'] = simulated_right_pose
                frame['right_hand']['quaternion'] = simulated_right_quat
                break
        
        print("\nSimulated updated frame_id 0 poses:")
        print(f"  Left: {simulated_left_pose}")
        print(f"  Right: {simulated_right_pose}")
        
        # Generate trajectory with updated initial pose
        trajectory_data = generator.get_trajectory_data()
        updated_left_positions = trajectory_data['left_hand']['positions']
        updated_right_positions = trajectory_data['right_hand']['positions']
        
        # Check that the first point of the trajectory matches the updated pose
        first_left_pos = updated_left_positions[0]
        first_right_pos = updated_right_positions[0]
        
        left_match = np.allclose(first_left_pos, simulated_left_pose, atol=1e-6)
        right_match = np.allclose(first_right_pos, simulated_right_pose, atol=1e-6)
        
        if left_match and right_match:
            print("✓ Trajectory correctly starts from updated initial poses")
            print(f"  First trajectory point (left): {first_left_pos}")
            print(f"  First trajectory point (right): {first_right_pos}")
        else:
            print("✗ Trajectory does not start from updated initial poses")
            print(f"  Expected left: {simulated_left_pose}, Got: {first_left_pos}")
            print(f"  Expected right: {simulated_right_pose}, Got: {first_right_pos}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during simulation: {e}")
        return False

def main():
    """Run all tests."""
    print("Robotic Bézier Tool - Current Pose Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("Basic Functionality", test_without_ros()))
    test_results.append(("Key-Point Structure", test_keypoint_structure()))
    test_results.append(("Pose Update Simulation", simulate_ros_pose_update()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25}: {status}")
        if not result:
            all_passed = False
    
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nThe new current robot pose integration feature is ready for use!")
        print("\nUsage examples:")
        print("  python3 robotic_bezier_action_record_tool.py --mode actions --use-current-pose --debug")
        print("  python3 robotic_bezier_action_record_tool.py --mode play_actions --use-current-pose --rate 10.0")
    
    return all_passed

if __name__ == "__main__":
    main()

# Robotic Bézier Action Record Tool

A comprehensive tool for generating smooth Bézier trajectories for robotic end-effector motion and publishing them as ROS messages.

## Features

- **Bézier Interpolation**: Generate smooth cubic Bézier curves between key-points
- **Dual-Arm Support**: Simultaneous trajectory generation for left and right hands
- **Continuity Assurance**: C1 continuity between consecutive trajectory segments
- **3D Visualization**: Interactive plots showing trajectories and key-points
- **ROS Integration**: Publish trajectories as `twoArmHandPoseCmd` messages
- **Elbow Configuration**: Specify elbow joint positions for improved IK solving
- **Multiple Modes**: Visualization, real-time playback, keyframe publishing, and data export

## Files

- `robotic_bezier_action_record_tool.py`: Main implementation
- `key_point.json`: Configuration file with key-points and parameters
- `README.md`: This documentation file

## Configuration

### Key-Points Format (`key_point.json`)

```json
{
    "key_points": {
        "frame": "base_link",
        "description": "Cartesian key-points for left and right hand end-effectors with elbow configurations",
        "elbow_positions": {
            "left_elbow": [-0.0178026345146559, 0.4004180715613648, 0.17417275957965042],
            "right_elbow": [-0.0178026345146559, -0.4004180715613648, 0.17417275957965042]
        },
        "ik_parameters": {
            "ik_solve_param": 1,
            "use_custom_ik_param": false,
            "joint_angles_as_q0": false
        },
        "keyframes": [
            {
                "frame_id": 0,
                "left_hand": {
                    "position": [x, y, z],
                    "quaternion": [x, y, z, w]
                },
                "right_hand": {
                    "position": [x, y, z],
                    "quaternion": [x, y, z, w]
                }
            }
        ]
    }
}
```

## Usage

### Command Line Interface

#### 1. Visualization Mode (Default)
```bash
python robotic_bezier_action_record_tool.py
```

Generate and display 3D visualization of trajectories:
```bash
python robotic_bezier_action_record_tool.py --mode visualize
```

Save plot without displaying:
```bash
python robotic_bezier_action_record_tool.py --mode visualize --no-plot
```

#### 2. ROS Trajectory Playback
```bash
# Play trajectory at 10 Hz
python robotic_bezier_action_record_tool.py --mode play --rate 10.0

# Play trajectory in continuous loop at 5 Hz
python robotic_bezier_action_record_tool.py --mode play --rate 5.0 --loop
```

#### 3. Keyframe Publishing
```bash
# Publish keyframe 0
python robotic_bezier_action_record_tool.py --mode keyframe --frame-id 0

# Publish keyframe 1
python robotic_bezier_action_record_tool.py --mode keyframe --frame-id 1
```

#### 4. Data Export
```bash
python robotic_bezier_action_record_tool.py --mode export
```

#### 5. Demo Mode
```bash
python robotic_bezier_action_record_tool.py demo
```

### Programmatic Usage

```python
from robotic_bezier_action_record_tool import BezierTrajectoryGenerator

# Initialize generator
generator = BezierTrajectoryGenerator()

# Get trajectory data
trajectory_data = generator.get_trajectory_data()
left_positions = trajectory_data['left_hand']['positions']
left_orientations = trajectory_data['left_hand']['orientations']

# Visualize trajectories
generator.visualize_trajectories()

# ROS publishing (requires ROS environment)
if generator.enable_ros:
    # Publish single keyframe
    generator.publish_keyframe(0)
    
    # Play full trajectory
    generator.play_trajectory(playback_rate=10.0, loop=False)
```

## ROS Integration

### Message Type
The tool publishes to `/ik/two_arm_hand_pose_cmd` using `twoArmHandPoseCmd` messages.

### Message Structure
```
twoArmHandPoseCmd:
  ik_param: 1
  use_custom_ik_param: false
  joint_angles_as_q0: false
  hand_poses:
    left_pose:
      pos_xyz: [x, y, z]
      quat_xyzw: [x, y, z, w]
      elbow_pos_xyz: [x, y, z]
      joint_angles: [0, 0, 0, 0, 0, 0, 0]
    right_pose:
      pos_xyz: [x, y, z]
      quat_xyzw: [x, y, z, w]
      elbow_pos_xyz: [x, y, z]
      joint_angles: [0, 0, 0, 0, 0, 0, 0]
```

### Subscribed Topics
- `/ik/result` (twoArmHandPose): IK solver results (optional callback)

## Dependencies

### Required
- `numpy`
- `scipy`
- `matplotlib`
- `json`

### Optional (for ROS functionality)
- `rospy`
- `kuavo_arm_tele.msg`

## Mathematical Background

### Cubic Bézier Formula
The tool uses cubic Bézier curves with the formula:

**B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃**

Where:
- P₀, P₃: Start and end points
- P₁, P₂: Control points
- t: Parameter from 0 to 1

### Continuity Assurance
The algorithm ensures C1 continuity by:
1. Computing direction vectors between consecutive key-points
2. Setting control points based on these directions
3. Using smoothness factor to control curve tension

### Orientation Interpolation
Quaternion orientations are interpolated using SLERP (Spherical Linear Interpolation) for smooth rotational motion.

## Example Output

```
Robotic Bézier Action Record Tool
========================================
ROS node initialized successfully
Generating Bézier trajectories for robotic end-effectors...
Plot saved as 'bezier_trajectories.png'

Trajectory Statistics:
Left hand trajectory points: 99
Right hand trajectory points: 99
Total trajectory length (left): 0.234 m
Total trajectory length (right): 0.234 m
```

## Troubleshooting

### ROS Issues
- Ensure ROS environment is properly sourced
- Check if `kuavo_arm_tele` package is available
- Verify topic names match your system configuration

### Trajectory Issues
- Adjust smoothness_factor in `generate_control_points()` for different curve behaviors
- Check key-point positions are within robot workspace
- Verify quaternion orientations are normalized

### Visualization Issues
- Install matplotlib with 3D support: `pip install matplotlib`
- Use `--no-plot` flag if display is not available
- Check saved plot files in working directory

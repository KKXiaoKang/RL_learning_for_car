# VR Intervention System for RLKuavo Environment

This document explains the VR intervention system that allows human operators to take control of the RLKuavo robot using Meta Quest3 VR devices during reinforcement learning training or evaluation.

## Architecture Overview

The VR intervention system consists of several components working together:

```
Quest3 VR Device
       ↓
monitor_quest3.py (VR data capture)
       ↓
quest3_node_incremental.py (VR → Robot commands)
       ↓
[/cmd_vel, /mm/kuavo_arm_traj] (ROS topics)
       ↓
RLKuavoMetaVRWrapper (Intervention detection & action conversion)
       ↓
RLKuavoGymEnv (RL environment)
```

### Key Components

1. **monitor_quest3.py**: Captures VR device data and publishes joystick/pose information
2. **quest3_node_incremental.py**: Converts VR data to robot control commands
3. **RLKuavoMetaVRWrapper**: Gym wrapper that detects intervention and manages control handoff
4. **RLKuavoGymEnv**: The base RL environment with modified action publishing

## How It Works

### Normal Operation (No Intervention)
1. RL policy generates actions
2. Environment publishes actions to `/cmd_vel` and `/kuavo_arm_traj`
3. Robot executes the actions

### VR Intervention Mode
1. VR operator grips right controller (≥ threshold)
2. `Quest3Controller` detects intervention signal
3. `RLKuavoMetaVRWrapper` switches to intervention mode:
   - Environment **stops** publishing actions
   - VR system takes control via existing ROS topics
   - Wrapper **listens** to VR-generated `/cmd_vel` and `/mm/kuavo_arm_traj`
   - Converts these ROS messages back to environment action format
   - Maintains RL loop for observation and reward calculation

### Key Features

- **Seamless handoff**: VR can take control at any time during episode
- **Action consistency**: VR commands are converted to same format as RL actions
- **No interference**: Environment doesn't publish conflicting commands during intervention
- **Full observability**: RL system continues to observe and compute rewards during VR control

## Setup and Usage

### Prerequisites

1. **ROS Environment**: Properly configured ROS workspace
2. **Quest3 VR Setup**: Calibrated and connected to network
3. **Isaac Lab**: Running with RLKuavo robot simulation

### Required ROS Nodes

Start these nodes **before** running the RL environment:

```bash
# Terminal 1: VR data capture
python monitor_quest3.py

# Terminal 2: VR → Robot command conversion  
python quest3_node_incremental.py --incremental_control 1
```

### Environment Usage

```python
import gym_hil
from gym_hil.wrappers.factory import make_rl_kuavo_meta_vr_env

# Create environment with VR intervention
env = make_rl_kuavo_meta_vr_env(
    intervention_threshold=1.0,    # Right grip threshold
    rerecord_threshold=1.0,        # Left grip threshold  
    auto_reset=False
)

# Normal RL loop - VR intervention works automatically
obs, info = env.reset()
while True:
    action = policy(obs)  # Your RL policy
    obs, reward, done, truncated, info = env.step(action)
    
    # Check intervention status
    if info["is_intervention"]:
        print("VR intervention active!")
        print(f"VR data available: {info['vr_cmd_vel_available']}")
    
    if done:
        break
```

## VR Control Mapping

### Intervention Control
- **Right Grip ≥ 1.0**: Activate intervention mode
- **Left Grip ≥ 1.0**: Trigger episode rerecord

### Episode Control  
- **Left Second Button (Y)**: Mark episode as successful
- **Left First Button (X)**: Mark episode as failed

### Robot Control (during intervention)
- **VR Hand Movements**: Control robot arm end-effectors
- **VR Controller Joysticks**: Generate base velocity commands
- **All VR inputs**: Processed through `quest3_node_incremental.py`

## Technical Details

### Action Space Conversion

The wrapper converts VR-generated ROS messages to environment actions:

```python
# CMD_VEL → Velocity action (4D or 6D)
vel_action = [cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.linear.z, cmd_vel.angular.z]

# ARM_TRAJ → Joint action (14D)  
arm_positions_rad = np.deg2rad(arm_traj.position[:14])
arm_action = (arm_positions_rad - joint_centers) / joint_scales

# Combined action
action = np.concatenate([vel_action, arm_action])
```

### Thread Safety

- VR data callbacks use locks to ensure thread-safe access
- Action conversion is atomic to prevent race conditions

### ROS Topic Monitoring

The wrapper subscribes to:
- `/cmd_vel`: Base velocity commands from VR
- `/mm/kuavo_arm_traj`: Arm joint commands from VR  
- `/quest_joystick_data`: VR controller state (via Quest3Controller)

## Debugging and Monitoring

### Info Dictionary Keys

During intervention, the `info` dict includes:
- `is_intervention`: Boolean intervention status
- `vr_cmd_vel_available`: Whether VR velocity data is available  
- `vr_arm_traj_available`: Whether VR arm data is available
- `vr_grip_values`: Tuple of (left_grip, right_grip) values
- `action_intervention`: The actual action sent to environment

### Common Issues

1. **No VR data available**: 
   - Check that `monitor_quest3.py` and `quest3_node_incremental.py` are running
   - Verify Quest3 is connected and broadcasting

2. **Actions not converted properly**:
   - Check ROS topic publication rates
   - Verify environment has `vel_action_scale` and joint parameters

3. **Intervention not detected**:
   - Check grip threshold values
   - Verify Quest3Controller is receiving joystick data

## Example Usage

See `examples/vr_intervention_example.py` for a complete working example.

## Architecture Benefits

1. **Modular Design**: VR system is independent of RL environment
2. **Minimal Coupling**: Only intervention detection couples VR and RL systems  
3. **Reusable**: Same VR setup works with any RLKuavo-based environment
4. **Debuggable**: Clear separation of concerns makes issues easier to trace
5. **Extensible**: Easy to add new VR input modalities or robot behaviors

This design allows seamless human intervention while maintaining the full RL training/evaluation pipeline. 
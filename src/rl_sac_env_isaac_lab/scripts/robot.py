# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
"""
    Configuration for the Kuavo Robotics legs and arms.
"""
"""
    # isaac lab 1.0 API接口
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.actuators import ImplicitActuatorCfg
    from omni.isaac.lab.assets.articulation import ArticulationCfg
    import os
"""
"""
    effort_limit - 组中关节的力/扭矩限制。
    velocity_limit - 组中关节的速度限制。
    effort_limit_sim - 组中关节应用于模拟物理解算器的努力极限。
    velocity_limit_sim - 应用于模拟物理解算器的组中关节的速度限制。
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg # 隐式执行器配置
from isaaclab.assets import ArticulationCfg # 多连体配置
import os
import rospkg
import rospy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
usd_dir_path = os.path.join(BASE_DIR, "usd/")

# 添加rospack获取USD文件路径
def get_robot_usd_path():
    rospack = rospkg.RosPack()
    try:
        rl_sac_env_path = rospack.get_path('rl_sac_env_isaac_lab')
        usd_path = os.path.join(rl_sac_env_path, 'Assets/dingo.usd')
        if not os.path.exists(usd_path):
            rospy.logwarn(f"USD file not found at {usd_path}, falling back to default path")
            usd_path = os.path.join(usd_dir_path, "dingo.usd")
    except rospkg.ResourceNotFound:
        rospy.logwarn("rl_sac_env_isaac_lab package not found, falling back to default path")
        usd_path = os.path.join(usd_dir_path, "dingo.usd")
    
    return usd_path

# 使用函数获取USD文件路径
robot_usd = get_robot_usd_path()
rospy.loginfo(f"Loading USD file from: {robot_usd}")

## 
# Configuration
##

# Global parameters for PD control
USE_TORQUE_CONTROL = True  # 设置为True时使用全力矩模式，False时使用PD控制

# Wheel parameters
WHEEL_STIFFNESS = 0.0 if USE_TORQUE_CONTROL else 60.0
WHEEL_DAMPING = 10.0 if USE_TORQUE_CONTROL else 10.0

KINOVA_ROBOTIQ = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=robot_usd,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            fix_root_link = False
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),  # Adjusted for dingo's height
        rot=(1.0, 0.0, 0.0, 0.0), # w x y z
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={
            # Wheel joints
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit=1e30,
            velocity_limit=1e30,
            stiffness=WHEEL_STIFFNESS,
            damping=WHEEL_DAMPING,
        ),
    },
)

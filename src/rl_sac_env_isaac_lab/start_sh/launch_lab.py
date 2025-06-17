import subprocess
import os
import rospy
import rospkg

rospy.init_node("launch_isaac_sim")

# 新增：获取当前用户名
username = os.getenv('USER')  # 适用于Linux/macOS
# 如果Windows系统使用：os.getenv('USERNAME')

# 动态构建目标目录路径
target_dir = os.path.expanduser(f"~{username}/IsaacLab") if username else os.path.expanduser("~/IsaacLab")

USE_CAMERA_FLAG_BOOL = rospy.get_param('use_camera_ros_topic_flag', True)
rospy.loginfo(f"launch_lab.py -- USE_CAMERA_FLAG_BOOL: {USE_CAMERA_FLAG_BOOL}")

USE_GPU_FLAG_BOOL = rospy.get_param('use_gpu_flag', True)
rospy.loginfo(f"launch_lab.py -- USE_GPU_FLAG_BOOL: {USE_GPU_FLAG_BOOL}")

USE_RENDER_RVIZ_FLAG_BOOL = rospy.get_param('only_render_rviz_flag', True)
rospy.loginfo(f"launch_lab.py -- USE_RENDER_RVIZ_FLAG_BOOL: {USE_RENDER_RVIZ_FLAG_BOOL}")

def launch_isaac_sim():
    global USE_CAMERA_FLAG_BOOL
    try:
        # 初始化rospkg
        rospack = rospkg.RosPack()
        
        # 获取当前功能包路径（controller_tcp）
        controller_tcp_path = rospack.get_path('rl_sac_env_isaac_lab')  # 假设功能包名为isaac_sim
        
        # 构建上级目录路径
        isaac_sim_root = os.path.dirname(os.path.dirname(controller_tcp_path))
        robot_mpc_path = os.path.join(isaac_sim_root, "src", "rl_sac_env_isaac_lab", "scripts")

        print(" robot_mpc_path :", robot_mpc_path)
        print(" controller_tcp_path :", controller_tcp_path)
        print(" isaac_sim_root :", isaac_sim_root)
        
        # 验证路径存在
        if not os.path.exists(robot_mpc_path):
            raise FileNotFoundError(f"kuavo_robot_mpc directory not found at: {robot_mpc_path}")

        os.chdir(target_dir)
        print(f"Changed directory to: {target_dir}")
        
        # 动态构建项目路径
        project_path = robot_mpc_path  # 使用动态获取的路径
        
        if USE_GPU_FLAG_BOOL:
            base_script = os.path.join(project_path, "env.py")
            command = f"./isaaclab.sh -p {base_script}"
            if USE_CAMERA_FLAG_BOOL:
                command += " --enable_cameras"        
        print(f"Executing command: {command}")
        
        # 运行命令
        process = subprocess.Popen(
            command, 
            shell=True,
            executable='/bin/bash',
            env=os.environ.copy()
        )
        print("Started Isaac Sim application")
        
        # 等待进程完成
        process.wait()
        
    except FileNotFoundError:
        print(f"Error: Directory {target_dir} not found")
    except subprocess.CalledProcessError as e:
        print(f"Error running Isaac Sim: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    launch_isaac_sim() 
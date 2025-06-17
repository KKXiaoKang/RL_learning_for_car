import rospy
from sensor_msgs.msg import Joy
from kuavo_msgs.msg import jointCmd

class JoyListener:
    """
        axes:
        [0] left/right
        [1] up/down
    """
    def __init__(self):
        # 从参数服务器获取最大速度值
        self.max_linear_vel = rospy.get_param("~max_linear_vel", 100.0)  # 默认5 m/s
        self.max_angular_vel = rospy.get_param("~max_angular_vel", 50.0) # 默认2 rad/s
        
        # 创建命令发布器
        self.cmd_pub = rospy.Publisher("/joint_cmd", jointCmd, queue_size=1)
        
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)

    def joy_callback(self, msg):
        # 使用混合公式：v = (forward + turn)/2
        forward = msg.axes[1] * self.max_linear_vel
        turn = msg.axes[2] * self.max_angular_vel
        
        # 混合计算保证速度不超过最大值
        left_wheel_vel = (forward - turn) 
        right_wheel_vel = (forward + turn)
        
        # 归一化处理
        max_speed = max(abs(left_wheel_vel), abs(right_wheel_vel))
        if max_speed > self.max_linear_vel:
            scale = self.max_linear_vel / max_speed
            left_wheel_vel *= scale
            right_wheel_vel *= scale

        # 创建并填充jointCmd消息
        cmd_msg = jointCmd()
        cmd_msg.tau = [left_wheel_vel, right_wheel_vel]
        self.cmd_pub.publish(cmd_msg)

if __name__ == "__main__":
    rospy.init_node("joy_listener")
    joy_listener = JoyListener()
    rospy.spin()
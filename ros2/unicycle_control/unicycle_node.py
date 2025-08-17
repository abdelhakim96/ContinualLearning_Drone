import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import sys
import os
import math

# Add the parent directory to the Python path to allow importing the unicycle module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from python.kinematics.unicycle_kinematics import Unicycle

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cy * cp * cr + sy * sp * sr
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr
    return q

class UnicycleROS2Wrapper(Node):
    def __init__(self):
        super().__init__('unicycle_controller')

        # Parameters
        self.declare_parameter('dt', 0.01)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('x_init', 0.0)
        self.declare_parameter('y_init', 0.0)
        self.declare_parameter('yaw_init', 0.0)

        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        x_init = self.get_parameter('x_init').get_parameter_value().double_value
        y_init = self.get_parameter('y_init').get_parameter_value().double_value
        yaw_init = self.get_parameter('yaw_init').get_parameter_value().double_value

        # Unicycle instance
        self.unicycle = Unicycle(self.dt, init=[x_init, y_init, yaw_init])
        self.command = [0.0, 0.0]

        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.odom_broadcaster = TransformBroadcaster(self)

        # Main loop
        self.timer = self.create_timer(self.dt, self.run)

    def cmd_vel_callback(self, msg):
        self.command[0] = msg.linear.x
        self.command[1] = msg.angular.z

    def run(self):
        current_time = self.get_clock().now().to_msg()

        # Simulate one step
        pose, _ = self.unicycle.simulate(self.command)
        x, y, yaw = pose

        # Create quaternion from yaw
        odom_quat = euler_to_quaternion(0, 0, yaw)

        # Publish the transform over tf
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation = odom_quat
        self.odom_broadcaster.sendTransform(t)

        # Publish the odometry message
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = self.odom_frame

        # Set the position
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = odom_quat

        # Set the velocity
        odom.child_frame_id = self.base_frame
        odom.twist.twist.linear.x = self.command[0] * self.unicycle.r * math.cos(yaw)
        odom.twist.twist.linear.y = self.command[0] * self.unicycle.r * math.sin(yaw)
        odom.twist.twist.angular.z = self.command[1]

        self.odom_pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = UnicycleROS2Wrapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

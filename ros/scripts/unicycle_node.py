#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
import tf
import sys
import os
import math

# Add the parent directory to the Python path to allow importing the unicycle module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python.kinematics.unicycle_kinematics import Unicycle

class UnicycleROSWrapper:
    def __init__(self):
        rospy.init_node('unicycle_controller', anonymous=True)

        # Parameters
        self.dt = rospy.get_param('~dt', 0.01)
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')

        # Initial state
        x_init = rospy.get_param('~x_init', 0.0)
        y_init = rospy.get_param('~y_init', 0.0)
        yaw_init = rospy.get_param('~yaw_init', 0.0)

        # Unicycle instance
        self.unicycle = Unicycle(self.dt, init=[x_init, y_init, yaw_init])
        self.command = [0.0, 0.0]

        # Subscribers and Publishers
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.odom_broadcaster = tf.TransformBroadcaster()

        # Main loop
        self.rate = rospy.Rate(1.0 / self.dt)
        self.last_time = rospy.Time.now()

    def cmd_vel_callback(self, msg):
        # The unicycle model takes linear velocity (w_y) and angular velocity (w_z)
        self.command[0] = msg.linear.x
        self.command[1] = msg.angular.z

    def run(self):
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()

            # Simulate one step
            pose, _ = self.unicycle.simulate(self.command)
            x, y, yaw = pose

            # Create quaternion from yaw
            odom_quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

            # Publish the transform over tf
            self.odom_broadcaster.sendTransform(
                (x, y, 0.),
                odom_quat,
                current_time,
                self.base_frame,
                self.odom_frame
            )

            # Publish the odometry message
            odom = Odometry()
            odom.header.stamp = current_time
            odom.header.frame_id = self.odom_frame

            # Set the position
            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation = Quaternion(*odom_quat)

            # Set the velocity
            odom.child_frame_id = self.base_frame
            odom.twist.twist.linear.x = self.command[0] * self.unicycle.r * math.cos(yaw)
            odom.twist.twist.linear.y = self.command[0] * self.unicycle.r * math.sin(yaw)
            odom.twist.twist.angular.z = self.command[1]

            self.odom_pub.publish(odom)

            self.last_time = current_time
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = UnicycleROSWrapper()
        node.run()
    except rospy.ROSInterruptException:
        pass

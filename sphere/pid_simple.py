# Hierarchical PID controller

import numpy as np
import math


class PID:

    def __init__(self, dt):
        self.dt = dt

        # Gains
        # position
        self.kp_p = 10
        self.kp_i = 1
        self.kp_d = 0.1
        # orientation
        self.ko_p = 5
        self.ko_i = 0.5
        self.ko_d = 0.05

        self.old_pose = np.zeros((3, 1))
        self.ie_pose = np.zeros((3, 1))

    def control(self, pose, trajectory):

        # Actual state

        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        # Reference values

        x_ref = trajectory[0]
        y_ref = trajectory[1]

        # Compute pose errors

        e_x = x_ref - x
        e_y = y_ref - y
        yaw_ref = math.atan2(e_y, e_x) - math.pi/4
        e_yaw = yaw_ref - yaw
        if np.abs(e_yaw) > math.pi:
            e_yaw = yaw_ref - yaw - 2 * math.pi * np.sign(e_yaw)

        # Pose controller

        self.ie_pose[0] = min(max(self.ie_pose[0] + e_x * self.dt, -1), 1)
        self.ie_pose[1] = min(max(self.ie_pose[1] + e_y * self.dt, -1), 1)
        self.ie_pose[2] = min(max(self.ie_pose[2] + e_yaw * self.dt, -1), 1)

        v_x = (pose[0] - self.old_pose[0])/self.dt
        v_y = (pose[1] - self.old_pose[1])/self.dt
        if np.abs(pose[2] - self.old_pose[2]) > math.pi:
            self.old_pose[2] -= 2*math.pi*np.sign(self.old_pose[2])
        w_z = (pose[2] - self.old_pose[2])/self.dt
        self.old_pose = pose.copy()

        v_x_ref = self.kp_p * e_x + self.kp_i * self.ie_pose[0] + self.kp_d * v_x
        v_y_ref = self.kp_p * e_y + self.kp_i * self.ie_pose[1] + self.kp_d * v_y
        w_z_ref = self.ko_p * e_yaw + self.ko_i * self.ie_pose[2] + self.ko_d * w_z

        # Velocities in body frame

        v_x_b = np.cos(yaw) * v_x_ref + np.sin(yaw) * v_y_ref
        v_y_b = -np.sin(yaw) * v_x_ref + np.cos(yaw) * v_y_ref
        w_x_ref = -v_y_b
        w_y_ref = v_x_b

        command = np.ravel(np.array([w_x_ref, w_y_ref, w_z_ref], dtype=object))
        return command

# Hierarchical PID controller

import numpy as np
import math


class PID:

    def __init__(self, dt):
        self.dt = dt

        # Gains
        # position
        self.kp_p = 200
        self.kp_i = 20
        self.kp_d = 0.2
        # orientation
        self.ko_p = 100
        self.ko_i = 1
        self.ko_d = 0.01

        self.old_pose = np.zeros((3, 1))
        self.i_distance = 0
        self.ie_yaw = 0

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
        yaw_ref = math.atan2(e_y, e_x)
        # if np.abs(yaw_ref) > math.pi/2:
        #     yaw_ref = yaw_ref + math.pi
        e_yaw = yaw_ref - yaw
        e_yaw -= (abs(e_yaw) > math.pi) * 2 * math.pi * np.sign(e_yaw)

        # Pose controller

        distance = np.cos(e_yaw)*np.sqrt(e_x**2 + e_y**2)
        self.i_distance = min(max(self.i_distance + distance*self.dt, - 1), 1)
        self.ie_yaw = min(max(self.ie_yaw + e_yaw*self.dt, - 1), 1)
        vx = (pose[0] - self.old_pose[0])/self.dt
        vy = (pose[1] - self.old_pose[1])/self.dt
        if (np.abs(math.atan2(vy, vx) - yaw) < math.pi/4) or (np.abs(math.atan2(vy, vx) - yaw) > 7/4*math.pi):
            direction = 1
        else:
            direction = - 1
        v = direction*np.sqrt(vx**2 + vy**2)
        if np.abs(pose[2] - self.old_pose[2]) > math.pi:
            self.old_pose[2] -= 2*math.pi*np.sign(self.old_pose[2])
        w = (pose[2] - self.old_pose[2])/self.dt
        self.old_pose = pose.copy()
        v_ref = self.kp_p*distance + self.kp_i*self.i_distance + self.kp_d*v
        w_ref = self.ko_p*e_yaw + self.ko_i*self.ie_yaw + self.ko_d*w

        trajectory[2] = yaw_ref  # for debug
        command = np.ravel(np.array([v_ref, w_ref], dtype=object))
        return command

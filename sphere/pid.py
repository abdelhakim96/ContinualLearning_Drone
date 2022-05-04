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
        # linear velocity
        self.kv_p = 100
        self.kv_i = 10
        self.kv_d = 1
        # angular velocity
        self.kw_p = 100
        self.kw_i = 10
        self.kw_d = 1

        self.old_pose = np.zeros((3, 1))
        self.ie_pose = np.zeros((3, 1))
        self.old_velocity = np.zeros((3, 1))
        self.ie_velocity = np.zeros((3, 1))

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

        vx = (pose[0] - self.old_pose[0])/self.dt
        vy = (pose[1] - self.old_pose[1])/self.dt

        if np.abs(pose[2] - self.old_pose[2]) > math.pi:
            self.old_pose[2] -= 2*math.pi*np.sign(self.old_pose[2])
        wz = (pose[2] - self.old_pose[2])/self.dt
        self.old_pose = pose.copy()

        vx_ref = self.kp_p * e_x + self.kp_i * self.ie_pose[0] + self.kp_d * vx
        vy_ref = self.kp_p * e_y + self.kp_i * self.ie_pose[1] + self.kp_d * vy
        wz_ref = self.ko_p * e_yaw + self.ko_i * self.ie_pose[2] + self.ko_d * wz

        # Compute velocity error

        e_vx = vx_ref - vx
        e_vy = vy_ref - vy
        e_wz = wz_ref - wz

        # Velocity controller

        self.ie_velocity[0] = min(max(self.ie_velocity[0] + e_vx * self.dt, -1), 1)
        self.ie_velocity[1] = min(max(self.ie_velocity[1] + e_vy * self.dt, -1), 1)
        self.ie_velocity[2] = min(max(self.ie_velocity[2] + e_wz * self.dt, -1), 1)

        ax = (vx - self.old_velocity[0]) / self.dt
        ay = (vy - self.old_velocity[1]) / self.dt
        az = (wz - self.old_velocity[2]) / self.dt
        self.old_velocity = np.array([vx, vy, wz])

        tau_x = -self.kv_p * e_vy - self.kv_i * self.ie_velocity[1] - self.kv_d * ay
        tau_y = self.kv_p * e_vx + self.kv_i * self.ie_velocity[0] + self.kv_d * ax
        tau_z = self.kw_p * e_wz + self.kw_i * self.ie_velocity[2] + self.kw_d * az

        tau_x_ref = np.cos(yaw) * tau_x + np.sin(yaw) * tau_y
        tau_y_ref = -np.sin(yaw) * tau_x + np.cos(yaw) * tau_y
        tau_z_ref = tau_z

        command = np.ravel(np.array([tau_x_ref, tau_y_ref, tau_z_ref], dtype=object))
        return command

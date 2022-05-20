# Inverse controller

import numpy as np
import math


class Inverse:

    def __init__(self, dt):
        self.dt = dt

        self.r = 0.5   # wheel radius
        self.i_y = 1   # wheel rotational inertia around y-axis
        self.i_z = 10  # wheel rotational inertia around z-axis

        self.old_pose = np.zeros((3, 1))

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
        e_yaw = yaw_ref - yaw
        if abs(e_yaw) > math.pi:
            e_yaw = yaw_ref - yaw - 2 * math.pi * np.sign(e_yaw)

        # Pose controller

        v_x = (pose[0] - self.old_pose[0]) / self.dt
        v_y = (pose[1] - self.old_pose[1]) / self.dt
        if (np.abs(math.atan2(v_y, v_x) - yaw) < math.pi / 4) or (np.abs(math.atan2(v_y, v_x) - yaw) > 7 / 4 * math.pi):
            direction = 1
        else:
            direction = - 1
        v = direction * np.sqrt(v_x ** 2 + v_y ** 2)
        w_y = v / self.r
        if np.abs(pose[2] - self.old_pose[2]) > math.pi:
            self.old_pose[2] -= 2 * math.pi * np.sign(self.old_pose[2])
        w_z = (pose[2] - self.old_pose[2]) / self.dt
        self.old_pose = pose.copy()

        # Inverse law

        # tau_y = self.i_y / self.r * (math.cos(yaw + w_z * self.dt) * (e_x + 2 * math.cos(yaw + w_z * self.dt) * self.r * w_y * self.dt) +
        #                              math.sin(yaw + w_z * self.dt) * (e_y + 2 * math.sin(yaw + w_z * self.dt) * self.r * w_y * self.dt)) / self.dt
        tau_y = self.i_y * (
                    1 / self.r * (
                        math.cos(yaw + w_z * self.dt) * e_x + 2*self.r * math.cos(yaw) * w_y * self.dt +
                        math.sin(yaw + w_z * self.dt) * e_y + 2*self.r * math.sin(yaw) * w_y * self.dt
                    ) / self.dt
                - w_y) / self.dt
        tau_y = 10
        tau_z = self.i_z/2 * ((e_yaw - w_z * self.dt) / self.dt - w_z) / self.dt

        command = np.ravel(np.array([tau_y, tau_z], dtype=object))
        return command

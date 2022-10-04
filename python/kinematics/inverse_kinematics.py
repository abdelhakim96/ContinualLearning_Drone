# Inverse controller

import numpy as np
import math


class Inverse:

    def __init__(self, unicycle):
        self.unicycle = unicycle
        self.dt = self.unicycle.dt

    def control(self, pose, trajectory):

        # Parameters

        r = self.unicycle.r
        disturbance = self.unicycle.disturbance

        # Actual state

        pose -= self.unicycle.noise
        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        # Reference values

        x_ref = trajectory[0]
        y_ref = trajectory[1]

        # Compute pose errors

        e_x = x_ref - x - disturbance[0] * self.dt
        e_y = y_ref - y - disturbance[1] * self.dt
        yaw_ref = math.atan2(e_y, e_x)
        e_yaw = yaw_ref - yaw
        e_yaw -= (abs(e_yaw) > math.pi) * 2 * math.pi * np.sign(e_yaw)
        trajectory[2] = yaw_ref  # for debug

        # Inverse law

        w_y = 1 / r * (math.cos(yaw) * e_x + math.sin(yaw) * e_y) / self.dt
        w_z = (e_yaw - disturbance[2] * self.dt) / self.dt

        command = np.ravel(np.array([w_y, w_z], dtype=object))
        return command

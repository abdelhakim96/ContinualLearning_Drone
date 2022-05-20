# Inverse controller

import numpy as np
import math


class Inverse:

    def __init__(self, unicycle):
        self.unicycle = unicycle
        self.dt = self.unicycle.dt

    def control(self, pose, trajectory):

        # Actual state

        x = pose[0] - self.unicycle.noise[0]
        y = pose[1] - self.unicycle.noise[1]
        yaw = pose[2] - self.unicycle.noise[2]

        # Reference values

        x_ref = trajectory[0]
        y_ref = trajectory[1]

        # Compute pose errors

        e_x = x_ref - x
        e_y = y_ref - y
        yaw_ref = math.atan2(e_y, e_x)
        e_yaw = yaw_ref - yaw
        e_yaw -= (abs(e_yaw) > math.pi) * 2 * math.pi * np.sign(e_yaw)

        # Inverse law

        w_y = 1 / self.unicycle.r * (math.cos(yaw) * e_x + math.sin(yaw) * e_y) / self.dt - self.unicycle.disturbance
        #w_y = (e_x**2 + e_y**2)**(1/2)/(self.unicycle.r*self.unicycle.dt)
        w_z = e_yaw / self.dt - self.unicycle.disturbance

        command = np.ravel(np.array([w_y, w_z], dtype=object))
        return command

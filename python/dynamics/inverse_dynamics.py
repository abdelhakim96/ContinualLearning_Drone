# Inverse controller

import numpy as np
import math


class Inverse:

    def __init__(self, unicycle):
        self.unicycle = unicycle
        self.dt = self.unicycle.dt

        self.r = 0.5   # wheel radius
        self.i_y = 1   # wheel rotational inertia around y-axis
        self.i_z = 10  # wheel rotational inertia around z-axis

        self.old_pose = np.zeros((3, 1))

    def control(self, pose, trajectory):

        # Actual state

        pose -= self.unicycle.noise
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
            e_yaw = e_yaw - 2 * math.pi * np.sign(e_yaw)

        # Pose controller

        v_x = (pose[0] - self.old_pose[0]) / self.dt
        v_y = (pose[1] - self.old_pose[1]) / self.dt
        if (np.abs(math.atan2(v_y, v_x) - self.old_pose[2]) < math.pi / 4) or \
           (np.abs(math.atan2(v_y, v_x) - self.old_pose[2]) > 7 / 4 * math.pi):
            direction = 1
        else:
            direction = - 1
        v = direction * np.sqrt(v_x ** 2 + v_y ** 2)
        w_y = v / self.r
        if np.abs(pose[2] - self.old_pose[2]) > math.pi:
            self.old_pose[2] -= 2 * math.pi * np.sign(self.old_pose[2])
        w_z = (pose[2] - self.old_pose[2]) / self.dt
        self.old_pose = pose.copy()

        # print([self.unicycle.state[3], ' = ', w_y, ', ', self.unicycle.state[4], ' = ', w_z])

        # get ground truth for angular velocities
        # w_y = self.unicycle.state[3]
        # w_z = self.unicycle.state[4]

        # Inverse law

        r = self.unicycle.r
        i_y = self.unicycle.i_y
        i_z = self.unicycle.i_z
        disturbance = self.unicycle.disturbance
        tau_y = -(i_y * (
                x * math.cos(yaw + w_z * self.dt) - x_ref * math.cos(yaw + w_z * self.dt) +
                y * math.sin(yaw + w_z * self.dt) - y_ref * math.sin(yaw + w_z * self.dt) +
                r * w_y * self.dt + r * w_y * self.dt * math.cos(w_z * self.dt))) / \
                (r * (self.dt ** 2)) - disturbance
        tau_z = i_z * (e_yaw - 2 * w_z * self.dt) / (self.dt ** 2) - disturbance

        # damping to compensate for bounding commands
        tau_y -= 30000 * w_y
        tau_z -= 1000000 * w_z

        command = np.ravel(np.array([tau_y, tau_z], dtype=object))
        return command

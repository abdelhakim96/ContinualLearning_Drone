# Inverse controller

import numpy as np
import math


class Inverse:

    def __init__(self, unicycle):
        self.unicycle = unicycle
        self.dt = self.unicycle.dt

        self.r = 0.3                            # initial wheel radius
        self.m = 20                             # initial mass
        self.i_y = self.m * self.r ** 2         # initial wheel rotational inertia around y-axis
        self.i_z = self.m * self.r ** 2 / 2     # initial wheel rotational inertia around z-axis

        self.old_pose = np.zeros((3, 1))

    def control(self, pose, trajectory):

        # Parameters

        r = self.unicycle.r
        i_y = self.unicycle.i_y
        i_z = self.unicycle.i_z

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
        e_yaw -= (abs(e_yaw) > math.pi) * 2 * math.pi * np.sign(e_yaw)  # denormalize heading
        trajectory[2] = yaw_ref  # for debug

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

        tau_y = i_y / (r * self.dt ** 2) * (
                e_x * math.cos(yaw + w_z * self.dt) +
                e_y * math.sin(yaw + w_z * self.dt) +
                -r * w_y * self.dt * (1 + math.cos(w_z * self.dt)))
        tau_z = i_z * (e_yaw - 2 * w_z * self.dt) / (self.dt ** 2)

        tau_y -= self.unicycle.disturbance[0]
        tau_z -= self.unicycle.disturbance[1]

        # Damp to compensate for saturated control inputs

        d = np.sqrt(e_x**2 + e_y**2)
        tau_y -= 100000 * w_y
        tau_z -= 100000 * w_z

        command = np.ravel(np.array([tau_y, tau_z], dtype=object))
        return command

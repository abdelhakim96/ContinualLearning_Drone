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
        self.kv_p = 400
        self.kv_i = 40
        self.kv_d = 4
        # angular velocity
        self.kw_p = 400
        self.kw_i = 40
        self.kw_d = 4

        self.old_pose = np.zeros((3, 1))
        self.i_distance = 0
        self.ie_yaw = 0
        self.old_velocity = np.zeros((3, 1))
        self.ie_v = 0
        self.ie_w = 0

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
        if np.abs(e_yaw) > math.pi:
            e_yaw = yaw_ref - yaw - 2*math.pi*np.sign(e_yaw)

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

        # Compute velocity error

        e_v = v_ref - v
        e_w = w_ref - w

        # Velocity controller

        self.ie_v = min(max(self.ie_v + e_v*self.dt, - 1), 1)
        self.ie_w = min(max(self.ie_w + e_w*self.dt, - 1), 1)
        a_v = (v - self.old_velocity[0])/self.dt
        a_w = (w - self.old_velocity[1])/self.dt
        self.old_velocity = np.array([v, w])
        tau_y = self.kv_p*e_v + self.kv_i*self.ie_v + self.kv_d*a_v
        tau_z = self.kw_p*e_w + self.kw_i*self.ie_w + self.kw_d*a_w

        command = np.ravel(np.array([tau_y, tau_z], dtype=object))
        return command

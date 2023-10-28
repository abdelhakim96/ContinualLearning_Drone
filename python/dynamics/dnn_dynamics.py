# Hierarchical DNN-based controller

import numpy as np
import torch
import math
import csv

from models.model import MLP


class DNN:
    def __init__(self, dt, model_name):
        self.dt = dt

        with open('models/properties_' + model_name + '.csv', 'r', newline='') as f:
            data = list(csv.reader(f))

        self.variables = data[0][1:]
        self.mu = np.array(data[1][1:], dtype=np.float32)
        self.sigma = np.array(data[2][1:], dtype=np.float32)
        self.data_min = np.array(data[3][1:], dtype=np.float32)
        self.data_max = np.array(data[4][1:], dtype=np.float32)
        self.type_scaling = int(data[6][1])
        self.num_inputs = int(data[7][1])
        self.num_outputs = int(data[8][1])

        num_hidden_units = []
        num = ''
        for c in model_name:
            if c.isdigit():
                num = num + c
            else:
                if num != '':
                    num_hidden_units = np.append(num_hidden_units, int(num))
                    num = ''
        if num != '':
            num_hidden_units = np.append(num_hidden_units, int(num))
        num_hidden_units = num_hidden_units.astype(int)

        self.model = MLP(self.num_inputs, num_hidden_units, self.num_outputs)
        self.model.load_state_dict(torch.load('models/' + model_name + '.pth'))
        self.model.eval()
        self.model = self.model.float()

        self.old_pose = np.zeros((3, 1))

    def get_data_min(self):
        return self.data_min

    def get_data_max(self):
        return self.data_max

    def control(self, pose, trajectory):

        # Actual state

        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        # Reference values

        diff_x = trajectory[0] - x
        diff_y = trajectory[1] - y

        diff_yaw = math.atan2(diff_y, diff_x) - yaw
        diff_yaw -= (abs(diff_yaw) > math.pi) * 2 * math.pi * np.sign(diff_yaw)

        # Compute velocities

        vx = (x - self.old_pose[0])/self.dt
        vy = (y - self.old_pose[1])/self.dt
        if (np.abs(math.atan2(vy, vx) - self.old_pose[2]) < math.pi/4) or \
           (np.abs(math.atan2(vy, vx) - self.old_pose[2]) > 7/4*math.pi):
            direction = 1
        else:
            direction = - 1
        v = direction*np.sqrt(vx**2 + vy**2).item()
        if np.abs(pose[2] - self.old_pose[2]) > math.pi:
            self.old_pose[2] -= 2*math.pi*np.sign(self.old_pose[2])
        w = (pose[2] - self.old_pose[2]).item()/self.dt
        self.old_pose = pose.copy()

        # Compute inputs to DNN
        dnn_input = np.array([diff_x, diff_y, np.sin(yaw), np.cos(yaw), v, w])
        dnn_input = np.minimum(np.maximum(dnn_input, self.data_min[:self.num_inputs]), self.data_max[:self.num_inputs])

        if self.type_scaling == 1:
            dnn_input = (dnn_input - self.mu[:self.num_inputs]) / self.sigma[:self.num_inputs]
        elif self.type_scaling == 2:
            dnn_input = 2 * (dnn_input - self.data_min[:self.num_inputs]) / \
                        (self.data_max[:self.num_inputs] - self.data_min[:self.num_inputs]) - 1

        # Predict the commands

        command = self.model(torch.from_numpy(dnn_input).float()).detach().numpy()

        # Unscale data

        if self.type_scaling == 1:
            command = command * self.sigma[self.num_inputs:] + self.mu[self.num_inputs:]
        elif self.type_scaling == 2:
            command = (command + 1) * (self.data_max[self.num_inputs:] - self.data_min[self.num_inputs:]) / 2 + \
                      self.data_min[self.num_inputs:]

        return command

    def learn(self, pose, pose_old, command_old):
        return

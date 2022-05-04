# Hierarchical DNN-based controller

import numpy as np
import torch
import pickle
import math

from torch import nn


class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden_units, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, num_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(num_hidden_units[0], num_hidden_units[1]),
            nn.Tanh(),
            nn.Linear(num_hidden_units[1], num_outputs)
        )

    def forward(self, x):
        return self.layers(x)


class DNN:
    def __init__(self, dt, model_name):
        self.dt = dt
        self.old_pose = [0, 0, 0]

        with open('models/sphere.pkl', 'rb') as f:
            dictionary = pickle.load(f)

        variables = ['diff_x(k)', 'diff_y(k)', 'diff_x(k+1)', 'diff_y(k+1)', 'diff_x(k+2)', 'diff_y(k+2)', 'sin', 'cos', 'v_x', 'v_y', 'w_z',
                     'tau_x', 'tau_y', 'tau_z']

        self.type_scaling = dictionary['type_scaling']
        self.data_min = dictionary['data_min'][variables].values
        self.data_max = dictionary['data_max'][variables].values
        self.mu = dictionary['mu'][variables].values
        self.sigma = dictionary['sigma'][variables].values
        self.num_inputs = dictionary['num_inputs']
        self.num_outputs = dictionary['num_outputs']

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

    def control(self, pose, trajectory):

        # Actual state

        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        # Compute velocities

        vx = (x - self.old_pose[0]) / self.dt
        vy = (y - self.old_pose[1]) / self.dt
        wz = (yaw - self.old_pose[2])
        if np.abs(wz) > math.pi:
            wz = (wz - 2 * math.pi * np.sign(wz)) / self.dt
        self.old_pose = [x, y, yaw]

        # Reference values

        x_ref1 = trajectory[0, 0]
        y_ref1 = trajectory[0, 1]
        x_ref2 = trajectory[1, 0]
        y_ref2 = trajectory[1, 1]
        x_ref3 = trajectory[2, 0]
        y_ref3 = trajectory[2, 1]

        # Compute inputs to DNN

        dnn_input = np.array([x_ref1 - x, y_ref1 - y, x_ref2 - x, y_ref2 - y, x_ref3 - x, y_ref3 - y, np.sin(yaw), np.cos(yaw), vx, vy, wz])
        dnn_input = np.minimum(np.maximum(dnn_input, self.data_min[:self.num_inputs]), self.data_max[:self.num_inputs])

        if self.type_scaling == 1:
            dnn_input = (dnn_input - self.mu[:self.num_inputs]) / self.sigma[:self.num_inputs]
        else:
            if self.type_scaling == 2:
                dnn_input = 2 * (dnn_input - self.data_min[:self.num_inputs]) / \
                            (self.data_max[:self.num_inputs] - self.data_min[:self.num_inputs]) - 1

        # Predict the commands

        command = self.model(torch.from_numpy(dnn_input).float()).detach().numpy()

        # Unscale data

        if self.type_scaling == 1:
            command = command * self.sigma[self.num_inputs:] + self.mu[self.num_inputs:]
        else:
            if self.type_scaling == 2:
                command = (command + 1) * (self.data_max[self.num_inputs:] - self.data_min[self.num_inputs:]) / 2 + \
                          self.data_min[self.num_inputs:]

        return command

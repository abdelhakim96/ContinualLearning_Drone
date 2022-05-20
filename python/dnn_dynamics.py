# Hierarchical DNN-based controller

import numpy as np
import torch
import math
import pickle

from model import MLP


class DNN:
    def __init__(self, dt, model_name):
        self.dt = dt

        with open('models/parameters_dynamics.pkl', 'rb') as f:
            dictionary = pickle.load(f)

        variables = ['sin', 'cos', 'v', 'w', 'diff_x(k+1)', 'diff_y(k+1)', 'diff_yaw(k+1)', 'tau_y', 'tau_z']

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

        self.old_pose = np.zeros((3, 1))

    def control(self, pose, trajectory):

        # Actual state

        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        # Reference values

        diff_x_k = trajectory[0, 0] - x
        diff_y_k = trajectory[0, 1] - y
        diff_x_k1 = trajectory[1, 0] - x
        diff_y_k1 = trajectory[1, 1] - y

        diff_yaw_k1 = math.atan2(diff_x_k1, diff_y_k1) - yaw
        diff_yaw_k1 -= (abs(diff_yaw_k1) > math.pi) * 2 * math.pi * np.sign(diff_yaw_k1)

        # Compute velocities

        vx = (x - self.old_pose[0])/self.dt
        vy = (y - self.old_pose[1])/self.dt
        if (np.abs(math.atan2(vy, vx) - yaw) < math.pi/4) or (np.abs(math.atan2(vy, vx) - yaw) > 7/4*math.pi):
            direction = 1
        else:
            direction = - 1
        v = direction*np.sqrt(vx**2 + vy**2)
        if np.abs(pose[2] - self.old_pose[2]) > math.pi:
            self.old_pose[2] -= 2*math.pi*np.sign(self.old_pose[2])
        w = (pose[2] - self.old_pose[2])/self.dt
        self.old_pose = pose.copy()

        # Compute inputs to DNN
        dnn_input = np.array([np.sin(yaw), np.cos(yaw), v.item(), w.item(), diff_x_k1, diff_y_k1, diff_yaw_k1])
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

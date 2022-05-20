# Hierarchical DNN-based controller
# https://github.com/mrifkikurniawan/sslad
# https://github.com/RaptorMai/online-continual-learning
# https://github.com/drimpossible/GDumb

import numpy as np
import torch
import math
import pickle

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from model import MLP


class DNN:
    def __init__(self, _, model_name):
        with open('models/parameters_kinematics.pkl', 'rb') as f:
            dictionary = pickle.load(f)

        variables = ['sin', 'cos', 'diff_x', 'diff_y', 'diff_yaw', 'w_y', 'w_z']

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

        # Define the loss function and optimizer

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001) # SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer.load_state_dict(torch.load('models/optimiser_' + model_name + '.pth'))

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

        diff_x_k = trajectory[0] - x
        diff_y_k = trajectory[1] - y

        yaw_ref = math.atan2(diff_y_k, diff_x_k)
        diff_yaw_k = yaw_ref - yaw
        diff_yaw_k -= (abs(diff_yaw_k) > math.pi) * 2 * math.pi * np.sign(diff_yaw_k)

        # Compute inputs to DNN

        dnn_input = np.array([np.sin(yaw), np.cos(yaw), diff_x_k, diff_y_k, diff_yaw_k])
        dnn_input = np.minimum(np.maximum(dnn_input, self.data_min[:self.num_inputs]), self.data_max[:self.num_inputs])

        # Scale inputs

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

    def learn(self, pose, pose_old, command_old):

        yaw = pose[2]

        diff_x_k = pose[0] - pose_old[0]
        diff_y_k = pose[1] - pose_old[1]
        diff_yaw_k = pose[2] - pose_old[2]
        diff_yaw_k -= (abs(diff_yaw_k) > math.pi) * 2 * math.pi * np.sign(diff_yaw_k)

        # Compute inputs and outputs

        dnn_data = np.array([np.sin(yaw), np.cos(yaw), diff_x_k, diff_y_k, diff_yaw_k, command_old[0], command_old[1]])

        # Scale data

        if self.type_scaling == 1:
            dnn_data = (dnn_data - self.mu) / self.sigma
        else:
            if self.type_scaling == 2:
                dnn_data = 2 * (dnn_data - self.data_min) / (self.data_max - self.data_min) - 1

        # Split into input and output

        dnn_input = dnn_data[:self.num_inputs]
        dnn_output = dnn_data[self.num_inputs:]
        dnn_input = torch.from_numpy(dnn_input).float()
        dnn_output = torch.from_numpy(dnn_output).float()

        # Run the training

        # self.model.train()
        # for epoch in range(1):
        #     dnn_output_prediction = self.model(dnn_input)  # perform forward pass
        #     loss = self.criterion(dnn_output_prediction, dnn_output)  # compute loss
        #     self.optimizer.zero_grad()  # zero the gradients
        #     loss.backward()  # perform backward pass
        #     self.optimizer.step()  # perform optimization

# Hierarchical DNN-based controller
# https://github.com/mrifkikurniawan/sslad
# https://github.com/RaptorMai/online-continual-learning
# https://github.com/drimpossible/GDumb

import numpy as np
import pandas as pd
import torch
import math
import csv

from models.model import MLP


class DFNN:
    def __init__(self, unicycle, model_name):

        self.unicycle = unicycle

        with open('models/data_properties_kinematics.csv', 'r', newline='') as f:
            data = list(csv.reader(f))

        self.variables = data[0][1:]
        self.mu = np.array(data[1][1:], dtype=np.float32)
        self.sigma = np.array(data[2][1:], dtype=np.float32)
        self.data_min = np.array(data[3][1:], dtype=np.float32)
        self.data_max = np.array(data[4][1:], dtype=np.float32)

        self.num_inputs = 15
        self.num_outputs = 2

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

    def get_data_min(self):
        return self.data_min

    def get_data_max(self):
        return self.data_max

    def gaussian(self, x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))

    def control(self, pose, trajectory):

        #pose = pose - self.unicycle.noise

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
        dnn_input = np.minimum(np.maximum(dnn_input, self.data_min[:5]), self.data_max[:5])

        # Fuzzify inputs

        sin_N = self.gaussian(dnn_input[0], -1, 0.5)
        sin_Z = self.gaussian(dnn_input[0], 0, 0.5)
        sin_P = self.gaussian(dnn_input[0], 1, 0.5)

        cos_N = self.gaussian(dnn_input[1], -1, 0.5)
        cos_Z = self.gaussian(dnn_input[1], 0, 0.5)
        cos_P = self.gaussian(dnn_input[1], 1, 0.5)

        diff_x_N = self.gaussian(dnn_input[2], -0.006, 0.003)
        diff_x_Z = self.gaussian(dnn_input[2], 0, 0.003)
        diff_x_P = self.gaussian(dnn_input[2], 0.006, 0.003)

        diff_y_N = self.gaussian(dnn_input[3], -0.006, 0.003)
        diff_y_Z = self.gaussian(dnn_input[3], 0, 0.003)
        diff_y_P = self.gaussian(dnn_input[3], 0.006, 0.003)

        diff_yaw_N = self.gaussian(dnn_input[4], -0.09, 0.045)
        diff_yaw_Z = self.gaussian(dnn_input[4], 0, 0.045)
        diff_yaw_P = self.gaussian(dnn_input[4], 0.09, 0.045)

        dnn_input_fuzzy = np.array([sin_N, sin_Z, sin_P,
                                    cos_N, cos_Z, cos_P,
                                    diff_x_N, diff_x_Z, diff_x_P,
                                    diff_y_N, diff_y_Z, diff_y_P,
                                    diff_yaw_N, diff_yaw_Z, diff_yaw_P])

        # Predict commands

        command = self.model(torch.from_numpy(dnn_input_fuzzy).float()).detach().numpy()

        # Unscale data

        command = (command + 1) * (self.data_max[5:] - self.data_min[5:]) / 2 + self.data_min[5:]

        return command

    def save(self, dnn_name):
        torch.save(self.model.state_dict(), 'models/' + dnn_name + '.pth')

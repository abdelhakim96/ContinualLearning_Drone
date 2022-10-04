# Hierarchical DNN-based controller

import numpy as np
import pandas as pd
import torch
import math
import csv

from models.model import MLP
from cl.buffer import Buffer
from cl.er import ExperienceReplay
from cl.cache import Cache

numHiddenUnits1 = [64, 64, 64]
numHiddenUnits2 = [8, 8]

datasetName = 'unicycle_dynamics_random_bound100'
modelName1 = 'dnn_dynamics_y_' + 'x'.join([str(s) for s in numHiddenUnits1])
modelName2 = 'dnn_dynamics_z_' + 'x'.join([str(s) for s in numHiddenUnits2])


class DNN:
    def __init__(self, dt, _):
        self.dt = dt

        # Load data properties

        with open('models/data_properties_dynamics.csv', 'r', newline='') as f:
            data = list(csv.reader(f))

        self.variables = data[0][1:]
        self.mu = np.array(data[1][1:], dtype=np.float32)
        self.sigma = np.array(data[2][1:], dtype=np.float32)
        self.data_min = np.array(data[3][1:], dtype=np.float32)
        self.data_max = np.array(data[4][1:], dtype=np.float32)
        self.type_scaling = int(data[6][1])

        # Load models

        self.model1 = MLP(5, numHiddenUnits1, 1)
        self.model1.load_state_dict(torch.load('models/' + modelName1 + '.pth'))
        self.model1.eval()
        self.model1 = self.model1.float()

        self.model2 = MLP(2, numHiddenUnits2, 1)
        self.model2.load_state_dict(torch.load('models/' + modelName2 + '.pth'))
        self.model2.eval()
        self.model2 = self.model2.float()

        # Define the loss function and optimizer

        self.criterion1 = torch.nn.MSELoss()
        self.criterion2 = torch.nn.MSELoss()

        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=0.001)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=0.001)

        # Define cache queue and buffer
        buffer_size = 100

        self.cache1 = Cache(size=32)
        self.buffer1 = Buffer(self.model1, buffer_size=buffer_size, input_size=4, output_size=1)
        self.agent1 = ExperienceReplay(model=self.model1, opt=self.optimizer1, buffer=self.buffer1, epoch=1,
                                       eps_mem_batch=32, mem_iters=1)

        self.cache2 = Cache(size=32)
        self.buffer2 = Buffer(self.model2, buffer_size=buffer_size, input_size=1, output_size=1)
        self.agent2 = ExperienceReplay(model=self.model2, opt=self.optimizer2, buffer=self.buffer2, epoch=1,
                                       eps_mem_batch=32, mem_iters=1)

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
        # self.old_pose[2] = -self.old_pose[2]

        # Reference values

        x_ref = trajectory[0]
        y_ref = trajectory[1]

        # Pose differences

        diff_x = x_ref - x
        diff_y = y_ref - y

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
        v = direction*np.sqrt(vx**2 + vy**2)
        if np.abs(pose[2] - self.old_pose[2]) > math.pi:
            self.old_pose[2] -= 2*math.pi*np.sign(self.old_pose[2])
        w = (pose[2] - self.old_pose[2])/self.dt
        self.old_pose = pose.copy()

        # Compute inputs to DNN
        dnn_input = np.array([yaw, v.item(), w.item(), diff_x, diff_y, diff_yaw])
        dnn_input = np.minimum(np.maximum(dnn_input, self.data_min[:len(dnn_input)]), self.data_max[:len(dnn_input)])

        # Scale inputs

        if self.type_scaling == 1:
            dnn_input = (dnn_input - self.mu[:len(dnn_input)]) / self.sigma[:len(dnn_input)]
        else:
            if self.type_scaling == 2:
                dnn_input = 2 * (dnn_input - self.data_min[:len(dnn_input)]) / \
                            (self.data_max[:len(dnn_input)] - self.data_min[:len(dnn_input)]) - 1

        # Predict the commands

        dnn_input1 = dnn_input[[0, 1, 2, 3, 4]]
        command1 = self.model1(torch.from_numpy(dnn_input1).float()).detach()[0]

        dnn_input2 = dnn_input[[2, 5]]
        command2 = self.model2(torch.from_numpy(dnn_input2).float()).detach()[0]

        command = np.array([command1, command2])

        # Unscale data

        if self.type_scaling == 1:
            command = command * self.sigma[len(dnn_input):] + self.mu[len(dnn_input):]
        else:
            if self.type_scaling == 2:
                command = (command + 1) * (self.data_max[len(dnn_input):] - self.data_min[len(dnn_input):]) / 2 + \
                          self.data_min[len(dnn_input):]

        return command

    # def learn(self, pose, pose_old, command_old):
    #     # pose, pose_old, command_old = self.get_random_sample()
    #
    #     yaw = pose[2]
    #
    #     diff_x_k = pose[0] - pose_old[0]
    #     diff_y_k = pose[1] - pose_old[1]
    #     diff_yaw_k = pose[2] - pose_old[2]
    #     diff_yaw_k -= (abs(diff_yaw_k) > math.pi) * 2 * math.pi * np.sign(diff_yaw_k)
    #
    #     # Compute inputs and outputs
    #
    #     dnn_data = np.array([np.sin(yaw), np.cos(yaw), diff_x_k, diff_y_k, diff_yaw_k, command_old[0], command_old[1]])
    #     # dnn_data = np.minimum(np.maximum(dnn_data, self.data_min), self.data_max)
    #
    #     # Scale data
    #
    #     if self.type_scaling == 1:
    #         dnn_data = (dnn_data - self.mu) / self.sigma
    #     else:
    #         if self.type_scaling == 2:
    #             dnn_data = 2 * (dnn_data - self.data_min) / (self.data_max - self.data_min) - 1
    #
    #     # Split into input and output
    #
    #     dnn_input1 = torch.from_numpy(dnn_data[[0, 1, 2, 3]]).float()
    #     dnn_input2 = torch.from_numpy(dnn_data[[4]]).float()
    #     dnn_output1 = torch.from_numpy(dnn_data[[5]]).float()
    #     dnn_output2 = torch.from_numpy(dnn_data[[6]]).float()
    #
    #     # Update the cache with this sample:
    #     self.cache1.update((dnn_input1, dnn_output1))
    #     self.cache2.update((dnn_input2, dnn_output2))
    #
    #     # Train the DNN only when cache has enough samples.
    #     if self.cache1.counter % self.cache1.size == 0:
    #         x_batch, y_batch = self.cache1.load_batch()  # tensor
    #
    #         # Run the training
    #         self.agent1.train(x_batch, y_batch)
    #
    #     if self.cache2.counter % self.cache2.size == 0:
    #         x_batch, y_batch = self.cache2.load_batch()  # tensor
    #
    #         # Run the training
    #         self.agent2.train(x_batch, y_batch)
    #
    #     # Update the buffer with the incoming sample.
    #     self.buffer1.reservoir_update(dnn_input1, dnn_output1)
    #     self.buffer2.reservoir_update(dnn_input2, dnn_output2)

    def save(self, _):
        torch.save(self.model1.state_dict(), 'models/new_' + modelName1 + '.pth')
        torch.save(self.model2.state_dict(), 'models/new_' + modelName2 + '.pth')

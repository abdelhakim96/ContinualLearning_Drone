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

numHiddenUnits1 = [16]
numHiddenUnits2 = [4]

datasetName = 'unicycle_kinematics_random_bound20_100'
modelName1 = 'dnn_kinematics_y_' + 'x'.join([str(s) for s in numHiddenUnits1])
modelName2 = 'dnn_kinematics_z_' + 'x'.join([str(s) for s in numHiddenUnits2])


class DNN:
    def __init__(self, _1, _2):
        # Load data properties

        with open('models/data_properties_kinematics.csv', 'r', newline='') as f:
            data = list(csv.reader(f))

        self.variables = data[0][1:]
        self.mu = np.array(data[1][1:], dtype=np.float32)
        self.sigma = np.array(data[2][1:], dtype=np.float32)
        self.data_min = np.array(data[3][1:], dtype=np.float32)
        self.data_max = np.array(data[4][1:], dtype=np.float32)
        self.type_scaling = int(data[6][1])

        # Load models

        self.model1 = MLP(4, numHiddenUnits1, 1)
        self.model1.load_state_dict(torch.load('models/' + modelName1 + '.pth'))
        self.model1.eval()
        self.model1 = self.model1.float()

        self.model2 = MLP(1, numHiddenUnits2, 1)
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

        # # Initialise cache and buffer with random samples
        # dataset = pd.read_csv('data/dataset_' + datasetName + '.csv')
        # np.random.seed(2022)  # FOR TESTING
        # random_indices = np.random.randint(len(dataset), size=buffer_size)
        # random_samples = dataset[['sin', 'cos', 'diff_x', 'diff_y', 'diff_yaw', 'w_y', 'w_z']].values[random_indices]
        # # random_samples = dataset.values  # FOR TESTING
        #
        # # Save smaller dataset
        # df = pd.DataFrame(data=random_samples, columns=['sin', 'cos', 'diff_x', 'diff_y', 'diff_yaw', 'w_y', 'w_z'])
        # df.to_csv('data/dataset_' + datasetName + '_smaller.csv', index=False)
        #
        # # Scale data
        # if self.type_scaling == 1:
        #     random_samples = (random_samples - self.mu) / self.sigma
        # if self.type_scaling == 2:
        #     random_samples = 2 * (random_samples - self.data_min) / (self.data_max - self.data_min) - 1
        #
        # for i in range(buffer_size):
        #     dnn_input1 = random_samples[i][[0, 1, 2, 3]]
        #     dnn_output1 = random_samples[i][[5]]
        #     dnn_input2 = random_samples[i][[4]]
        #     dnn_output2 = random_samples[i][[6]]
        #
        #     dnn_input1 = torch.from_numpy(dnn_input1).float()
        #     dnn_output1 = torch.from_numpy(dnn_output1).float()
        #     dnn_input2 = torch.from_numpy(dnn_input2).float()
        #     dnn_output2 = torch.from_numpy(dnn_output2).float()
        #
        #     self.cache1.update((dnn_input1, dnn_output1))
        #     self.buffer1.reservoir_update(dnn_input1, dnn_output1)
        #     self.cache2.update((dnn_input2, dnn_output2))
        #     self.buffer2.reservoir_update(dnn_input2, dnn_output2)

        log = pd.read_csv('data/log_' + datasetName + '.csv')
        self.samples = log[['x', 'y', 'yaw', 'w_y', 'w_z']].values

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
        dnn_input = np.minimum(np.maximum(dnn_input, self.data_min[:len(dnn_input)]), self.data_max[:len(dnn_input)])

        # Scale inputs

        if self.type_scaling == 1:
            dnn_input = (dnn_input - self.mu[:len(dnn_input)]) / self.sigma[:len(dnn_input)]
        else:
            if self.type_scaling == 2:
                dnn_input = 2 * (dnn_input - self.data_min[:len(dnn_input)]) / \
                            (self.data_max[:len(dnn_input)] - self.data_min[:len(dnn_input)]) - 1

        # Predict the commands

        dnn_input1 = dnn_input[[0, 1, 2, 3]]
        command1 = self.model1(torch.from_numpy(dnn_input1).float()).detach()[0]

        dnn_input2 = dnn_input[[4]]
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

    def learn(self, pose, pose_old, command_old):
        # pose, pose_old, command_old = self.get_random_sample()

        yaw = pose[2]

        diff_x_k = pose[0] - pose_old[0]
        diff_y_k = pose[1] - pose_old[1]
        diff_yaw_k = pose[2] - pose_old[2]
        diff_yaw_k -= (abs(diff_yaw_k) > math.pi) * 2 * math.pi * np.sign(diff_yaw_k)

        # Compute inputs and outputs

        dnn_data = np.array([np.sin(yaw), np.cos(yaw), diff_x_k, diff_y_k, diff_yaw_k, command_old[0], command_old[1]])
        # dnn_data = np.minimum(np.maximum(dnn_data, self.data_min), self.data_max)

        # Scale data

        if self.type_scaling == 1:
            dnn_data = (dnn_data - self.mu) / self.sigma
        else:
            if self.type_scaling == 2:
                dnn_data = 2 * (dnn_data - self.data_min) / (self.data_max - self.data_min) - 1

        # Split into input and output

        dnn_input1 = torch.from_numpy(dnn_data[[0, 1, 2, 3]]).float()
        dnn_input2 = torch.from_numpy(dnn_data[[4]]).float()
        dnn_output1 = torch.from_numpy(dnn_data[[5]]).float()
        dnn_output2 = torch.from_numpy(dnn_data[[6]]).float()

        # Update the cache with this sample:
        self.cache1.update((dnn_input1, dnn_output1))
        self.cache2.update((dnn_input2, dnn_output2))

        # Train the DNN only when cache has enough samples.
        if self.cache1.counter % self.cache1.size == 0:
            x_batch, y_batch = self.cache1.load_batch()  # tensor

            # Run the training
            self.agent1.train(x_batch, y_batch)

        if self.cache2.counter % self.cache2.size == 0:
            x_batch, y_batch = self.cache2.load_batch()  # tensor

            # Run the training
            self.agent2.train(x_batch, y_batch)

        # Update the buffer with the incoming sample.
        self.buffer1.reservoir_update(dnn_input1, dnn_output1)
        self.buffer2.reservoir_update(dnn_input2, dnn_output2)

    def get_random_sample(self):
        random_index = np.random.randint(len(self.samples) - 1)
        samples = self.samples[random_index:random_index + 2, :]

        return samples[1, 0:3], samples[0, 0:3], samples[0, 3:5]

    def save(self, _):
        torch.save(self.model1.state_dict(), 'models/new_' + modelName1 + '.pth')
        torch.save(self.model2.state_dict(), 'models/new_' + modelName2 + '.pth')

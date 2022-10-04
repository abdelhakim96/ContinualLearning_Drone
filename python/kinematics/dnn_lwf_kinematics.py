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
from cl.buffer import Buffer
from cl.lwf import LwF
from cl.cache import Cache

datasetName = 'unicycle_kinematics_random_bound20_100'

class DNN:
    def __init__(self, _, model_name):
        with open('models/data_properties_kinematics.csv', 'r', newline='') as f:
            data = list(csv.reader(f))

        self.variables = data[0][1:]
        self.mu = np.array(data[1][1:], dtype=np.float32)
        self.sigma = np.array(data[2][1:], dtype=np.float32)
        self.data_min = np.array(data[3][1:], dtype=np.float32)
        self.data_max = np.array(data[4][1:], dtype=np.float32)
        self.type_scaling = int(data[6][1])

        self.num_inputs = 5
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

        # Define the loss function and optimizer

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Define cache queue and buffer
        batch_size = 2
        buffer_size = 1000
        self.cache = Cache(size=batch_size)
        self.buffer = Buffer(self.model, buffer_size=buffer_size, input_size=5, output_size=2)
        self.agent = LwF(model=self.model,
                         opt=self.optimizer,
                         epoch=1,
                         lambda_=0.5,
                         buffer=self.buffer)

        # Initialise cache and buffer with random samples
        dataset = pd.read_csv('data/dataset_' + datasetName + '.csv')
        #np.random.seed(2022)  # FOR TESTING
        random_indices = np.random.randint(len(dataset), size=buffer_size)
        random_samples = dataset[['sin', 'cos', 'diff_x', 'diff_y', 'diff_yaw', 'w_y', 'w_z']].values[random_indices]
        #random_samples = dataset.values  # FOR TESTING

        # Save smaller dataset
        df = pd.DataFrame(data=random_samples, columns=['sin', 'cos', 'diff_x', 'diff_y', 'diff_yaw', 'w_y', 'w_z'])
        df.to_csv('data/dataset_' + datasetName + '_smaller.csv', index=False)

        # Scale data
        if self.type_scaling == 1:
            random_samples = (random_samples - self.mu) / self.sigma
        if self.type_scaling == 2:
            random_samples = 2 * (random_samples - self.data_min) / (self.data_max - self.data_min) - 1

        for i in range(buffer_size):
            dnn_input = random_samples[i][:self.num_inputs]
            dnn_output = random_samples[i][self.num_inputs:]

            dnn_input = torch.from_numpy(dnn_input).float()
            dnn_output = torch.from_numpy(dnn_output).float()

            self.cache.update((dnn_input, dnn_output))
            self.buffer.reservoir_update(dnn_input, dnn_output)

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
        """
        pose, pose_old, command_old: single sample, used to calculate training dat & target,

        :return:
        """

        yaw = pose[2]

        diff_x_k = pose[0] - pose_old[0]
        diff_y_k = pose[1] - pose_old[1]
        diff_yaw_k = pose[2] - pose_old[2]
        diff_yaw_k -= (abs(diff_yaw_k) > math.pi) * 2 * math.pi * np.sign(diff_yaw_k)

        # Compute inputs and outputs

        dnn_data = np.array([np.sin(yaw), np.cos(yaw), diff_x_k, diff_y_k, diff_yaw_k, command_old[0], command_old[1]])
        dnn_data = np.minimum(np.maximum(dnn_data, self.data_min), self.data_max)

        # Scale data

        if self.type_scaling == 1:
            dnn_data = (dnn_data - self.mu) / self.sigma
        else:
            if self.type_scaling == 2:
                dnn_data = 2 * (dnn_data - self.data_min) / (self.data_max - self.data_min) - 1

        # Split into input and output

        dnn_input = dnn_data[:self.num_inputs]  # 5
        dnn_output = dnn_data[self.num_inputs:]  # 2
        dnn_input = torch.from_numpy(dnn_input).float()  # cpu
        dnn_output = torch.from_numpy(dnn_output).float()

        # Update the cache with this sample:
        self.cache.update((dnn_input, dnn_output))

        # Train the DNN only when cache has enough samples.
        if self.cache.n_samples >= self.cache.size:
            x_batch, y_batch = self.cache.load_batch()  # tensor

            # Run the training
            self.agent.train(x_batch, y_batch)

        # Update the buffer with the incoming sample.
        self.buffer.reservoir_update(dnn_input, dnn_output)

    def save(self, dnn_name):
        torch.save(self.model.state_dict(), 'models/' + dnn_name + '.pth')

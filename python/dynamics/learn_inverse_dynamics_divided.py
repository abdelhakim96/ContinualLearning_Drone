import torch
import pandas as pd
import numpy as np
import math
import csv
import time

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from models.model import MLP
import matplotlib.pyplot as plt

# Parameters

device = 'cpu'

typeAction = 1  # 1 - train, 2 - test, 3 - train more
typeScaling = 1  # 0 - no scaling, 1 - standardization, 2 - normalization

numHiddenUnits1 = [64, 64, 64]
numHiddenUnits2 = [8, 8]
maxEpochs = 1000

datasetName = 'unicycle_dynamics_random_bound100'

trainingPercentage = 0.9
validationPercentage = 0.1


class UnicycleDynamicsYDataset(Dataset):
    def __init__(self, data):

        # Define inputs and outputs to the network

        self.input = data[['yaw', 'v', 'w', 'diff_x', 'diff_y']].values
        self.output = data[['tau_y']].values

    def num_inputs(self):
        return len(self.input[0])

    def num_outputs(self):
        return len(self.output[0])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.output[index]


class UnicycleDynamicsZDataset(Dataset):
    def __init__(self, data):

        # Define inputs and outputs to the network

        self.input = data[['w', 'diff_yaw']].values
        self.output = data[['tau_z']].values

    def num_inputs(self):
        return len(self.input[0])

    def num_outputs(self):
        return len(self.output[0])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.output[index]


def plot_histograms(data):
    plt.figure(1)
    plt.title('yaw')
    plt.hist(data[['yaw']].values)
    plt.figure(3)
    plt.title('diff_x')
    plt.hist(data[['diff_x']].values)
    plt.figure(4)
    plt.title('diff_y')
    plt.hist(data[['diff_y']].values)
    plt.figure(5)
    plt.title('diff_yaw')
    plt.hist(data[['diff_yaw']].values)
    plt.figure(6)
    plt.title('v')
    plt.hist(data[['v']].values)
    plt.figure(7)
    plt.title('w')
    plt.hist(data[['w']].values)
    plt.figure(8)
    plt.title('tau_y')
    plt.hist(data[['tau_y']].values)
    plt.figure(9)
    plt.title('tau_z')
    plt.hist(data[['tau_z']].values)
    plt.show()


def train_model(network_name, dataset, num_hidden_units):

    start = time.time()

    # Split into train, validation and test datasets

    num_train_samples = round(trainingPercentage * len(dataset))
    num_validation_samples = round(validationPercentage * len(dataset))
    train, validation = random_split(dataset, [num_train_samples, num_validation_samples])

    train_loader = DataLoader(train, batch_size=1000, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation, batch_size=100, shuffle=True, pin_memory=True)

    # Initialize the MLP

    model = MLP(dataset.num_inputs(), num_hidden_units, dataset.num_outputs()).to(device)

    # Define the loss function and optimizer

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run the training loop

    train_losses = []
    validation_losses = []
    writer = SummaryWriter()
    time_start = time.time()
    for epoch in range(0, maxEpochs):
        print(f'Epoch {epoch + 1}/{maxEpochs}')

        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = model(x)  # perform forward pass
            loss = criterion(y_pred, y)  # compute loss
            optimizer.zero_grad()  # zero the gradients
            loss.backward()  # perform backward pass
            optimizer.step()  # perform optimization
            total_loss += loss.item()
            writer.add_scalar("Loss/train", loss, epoch)
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        total_loss = 0
        for _, (x, y) in enumerate(validation_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            with torch.no_grad():
                y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            writer.add_scalar('Loss/val', loss, epoch)
        validation_losses.append(total_loss / len(validation_loader))

        time_end = time.time()
        elapsed_time = time_end - time_start
        print('Time: ' +
              time.strftime('%H:%M:%S', time.gmtime(elapsed_time)) + ' + ' +
              time.strftime('%H:%M:%S', time.gmtime(elapsed_time / (epoch + 1) * (maxEpochs - epoch - 1))) + ' = ' +
              time.strftime('%H:%M:%S', time.gmtime(elapsed_time / (epoch + 1) * maxEpochs)))

    print('Scaled Train Loss:', train_losses[-1])
    print('Scaled Validation Loss:', validation_losses[-1])

    writer.flush()

    end = time.time()
    print('Total time: ' + time.strftime('%H:%M:%S', time.gmtime(end - start)))

    # Save model

    torch.save(model.state_dict(), '../models/' + network_name + '.pth')
    torch.save(optimizer.state_dict(), '../models/optimiser_' + network_name + '.pth')

    # Plot losses

    plt.figure()
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="train")
    plt.plot(validation_losses, label="validation")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()


if __name__ == '__main__':

    # Load log file

    data = pd.read_csv('../data/log_' + datasetName + '.csv', usecols=['x', 'y', 'yaw', 'v', 'w', 'tau_y', 'tau_z'])

    # Process data

    # translation invariant: input = [x(k + 2) - x(k), y(k + 2) - y(k)];
    data[['diff_x', 'diff_y']] = data[['x', 'y']].values[2:] - data[['x', 'y']][:-2]

    # make orientation periodic with sin and cos
    data['sin'] = np.sin(data['yaw'].values)
    data['cos'] = np.cos(data['yaw'].values)

    # rotation invariant
    data[['diff_yaw']] = data[['yaw']].values[2:] - data[['yaw']][:-2]

    #############################################################
    # generate fake data
    dt = 0.001
    r = 0.3
    m = 20
    i_y = m * r ** 2
    i_z = m * r ** 2 / 2

    data['yaw'] = np.random.rand(len(data['x'])) * 2 * np.pi - np.pi
    data['v'] = np.random.rand(len(data['x'])) * 9 - 3
    data['w'] = np.random.rand(len(data['x'])) * 180 - 90
    data['diff_x'] = np.random.rand(len(data['x'])) * 0.024 - 0.012
    data['diff_y'] = np.random.rand(len(data['x'])) * 0.024 - 0.012
    data['diff_yaw'] = np.random.rand(len(data['x'])) * 0.36 - 0.18

    # data['diff_yaw'] = np.multiply(data['diff_yaw'].values % np.pi, np.sign(data['diff_yaw'].values))

    # clip velocities
    data['v'] = np.clip(data['v'].values, -3, 6)
    data['w'] = np.clip(data['w'].values, -90, 90)

    data['tau_y'] = -(i_y * (
            np.multiply(data['diff_x'].values, np.cos(data['yaw'].values + data['w'].values * dt)) +
            np.multiply(data['diff_y'].values, np.sin(data['yaw'].values + data['w'].values * dt)) +
            data['v'].values * dt + np.multiply(data['v'].values * dt, np.cos(data['w'].values * dt)))) / \
            (r * (dt ** 2)) - 110000 / r * data['v'].values
    data['tau_z'] = i_z * (data['diff_yaw'].values - 2 * data['w'].values * dt) / (dt ** 2) - 110000 * data['w'].values
    #######################################################

    # make rotation difference periodic between -pi and pi
    data[['diff_yaw']] -= \
        (np.abs(data[['diff_yaw']].values) > np.pi) * 2 * np.pi * np.sign(data[['diff_yaw']].values)

    # clip control inputs
    data['tau_y'] = np.clip(data['tau_y'].values, -100, 100)
    data['tau_z'] = np.clip(data['tau_z'].values, -100, 100)

    # remove NaN values
    data.drop(data.tail(2).index, inplace=True)

    # Save dataset

    header = ['yaw', 'v', 'w', 'diff_x', 'diff_y', 'diff_yaw', 'tau_y', 'tau_z']
    data[header].to_csv('../data/dataset_' + datasetName + '.csv', index=False)

    # Plot histograms

    # plot_histograms(data)

    # Data scaling

    mu = data.mean(0)
    sigma = data.std(0)
    data_min = data.min(0)
    data_max = data.max(0)
    if typeScaling == 1:
        data = (data - mu) / sigma
    if typeScaling == 2:
        data = 2 * (data - data_min) / (data_max - data_min) - 1

    # Save data information

    with open('../models/data_properties_dynamics.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['property'] + header)
        writer.writerow(['mu'] + list(mu[header]))
        writer.writerow(['sigma'] + list(sigma[header]))
        writer.writerow(['min'] + list(data_min[header]))
        writer.writerow(['max'] + list(data_max[header]))
        writer.writerow([''])
        writer.writerow(['type_scaling', typeScaling])

    networkName = 'dnn_dynamics_y_' + 'x'.join([str(s) for s in numHiddenUnits1])
    dataset = UnicycleDynamicsYDataset(data)
    if typeAction == 1:
        train_model(networkName, dataset, numHiddenUnits1)

    # networkName = 'dnn_dynamics_z_' + 'x'.join([str(s) for s in numHiddenUnits2])
    # dataset = UnicycleDynamicsZDataset(data)
    # if typeAction == 1:
    #     train_model(networkName, dataset, numHiddenUnits2)

    plt.show()

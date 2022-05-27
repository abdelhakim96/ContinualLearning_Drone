import torch
import pandas as pd
import numpy as np
import math
import csv
import time

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from models.model import MLP

# Parameters

device = 'cpu'

typeAction = 1  # 1 - train, 2 - test, 3 - train more
typeScaling = 1  # 0 - no scaling, 1 - standardization, 2 - normalization

numHiddenUnits1 = [16]
numHiddenUnits2 = [4]
maxEpochs = 100

datasetName = 'unicycle_kinematics_random_bound20_100'

trainingPercentage = 0.9
validationPercentage = 0.1


class UnicycleKinematicsYDataset(Dataset):
    def __init__(self, data):

        # Define inputs and outputs to the network

        self.input = data[['sin', 'cos', 'diff_x', 'diff_y']].values
        self.output = data[['w_y']].values

        # Save data information

    def num_inputs(self):
        return len(self.input[0])

    def num_outputs(self):
        return len(self.output[0])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.output[index]


class UnicycleKinematicsZDataset(Dataset):
    def __init__(self, data):

        # Define inputs and outputs to the network

        self.input = data[['diff_yaw']].values
        self.output = data[['w_z']].values

        # Save data information

    def num_inputs(self):
        return len(self.input[0])

    def num_outputs(self):
        return len(self.output[0])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.output[index]


def train_models(network_name, dataset, num_hidden_units):

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


if __name__ == '__main__':

    # Load dataset

    data = pd.read_csv('../data/log_' + datasetName + '.csv', usecols=['x', 'y', 'yaw', 'w_y', 'w_z'])

    # Process data

    # translation invariant: input = [x(k + 1) - x(k), y(k + 1) - y(k)];
    data[['diff_x', 'diff_y']] = data[['x', 'y']].values[1:] - data[['x', 'y']][:-1]

    # make orientation periodic with sin and cos
    data['sin'] = np.sin(data['yaw'].values)
    data['cos'] = np.cos(data['yaw'].values)

    # rotation invariant
    data[['diff_yaw']] = data[['yaw']].values[1:] - data[['yaw']][:-1]
    data[['diff_yaw']] -= \
        (np.abs(data[['diff_yaw']].values) > math.pi) * 2 * math.pi * np.sign(data[['diff_yaw']].values)

    # remove NaN values
    data.drop(data.tail(2).index, inplace=True)

    # Save dataset

    header = ['sin', 'cos', 'diff_x', 'diff_y', 'diff_yaw', 'w_y', 'w_z']
    data[header].to_csv('../data/dataset_' + datasetName + '.csv', index=False)

    # Data scaling

    mu = data.mean(0)
    sigma = data.std(0)
    data_min = data.min(0)
    data_max = data.max(0)
    if typeScaling == 1:
        data = (data - mu) / sigma
    if typeScaling == 2:
        data = 2 * (data - data_min) / (data_max - data_min) - 1

    with open('../models/data_properties_kinematics.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['property'] + header)
        writer.writerow(['mu'] + list(mu[header]))
        writer.writerow(['sigma'] + list(sigma[header]))
        writer.writerow(['min'] + list(data_min[header]))
        writer.writerow(['max'] + list(data_max[header]))
        writer.writerow([''])
        writer.writerow(['type_scaling', typeScaling])

    networkName = 'dnn_kinematics_y_' + 'x'.join([str(s) for s in numHiddenUnits1])
    dataset = UnicycleKinematicsYDataset(data)
    train_models(networkName, dataset, numHiddenUnits1)

    networkName = 'dnn_kinematics_z_' + 'x'.join([str(s) for s in numHiddenUnits2])
    dataset = UnicycleKinematicsZDataset(data)
    train_models(networkName, dataset, numHiddenUnits2)

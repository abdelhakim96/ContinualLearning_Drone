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

device = 'cpu'  # 'cpu' or 'cuda'

typeAction = 1  # 1 - train, 2 - test, 3 - train more
typeScaling = 1  # 0 - no scaling, 1 - standardization, 2 - normalization

numHiddenUnits = [100, 10]
maxEpochs = 1000

datasetName = 'unicycle_dynamics_artificial'

trainingPercentage = 0.7
validationPercentage = 0.3

networkName = 'dnn_dynamics_' + 'x'.join([str(s) for s in numHiddenUnits])

class UnicycleDataset(Dataset):
    def __init__(self, dataset_name, type_scaling):

        # Load dataset

        #data = pd.read_csv('../data/log_' + dataset_name + '.csv', usecols=['x', 'y', 'yaw', 'v', 'w', 'tau_y', 'tau_z'])
        data = pd.read_csv('../data/log_' + dataset_name + '.csv', usecols=['yaw', 'e_x', 'e_y', 'e_yaw', 'v', 'w', 'tau_y', 'tau_z'])

        # Process data

        # Position translation invariant: input = [x(k + 2) - x(k), y(k + 2) - y(k)];
        # data['diff_x'] = np.zeros(data['x'].shape)
        # data['diff_y'] = np.zeros(data['y'].shape)
        # data['diff_x'][:-2] = data['x'].values[2:] - data['x'].values[:-2]
        # data['diff_y'][:-2] = data['y'].values[2:] - data['y'].values[:-2]

        # Make orientation periodic with sin and cos
        data['sin'] = np.sin(data['yaw'])
        data['cos'] = np.cos(data['yaw'])

        # # rotation invariant
        # data[['diff_yaw']] = data[['yaw']].values[2:] - data[['yaw']][:-2]
        # data[['diff_yaw']] -= \
        #     (np.abs(data[['diff_yaw']].values) > math.pi) * 2 * math.pi * np.sign(data[['diff_yaw']].values)

        data[['diff_x', 'diff_y', 'diff_yaw']] = data[['e_x', 'e_y', 'e_yaw']]

        # remove NaN values
        data.drop(data.tail(2).index, inplace=True)

        # Save dataset
        header_input = ['diff_x', 'diff_y', 'sin', 'cos', 'v', 'w']
        header_output = ['tau_y', 'tau_z']
        header = header_input + header_output
        data[header].to_csv('../data/dataset_' + dataset_name + '.csv', index=False)

        # Plot histograms

        # plt.figure(1)
        # plt.title('diff_x')
        # plt.hist(data['diff_x'])
        # plt.figure(2)
        # plt.title('diff_y')
        # plt.hist(data[['diff_y']])
        # plt.figure(3)
        # plt.title('yaw')
        # plt.hist(data[['yaw']])
        # plt.figure(4)
        # plt.title('v')
        # plt.hist(data[['v']])
        # plt.figure(5)
        # plt.title('w')
        # plt.hist(data[['w']])
        # plt.figure(6)
        # plt.title('tau_y')
        # plt.hist(data[['tau_y']])
        # plt.figure(7)
        # plt.title('tau_z')
        # plt.hist(data[['tau_z']])
        # plt.show()

        # Data scaling

        mu = data.mean(0)
        sigma = data.std(0)
        data_min = data.min(0)
        data_max = data.max(0)
        if type_scaling == 1:
            data = (data - mu) / sigma
        if type_scaling == 2:
            data = 2 * (data - data_min) / (data_max - data_min) - 1

        # Define inputs and outputs to the network

        self.input = data[header_input].values
        self.output = data[header_output].values

        # Save data information

        with open('../models/properties_' + networkName + '.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['property'] + header)
            writer.writerow(['mu'] + list(mu[header]))
            writer.writerow(['sigma'] + list(sigma[header]))
            writer.writerow(['min'] + list(data_min[header]))
            writer.writerow(['max'] + list(data_max[header]))
            writer.writerow([''])
            writer.writerow(['type_scaling', type_scaling])
            writer.writerow(['num_inputs', len(self.input[0])])
            writer.writerow(['num_outputs', len(self.output[0])])

    def num_inputs(self):
        return len(self.input[0])

    def num_outputs(self):
        return len(self.output[0])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.output[index]


if __name__ == '__main__':
    start = time.time()

    # Define dataset

    dataset = UnicycleDataset(datasetName, typeScaling)

    # Split into train, validation and test datasets

    trainSamples = round(trainingPercentage * len(dataset))
    validationSamples = round(validationPercentage * len(dataset))
    testSamples = len(dataset) - (trainSamples + validationSamples)
    train, validation, test = random_split(dataset, [trainSamples, validationSamples, testSamples])

    trainLoader = DataLoader(train, batch_size=1000, shuffle=True, pin_memory=True)
    validationLoader = DataLoader(validation, batch_size=100, shuffle=True, pin_memory=True)
    #testLoader = DataLoader(test, batch_size=100, shuffle=True, pin_memory=True)

    # Initialize the MLP

    model = MLP(dataset.num_inputs(), numHiddenUnits, dataset.num_outputs()).to(device)

    # Define the loss function and optimizer

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run the training loop

    train_losses = []
    validation_losses = []
    writer = SummaryWriter()
    time_start = time.time()
    for epoch in range(0, maxEpochs):
        print(f'Epoch {epoch + 1}/{maxEpochs}')

        model.train()
        totalLoss = 0
        for i, (x, y) in enumerate(trainLoader):
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = model(x)  # perform forward pass
            loss = criterion(y_pred, y)  # compute loss
            optimizer.zero_grad()  # zero the gradients
            loss.backward()  # perform backward pass
            optimizer.step()  # perform optimization
            totalLoss += loss.item()
            writer.add_scalar("Loss/train", loss, epoch)
        train_losses.append(totalLoss / len(trainLoader))

        model.eval()
        totalLoss = 0
        for _, (x, y) in enumerate(validationLoader):
            x = x.float().to(device)
            y = y.float().to(device)
            with torch.no_grad():
                y_pred = model(x)
            loss = criterion(y_pred, y)
            totalLoss += loss.item()
            writer.add_scalar('Loss/val', loss, epoch)
        validation_losses.append(totalLoss / len(validationLoader))

        time_end = time.time()
        elapsedTime = time_end - time_start
        print('Time: ' +
              time.strftime('%H:%M:%S', time.gmtime(elapsedTime)) + ' + ' +
              time.strftime('%H:%M:%S', time.gmtime(elapsedTime / (epoch + 1) * (maxEpochs - epoch - 1))) + ' = ' +
              time.strftime('%H:%M:%S', time.gmtime(elapsedTime / (epoch + 1) * maxEpochs)))

    print('Train Loss:', train_losses[-1])
    print('Validation Loss:', validation_losses[-1])

    writer.flush()

    end = time.time()
    print('Total time: ' + time.strftime('%H:%M:%S', time.gmtime(end - start)))

    # Save model

    torch.save(model.state_dict(), '../models/' + networkName + '.pth')
    torch.save(optimizer.state_dict(), '../models/optimiser_' + networkName + '.pth')

    # Plot losses

    plt.figure()
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="train")
    plt.plot(validation_losses, label="validation")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()

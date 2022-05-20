import torch
import pandas as pd
import numpy as np
import math
import pickle
import time

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from model import MLP


# Parameters

device = 'cpu'

typeAction = 1  # 1 - train, 2 - test, 3 - train more
typeScaling = 1  # 0 - no scaling, 1 - standardization, 2 - normalization

numHiddenUnits = [32, 32, 32]
maxEpochs = 100

datasetName = 'unicycle_dynamics_random'

trainingPercentage = 0.9
validationPercentage = 0.1


class UnicycleDataset(Dataset):
    def __init__(self, dataset_name, type_scaling):

        # Load dataset

        data = pd.read_csv('data/log_' + dataset_name + '.csv', usecols=['x', 'y', 'yaw', 'v', 'w', 'tau_y', 'tau_z'])

        # Process data

        # translation invariant: input = [x(k + 2) - x(k), y(k + 2) - y(k)];
        data[['diff_x', 'diff_y']] = data[['x', 'y']].values[2:] - data[['x', 'y']][:-2]

        # make orientation periodic with sin and cos
        data['sin'] = np.sin(data['yaw'].values)
        data['cos'] = np.cos(data['yaw'].values)

        # rotation invariant
        data[['diff_yaw']] = data[['yaw']].values[2:] - data[['yaw']][:-2]
        data[['diff_yaw']] -= \
            (np.abs(data[['diff_yaw']].values) > math.pi) * 2 * math.pi * np.sign(data[['diff_yaw']].values)

        # future angular velocity
        #data.loc[0:len(data[['w']].values[1:]) - 1, 'w(k+1)'] = data[['w']].values[1:]

        # remove NaN values
        data.drop(data.tail(2).index, inplace=True)

        # Save dataset

        header = ['sin', 'cos', 'v', 'w', 'diff_x', 'diff_y', 'diff_yaw', 'tau_y', 'tau_z']
        dataframe = pd.DataFrame(data[header].values, columns=header)
        dataframe.to_csv('data/dataset_' + dataset_name + '.csv', index=False)

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

        self.input = data[['sin', 'cos', 'v', 'w', 'diff_x', 'diff_y', 'diff_yaw']].values
        self.output = data[['tau_y', 'tau_z']].values

        # Save data information

        dictionary = {'type_scaling': type_scaling,
                      'mu': mu,
                      'sigma': sigma,
                      'data_min': data_min,
                      'data_max': data_max,
                      'num_inputs': len(self.input[0]),
                      'num_outputs': len(self.output[0])}
        with open('models/unicycle_dynamics.pkl', 'wb') as f:
            pickle.dump(dictionary, f)

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

    # Parameters

    networkName = 'dnn_dynamics_' + 'x'.join([str(s) for s in numHiddenUnits])

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

    torch.save(model.state_dict(), 'models/' + networkName + '.pth')
    torch.save(optimizer.state_dict(), 'models/optimiser_' + networkName + '.pth')

    # Plot losses

    # plt.figure()
    # plt.title("Training and Validation Loss")
    # plt.plot(train_losses, label="train")
    # plt.plot(validation_losses, label="validation")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.grid()
    # plt.legend()
    # plt.show()

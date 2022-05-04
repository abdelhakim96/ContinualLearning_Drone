import torch
import pandas as pd
import numpy as np
import pickle
import time

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class SphereSimpleDataset(Dataset):
    def __init__(self, dataset_name, type_scaling):

        # Load dataset

        data = pd.read_csv('data/log_' + dataset_name + '.csv', usecols=['x', 'y', 'yaw', 'w_x', 'w_y', 'w_z'])

        # Process data

        # translation invariant: input = [x(k + r_x) - x(k), y(k + r_y) - y(k)];
        data[['diff_x(k)', 'diff_y(k)']] = data[['x', 'y']].values[1:] - data[['x', 'y']][:-1]
        #data[['diff_x(k+1)', 'diff_y(k+1)']] = data[['x', 'y']].values[2:] - data[['x', 'y']][:-2]

        # make orientation periodic with sin and cos
        data['sin'] = np.sin(data['yaw'].values)
        data['cos'] = np.cos(data['yaw'].values)

        # remove NaN values
        data.drop(data.tail(2).index, inplace=True)

        # Save dataset

        header = ['I:diff_x(k)', 'I:diff_y(k)', 'I:sin', 'I:cos', 'O:w_x', 'O:w_y', 'O:w_z']
        dataframe = pd.DataFrame(data[['diff_x(k)', 'diff_y(k)', 'sin', 'cos', 'w_x', 'w_y', 'w_z']].values,
                                 columns=header)
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

        self.input = data[['diff_x(k)', 'diff_y(k)', 'sin', 'cos']].values
        self.output = data[['w_x', 'w_y', 'w_z']].values

        # Save data information

        dictionary = {'type_scaling': type_scaling,
                      'mu': mu,
                      'sigma': sigma,
                      'data_min': data_min,
                      'data_max': data_max,
                      'num_inputs': len(self.input[0]),
                      'num_outputs': len(self.output[0])}
        with open('models/sphere_simple.pkl', 'wb') as f:
            pickle.dump(dictionary, f)

    def num_inputs(self):
        return len(self.input[0])

    def num_outputs(self):
        return len(self.output[0])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.output[index]


class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden_units, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, num_hidden_units[0]),
            nn.Tanh(),
            nn.Linear(num_hidden_units[0], num_outputs)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    start = time.time()

    # Parameters

    typeAction = 1  # 1 - train, 2 - test, 3 - train more
    typeScaling = 1  # 0 - no scaling, 1 - standardization, 2 - normalization

    numHiddenUnits = [8]
    maxEpochs = 100

    datasetName = 'sphere_simple_random'
    networkName = 'dnn_simple_' + 'x'.join([str(s) for s in numHiddenUnits])

    trainingPercentage = 0.9
    validationPercentage = 0.1

    # Define dataset

    dataset = SphereSimpleDataset(datasetName, typeScaling)

    # Split into train, validation and test datasets

    trainSamples = round(trainingPercentage * len(dataset))
    validationSamples = round(validationPercentage * len(dataset))
    testSamples = len(dataset) - (trainSamples + validationSamples)
    train, validation, test = random_split(dataset, [trainSamples, validationSamples, testSamples])

    trainLoader = DataLoader(train, batch_size=512, shuffle=False)
    validationLoader = DataLoader(validation, batch_size=256, shuffle=False)
    testLoader = DataLoader(test, batch_size=128, shuffle=False)

    # Initialize the MLP

    model = MLP(dataset.num_inputs(), numHiddenUnits, dataset.num_outputs()).cuda()

    # Define the loss function and optimizer

    criterion = nn.MSELoss(reduction='mean')
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
            x = x.float().cuda()
            y = y.float().cuda()
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
            x = x.float().cuda()
            y = y.float().cuda()
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

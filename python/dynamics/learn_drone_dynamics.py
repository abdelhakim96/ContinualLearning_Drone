import torch
import pandas as pd
import numpy as np
import math
import csv
import time
import os

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from models.model import MLP
import matplotlib.pyplot as plt

# Parameters
device = 'cpu'  # 'cpu' or 'cuda'

numHiddenUnits = [256, 256, 256]
maxEpochs = 500

datasetPath = 'data/drone/100hz_real.csv'
networkName = 'dnn_drone_dynamics_' + 'x'.join([str(s) for s in numHiddenUnits])

class DroneDataset(Dataset):
    def __init__(self, csv_path, type_scaling=1):
        # Load dataset
        # Columns in CSV: time,dt,pos_x,pos_y,pos_z,vel_body_x,vel_body_y,vel_body_z,acc_x,acc_y,acc_z,
        # ang_vel_x,ang_vel_y,ang_vel_z,ang_acc_x,ang_acc_y,ang_acc_z,roll,pitch,yaw,qw,qx,qy,qz,
        # pwm_1,pwm_2,pwm_3,pwm_4,total_thrust,cmd_thrust,cmd_roll,cmd_pitch,cmd_yaw,trajectory_type
        
        data = pd.read_csv(csv_path)

        # Physics-informed transformations
        # Inputs: {cos(psi[k]), sin(psi[k]), vx[k], vy[k], x[k+1]-x[k], y[k+1]-y[k], z[k+1]-z[k], psi[k+1]-psi[k]}
        # Outputs: {vz[k], phi[k], theta[k], omega_psi[k]}

        df = pd.DataFrame()
        
        # Orientations
        df['cos_yaw'] = np.cos(data['yaw'])
        df['sin_yaw'] = np.sin(data['yaw'])
        
        # Velocities
        df['vx'] = data['vel_body_x']
        df['vy'] = data['vel_body_y']
        
        # Current attitude
        df['roll'] = data['roll']
        df['pitch'] = data['pitch']
        
        # State transitions (Horizon +1)
        df['diff_x'] = data['pos_x'].shift(-1) - data['pos_x']
        df['diff_y'] = data['pos_y'].shift(-1) - data['pos_y']
        df['diff_z'] = data['pos_z'].shift(-1) - data['pos_z']
        
        # Yaw transition with wrap-around
        diff_yaw = data['yaw'].shift(-1) - data['yaw']
        diff_yaw = (diff_yaw + np.pi) % (2 * np.pi) - np.pi
        df['diff_yaw'] = diff_yaw
        
        # Targets (at time k)
        df['cmd_thrust'] = data['cmd_thrust']
        df['cmd_roll'] = data['cmd_roll']
        df['cmd_pitch'] = data['cmd_pitch']
        df['cmd_yaw'] = data['cmd_yaw']
        
        # Drop last row due to shift(-1)
        df.dropna(inplace=True)

        header_input = ['cos_yaw', 'sin_yaw', 'vx', 'vy', 'roll', 'pitch', 'diff_x', 'diff_y', 'diff_z', 'diff_yaw']
        header_output = ['cmd_thrust', 'cmd_roll', 'cmd_pitch', 'cmd_yaw']
        header = header_input + header_output

        # Data scaling
        self.mu = df[header].mean(0)
        self.sigma = df[header].std(0)
        self.data_min = df[header].min(0)
        self.data_max = df[header].max(0)
        
        scaled_data = df[header].copy()
        if type_scaling == 1:
            scaled_data = (scaled_data - self.mu) / self.sigma
        elif type_scaling == 2:
            scaled_data = 2 * (scaled_data - self.data_min) / (self.data_max - self.data_min) - 1

        self.input = scaled_data[header_input].values
        self.output = scaled_data[header_output].values

        # Save properties for inference
        os.makedirs('models', exist_ok=True)
        with open('models/properties_' + networkName + '.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['property'] + header)
            writer.writerow(['mu'] + list(self.mu))
            writer.writerow(['sigma'] + list(self.sigma))
            writer.writerow(['min'] + list(self.data_min))
            writer.writerow(['max'] + list(self.data_max))
            writer.writerow([''])
            writer.writerow(['type_scaling', type_scaling])
            writer.writerow(['num_inputs', len(header_input)])
            writer.writerow(['num_outputs', len(header_output)])

    def num_inputs(self):
        return self.input.shape[1]

    def num_outputs(self):
        return self.output.shape[1]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.output[index]

if __name__ == '__main__':
    start_time_total = time.time()

    # Define dataset
    dataset = DroneDataset(datasetPath, type_scaling=1)

    # Split into train and validation
    trainPercentage = 0.8
    trainSamples = int(trainPercentage * len(dataset))
    valSamples = len(dataset) - trainSamples
    train, val = random_split(dataset, [trainSamples, valSamples])

    trainLoader = DataLoader(train, batch_size=128, shuffle=True)
    valLoader = DataLoader(val, batch_size=128, shuffle=False)

    # Initialize the MLP
    model = MLP(dataset.num_inputs(), numHiddenUnits, dataset.num_outputs()).to(device)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {maxEpochs} epochs...")
    for epoch in range(maxEpochs):
        model.train()
        total_train_loss = 0
        for x, y in trainLoader:
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(trainLoader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in valLoader:
                x, y = x.float().to(device), y.float().to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(valLoader)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{maxEpochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), 'models/' + networkName + '.pth')
    print(f"Model saved to models/{networkName}.pth")

    # Final timing
    end_time_total = time.time()
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(end_time_total - start_time_total))}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('drone_training_loss.png')
    print("Loss plot saved to drone_training_loss.png")
    # plt.show() # Can't show in headless environment easily

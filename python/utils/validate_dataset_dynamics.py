import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Parameters

datasetName = 'unicycle_dynamics_random_bound100'

interval1 = 00000
interval2 = 1000000

# Constants

r = 0.3  # initial wheel radius
m = 20  # initial mass
i_y = m * r ** 2  # initial wheel rotational inertia around y-axis
i_z = m * r ** 2 / 2  # initial wheel rotational inertia around z-axis
dt = 0.001  # time difference

# Load log file

data = pd.read_csv('../data/log_' + datasetName + '.csv', usecols=['x', 'y', 'yaw', 'v', 'w', 'tau_y', 'tau_z'])
data = data.iloc[interval1:interval2, :]

# Process data

data[['diff_x']] = data[['x']].values[2:] - data[['x']][:-2]
data[['diff_y']] = data[['y']].values[2:] - data[['y']][:-2]
data[['diff_yaw']] = data[['yaw']].values[2:] - data[['yaw']][:-2]
data[['diff_yaw']] -= (np.abs(data[['diff_yaw']]) > np.pi) * 2 * np.pi * np.sign(data[['diff_yaw']])
data[['w_y']] = data[['v']].values / r
data[['w_z']] = data[['w']].values

# Remove samples with maximum speed

data = data[np.logical_and(np.logical_and(data['w_y'] > -9.9, data['w_y'] < 19.9),
                           np.logical_and(data['w_z'] > -89.9, data['w_z'] < 89.9))]

# Variables

yaw = data[['yaw']].values[:-2]
diff_x = data[['diff_x']].values[:-2]
diff_y = data[['diff_y']].values[:-2]
diff_yaw = data[['diff_yaw']].values[:-2]
w_y = data[['w_y']].values[:-2]
w_z = data[['w_z']].values[:-2]

tau_y1 = data[['tau_y']].values[:-2]
tau_z1 = data[['tau_z']].values[:-2]

# Analytical inverse

inverse_tau_y1 = (i_y * (np.multiply(diff_x, np.cos(yaw + w_z * dt)) + np.multiply(diff_y, np.sin(yaw + w_z * dt)) -
                         r * w_y * dt - r * w_y * dt * np.cos(w_z * dt))) / (r * (dt ** 2))
inverse_tau_z1 = i_z * (diff_yaw - 2 * w_z * dt) / (dt ** 2)

# Load dataset file

data = pd.read_csv('../data/dataset_' + datasetName + '.csv',
                   usecols=['sin', 'cos', 'v', 'w', 'diff_x', 'diff_y', 'diff_yaw', 'tau_y', 'tau_z'])
data = data.iloc[interval1:interval2 - 2, :]

# Variables

data[['yaw']] = np.arctan2(data[['sin']].values, data[['cos']].values)
data[['w_y']] = data[['v']].values / r
data[['w_z']] = data[['w']].values

# Remove samples with maximum speed

data = data[np.logical_and(np.logical_and(data['w_y'] > -9.9, data['w_y'] < 19.9),
                           np.logical_and(data['w_z'] > -89.9, data['w_z'] < 89.9))]

# Variables

yaw = data[['yaw']].values[:-2]
diff_x = data[['diff_x']].values[:-2]
diff_y = data[['diff_y']].values[:-2]
diff_yaw = data[['diff_yaw']].values[:-2]
w_y = data[['w_y']].values[:-2]
w_z = data[['w_z']].values[:-2]

tau_y2 = data[['tau_y']].values[:-2]
tau_z2 = data[['tau_z']].values[:-2]

# Analytical inverse

inverse_tau_y2 = (i_y * (diff_x * np.cos(yaw + w_z * dt) + diff_y * np.sin(yaw + w_z * dt) -
                         r * w_y * dt - r * w_y * dt * np.cos(w_z * dt))) / (r * (dt ** 2))
inverse_tau_z2 = i_z * (diff_yaw - 2 * w_z * dt) / (dt ** 2)

# damping to compensate for bounding commands

# inverse_tau_y -= 110000 * w_y
# inverse_tau_z -= 110000 * w_z

# Calculate difference

print([np.mean(np.abs(tau_y1 - inverse_tau_y1)), np.mean(np.abs(tau_z1 - inverse_tau_z1))])
print([np.mean(np.abs(tau_y2 - inverse_tau_y2)), np.mean(np.abs(tau_z2 - inverse_tau_z2))])

# Plot control signal

plt.figure(1)
plt.title('tau_y')
plt.plot(tau_y1, 'r', label="measured in log")
plt.plot(inverse_tau_y1, 'g', label="inverse from log")
plt.plot(tau_y2, 'b', label="measured in dataset")
plt.plot(inverse_tau_y2, 'y', label="inverse from dataset")
plt.plot(w_y, 'c', label="velocity")
plt.legend()

plt.figure(2)
plt.title('tau_z')
plt.plot(tau_z1, 'r', label="measured in log")
plt.plot(inverse_tau_z1, 'g', label="inverse from log")
plt.plot(tau_z2, 'b', label="measured in dataset")
plt.plot(inverse_tau_z2, 'y', label="inverse from dataset")
plt.plot(w_z, 'c', label="velocity")
plt.legend()
plt.show()

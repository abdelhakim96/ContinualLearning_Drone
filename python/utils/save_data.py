# Save results

import numpy as np
import pandas as pd
from utils.data_engineering import data_engineering, get_velocity3, get_velocity2
import matplotlib.pyplot as plt


def save_data_dynamics(t, trajectory, pose, command, name):
    dt = t[1] - t[0]

    # Log file

    velocity = get_velocity2(dt, pose)

    data = np.column_stack([t[:-2], trajectory[:-2], pose[:-2], velocity[:-1], command[:-2]])
    header = ['time', 'x_d', 'y_d', 'yaw_d', 'x', 'y', 'yaw', 'v', 'w', 'tau_y', 'tau_z']
    dataset = pd.DataFrame(data, columns=header)
    dataset.to_csv('data/log_' + name + '.csv', index=False)

    # Dataset file

    # data = data_engineering(dt, pose, command)
    #
    # header = ['diff_x(k)', 'diff_y(k)', 'sin_theta(k)', 'cos_theta(k)', 'v(k)', 'w(k)',  # input: state
    #           'diff_x(k+1)', 'diff_y(k+1)', 'diff_x(k+2)', 'diff_y(k+2)',  # input: future outputs
    #           'tau_y(k)', 'tau_z(k)']  # output: control inputs
    # dataset = pd.DataFrame(data[:-3], columns=header)
    # dataset.to_csv('data/dataset_' + name + '.csv', index=False)

    # translation invariant: input = [x(k + 2) - x(k), y(k + 2) - y(k)];
    dataset[['diff_x', 'diff_y']] = dataset[['x', 'y']].values[2:] - dataset[['x', 'y']][:-2]

    # rotation invariant
    dataset[['diff_yaw']] = dataset[['yaw']].values[2:] - dataset[['yaw']][:-2]
    dataset[['diff_yaw']] -= \
        (np.abs(dataset[['diff_yaw']].values) > np.pi) * 2 * np.pi * np.sign(dataset[['diff_yaw']].values)

    # Plot histograms

    plt.figure(10)
    plt.title('yaw')
    plt.hist(dataset[['yaw']].values)
    plt.figure(3)
    plt.title('diff_x')
    plt.hist(dataset[['diff_x']].values)
    plt.figure(4)
    plt.title('diff_y')
    plt.hist(dataset[['diff_y']].values)
    plt.figure(5)
    plt.title('diff_yaw')
    plt.hist(dataset[['diff_yaw']].values)
    plt.figure(6)
    plt.title('v')
    plt.hist(dataset[['v']].values)
    plt.figure(7)
    plt.title('w')
    plt.hist(dataset[['w']].values)
    plt.figure(8)
    plt.title('tau_y')
    plt.hist(dataset[['tau_y']].values)
    plt.figure(9)
    plt.title('tau_z')
    plt.hist(dataset[['tau_z']].values)
    plt.figure(1)
    plt.title('2D Trajectory')
    plt.plot(trajectory[0:10000, 0], trajectory[0:10000, 1], 'k--', label="desired")
    plt.plot(pose[0:10000, 0], pose[0:10000, 1], 'b', label="actual")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid()
    plt.show()


def save_data_kinematics(t, trajectory, pose, command, name):

    # Log file

    data = np.column_stack([t[:-1], trajectory[:-1], pose[:-1], command[:-1]])
    header = ['time', 'x_d', 'y_d', 'yaw_d', 'x', 'y', 'yaw', 'w_y', 'w_z']
    dataset = pd.DataFrame(data, columns=header)
    dataset.to_csv('data/log_' + name + '.csv', index=False)


def save_experiment_kinematics(t, trajectory, pose, command, command_inverse, name):

    # Log file

    data = np.column_stack([t[:-1], trajectory[:-1], pose[:-1], command[:-1], command_inverse[:-1]])
    header = ['time', 'x_d', 'y_d', 'yaw_d', 'x', 'y', 'yaw', 'w_y', 'w_z', 'w_y_inverse', 'w_z_inverse']
    dataset = pd.DataFrame(data, columns=header)
    dataset.to_csv('experiments/' + name + '.csv', index=False)


def save_experiment_kinematics1(t, trajectory, pose, command, command_dnn0, command_odnn, name):
    # Log file

    data = np.column_stack([t[:-1], trajectory[:-1], pose[:-1], command[:-1], command_dnn0[:-1], command_odnn[:-1]])
    header = ['time', 'x_d', 'y_d', 'yaw_d', 'x', 'y', 'yaw', 'w_y', 'w_z',
              'w_y_dnn0', 'w_z_dnn0', 'w_y_odnn', 'w_z_odnn']
    dataset = pd.DataFrame(data, columns=header)
    dataset.to_csv('experiments/' + name + '1.csv', index=False)

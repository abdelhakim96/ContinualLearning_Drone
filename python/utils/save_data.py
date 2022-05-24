# Save results

import numpy as np
import pandas as pd
from utils.data_engineering import data_engineering, get_velocity3, get_velocity2


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


def save_data_kinematics(t, trajectory, pose, command, name):
    dt = t[1] - t[0]

    # Log file

    data = np.column_stack([t[:-1], trajectory[:-1], pose[:-1], command[:-1]])
    header = ['time', 'x_d', 'y_d', 'yaw_d', 'x', 'y', 'yaw', 'w_y', 'w_z']
    dataset = pd.DataFrame(data, columns=header)
    dataset.to_csv('data/log_' + name + '.csv', index=False)

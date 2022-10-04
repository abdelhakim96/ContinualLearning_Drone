# Prepare data for training

import math
import numpy as np


def data_engineering(dt, pose, command):
    scaling = 1  # 0 - no scaling, 1 - standardization, 2 - normalization

    size = len(pose)

    # Translation invariant: input = [x(k + r_x) - x(k), y(k + r_y) - y(k)]
    translation = pose[1:, :2] - pose[:-1, :2]

    # Derive velocities
    velocity = get_velocity3(dt, pose)

    # Compose dataset
    data = np.zeros((size, 13))
    data[:-1, 0:2] = translation  # position translation
    data[:, 2] = np.sin(pose[:, 2])  # make orientation periodic with sin and cos
    data[:, 3] = np.cos(pose[:, 2])  # make orientation periodic with sin and cos
    data[:-1, 4] = velocity[:, 0]  # linear x velocity
    data[:-1, 5] = velocity[:, 1]  # linear y velocity
    data[:-1, 6] = velocity[:, 1]  # angular z velocity
    data[:-2, 7:9] = translation[1:]  # future translation at (k + 1)
    data[:-3, 9:11] = translation[2:]  # future translation at (k + 2)
    data[:, 11:13] = command  # control inputs1

    # Scaling
    mu = data.mean(0)
    sigma = data.std(0)
    data_min = data.min(0)
    data_max = data.max(0)
    if scaling == 1:
        data = (data - mu)/sigma
    if scaling == 2:
        data = 2 * (data - data_min)/(data_max - data_min) - 1

    data = np.row_stack([mu, sigma, data_min, data_max, np.nan * np.ones((1, 12)), data])
    data[4, 0] = scaling

    return data


def get_velocities2(dt, pose):
    size = len(pose)

    # Derive linear velocities

    vx = (pose[1:, 0] - pose[:-1, 0]) / dt
    vy = (pose[1:, 1] - pose[:-1, 1]) / dt
    direction = np.zeros((size - 1))
    for i in np.arange(size - 1):
        if (np.abs(math.atan2(vy[i], vx[i]) - pose[i, 2]) < math.pi / 4) or \
           (np.abs(math.atan2(vy[i], vx[i]) - pose[i, 2]) > 7 / 4 * math.pi):
            direction[i] = 1
        else:
            direction[i] = - 1
    v = direction * np.sqrt(vx ** 2 + vy ** 2)

    # Derive angular velocities

    w = (pose[1:, 2] - pose[:-1, 2])
    for i in np.arange(size - 1):
        if np.abs(w[i]) > math.pi:
            w[i] -= 2 * math.pi * np.sign(w[i])
    w /= dt

    return np.column_stack([v, w])


def get_velocity2(dt, pose):
    velocity = get_velocities2(dt, pose)
    return velocity[0, :]


def get_velocity3(dt, pose):
    size = len(pose)

    # Derive linear velocities

    vx = (pose[1:, 0] - pose[:-1, 0]) / dt
    vy = (pose[1:, 1] - pose[:-1, 1]) / dt

    # Derive angular velocities

    wz = (pose[1:, 2] - pose[:-1, 2])
    for i in np.arange(size - 1):
        if np.abs(wz[i]) > math.pi:
            wz[i] -= 2 * math.pi * np.sign(wz[i])
    wz /= dt

    return np.column_stack([vx, vy, wz])

# Prepare data for training

import math
import numpy as np


def data_engineering(t, pose, command):
    dt = t[1] - t[0]
    size = len(t)

    # Derive linear velocities

    vx = (pose[1:, 0] - pose[:-1, 0]) / dt
    vy = (pose[1:, 1] - pose[:-1, 1]) / dt
    direction = np.zeros((size - 1))
    for i in np.arange(size - 1):
        if (np.abs(math.atan2(vy[i], vx[i]) - pose[i + 1, 2]) < math.pi / 4) or \
                (np.abs(math.atan2(vy[i], vx[i]) - pose[i + 1, 2]) > 7 / 4 * math.pi):
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

    # Translation invariant: input = [x(k + r_x) - x(k), y(k + r_y) - y(k)]
    translation = pose[1:, :2] - pose[:-1, :2]

    # Compose dataset
    data = np.empty((size, 12))
    data[:-1, 0:2] = translation  #
    data[:, 2] = np.sin(pose[:, 2])  # make orientation periodic with sin and cos
    data[:, 3] = np.cos(pose[:, 2])  # make orientation periodic with sin and cos
    data[:-1, 4] = v  #
    data[:-1, 5] = w  #
    data[:-2, 6:8] = translation[1:]  #
    data[:-3, 8:10] = translation[2:]  #
    data[:, 10:12] = command  # control input
    return data

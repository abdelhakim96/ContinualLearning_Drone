# MAIN

import numpy as np
import time

from kinematics.pid_kinematics import PID
from kinematics.dnn_kinematics import DNN
from kinematics.inverse_kinematics import Inverse
from kinematics.unicycle_kinematics import Unicycle
from utils.save_data import save_data_kinematics
from utils.show import show_plots, show_animation

# Parameters

collect_data = False  # True - collect data, False - test performance
controller = 4  # 0 - random, 1 - PID, 2 - inverse, 3 - DNN0, 4 - DNN + CL
trajectory = 2  # 0 - random points, 1 - circular, 2 - 8-shaped, 3 - set-point, 4 - square-wave
uncertainty = 1  # internal uncertainty: 0 - no uncertainty, 1 - all parameters double, -1 - all parameters half;
# default = 1
disturbance = -10  # external disturbance: 0 - no disturbance, >0 - positive disturbance, <0 - negative disturbance
# default = -2
noise = 0  # measurement noise standard deviation: 0 - no noise, >0 - white noise
# default = 0.0001

animation = False  # True - enable online animation, False - disable online animation

dnn_name = 'dnn_kinematics_32'

k_end = 6000
dt = 0.001

if collect_data:
    controller = 0
    trajectory = 0
    uncertainty = 0
    disturbance = 0
    noise = 0
    k_end = 1000000

# Initial pose

x_init = 0
y_init = 0
yaw_init = 0

# Initialise arrays

pose = np.zeros((k_end + 1, 3))
pose[0, :] = [x_init, y_init, yaw_init]
pose_real = np.zeros((k_end + 1, 3))
pose_real[0, :] = [x_init, y_init, yaw_init]
command = np.zeros((k_end + 1, 2))
command_random = np.zeros((k_end + 1, 2))
command_pid = np.zeros((k_end + 1, 2))
command_dnn = np.zeros((k_end + 1, 2))
command_inverse = np.zeros((k_end + 1, 2))
t = np.linspace(0, dt * k_end, num=(k_end + 1))

# Generate trajectory

if trajectory == 0:  # random points (for collecting training data)
    reference = np.random.randn(k_end + 1, 3)
if trajectory == 1:  # circular
    reference = np.column_stack([2 * np.cos(2 * t), -2 * np.sin(2 * t), np.zeros((k_end + 1, 1))])
if trajectory == 2:  # 8-shaped
    t -= 0*np.pi/2
    reference = np.column_stack([
        4 / (3 - np.cos(2 * t)) * np.cos(t),
        4 / (3 - np.cos(2 * t)) * np.sin(-2 * t) / np.sqrt(2),
        np.zeros((k_end + 1, 1))])
if trajectory == 3:  # set-point
    reference = np.column_stack([2 * np.ones((k_end + 1, 1)), -2 * np.ones((k_end + 1, 1)), np.zeros((k_end + 1, 1))])
if trajectory == 4:  # square-wave
    d = np.fix(t)
    d = d % 4
    b1 = np.fix(d / 2)
    b0 = d - 2 * b1
    reference = np.column_stack([b1, b0, np.zeros((k_end + 1, 1))])

unicycle = Unicycle(dt, [x_init, y_init, yaw_init])

pid = PID(dt)
dnn = DNN(dt, dnn_name)
inverse = Inverse(unicycle)

# Main loop

time_pid = 0
time_inverse = 0
time_dnn0 = 0
time_dnn = 0
for k in range(1, k_end - 1):

    # Unicycle control
    if collect_data:
        command_random[k, 0] = np.random.uniform(-10, 20, 1)
        command_random[k, 1] = np.random.uniform(-90, 90, 1)
    else:
        time_start = time.time()
        command_pid[k, :] = pid.control(pose[k, :], reference[k, :])
        time_pid += (time.time() - time_start)

        time_start = time.time()
        command_dnn[k, :] = dnn.control(pose[k, :], reference[k + 1, :])
        time_dnn0 += (time.time() - time_start)
        if controller == 4 and k > 0:
            time_start = time.time()
            dnn.learn(pose[k, :], pose[k - 1, :], command[k - 1, :])
            time_dnn += (time.time() - time_start)

        time_start = time.time()
        command_inverse[k, :] = inverse.control(pose[k, :], reference[k + 1, :])
        time_inverse += (time.time() - time_start)

    if controller == 0:
        command[k, :] = command_random[k, :]
    else:
        if controller == 1:
            command[k, :] = command_pid[k, :]
        else:
            if controller == 2:
                command[k, :] = command_inverse[k, :]
            else:
                if controller == 3 or controller == 4:
                    command[k, :] = command_dnn[k, :]

    # Simulate unicycle
    if k < k_end * 1 / 7:
        pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], 0.0, 0.0, 0.0)
    if k_end * 1 / 7 <= k < k_end * 2 / 7:
        pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], 0.0, 0.0, noise)
    if k_end * 2 / 7 <= k < k_end * 3 / 7:
        pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], uncertainty, 0.0, noise)
    if k_end * 3 / 7 <= k < k_end * 4 / 7:
        pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], uncertainty, disturbance, noise)
    if k_end * 4 / 7 <= k < k_end * 5 / 7:
        pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], uncertainty, 0.0, noise)
    if k_end * 5 / 7 <= k < k_end * 6 / 7:
        pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], 0.0, 0.0, noise)
    if k_end * 6 / 7 <= k:
        pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], 0.0, 0.0, 0.0)

    # Animation
    if animation:
        show_animation(pose[:k + 1], reference[:k + 1])

# Save online DNN

if controller == 4:
    dnn.save('new_' + dnn_name)

# Computational time

print('PID time: %.3f ms' % (time_pid / k_end * 1000))
print('Inverse time: %.3f ms' % (time_inverse / k_end * 1000))
print('DNN0 time: %.3f ms' % (time_dnn0 / k_end*1000))
print('DNN time: %.3f ms' % ((time_dnn + time_dnn0) / k_end*1000))

# Save results

if collect_data:
    save_data_kinematics(t, reference, pose, command, 'unicycle_kinematics_random_bound20_100')

# Plot results

if not collect_data:
    show_plots(t, pose_real, reference, command, command_dnn, command_inverse)

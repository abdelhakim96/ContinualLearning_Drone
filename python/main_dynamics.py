# MAIN

import numpy as np
import time

from dynamics.pid_dynamics import PID
from dynamics.dnn_dynamics import DNN
from dynamics.inverse_dynamics import Inverse
from dynamics.unicycle_dynamics import Unicycle
from utils.save_data import save_data_dynamics
from utils.show import show_plots_dynamics
from utils.animate import AnimationUnicycle

# Parameters

collect_data = False  # True - collect data, False - test performance
controller = 2  # 0 - random, 1 - PID, 2 - inverse, 3 - DNN0, 4 - online DNN
trajectory = 2  # 0 - random points, 1 - circular, 2 - 8-shaped, 3 - set-point, 4 - square-wave
uncertainty = 0  # internal uncertainty: 0 - no uncertainty, 1 - all parameters double, -1 - all parameters half;
# default = -1
disturbance = 20  # external disturbance: 0 - no disturbance, >0 - positive disturbance, <0 - negative disturbance
# default = 20
noise = 0.0  # measurement noise standard deviation: 0 - no noise, >0 - white noise
# default = 0.0001

show_animation = True  # True - enable online animation, False - disable online animation

dnn_name = 'dnn_dynamics_100x10'

k_end = 6000
dt = 0.001

# Initial pose

x_init = 1
y_init = 0
yaw_init = 3

if collect_data:
    controller = 1
    trajectory = 0
    uncertainty = 0
    disturbance = 0
    noise = 0
    k_end = 10000

    x_init = 0
    y_init = 0
    yaw_init = 0

# Initialise arrays

pose = np.zeros((k_end + 1, 3))
pose[0, :] = [x_init, y_init, yaw_init]
pose[1, :] = [x_init, y_init, yaw_init]
pose_real = np.zeros((k_end + 1, 3))
pose_real[0, :] = [x_init, y_init, yaw_init]
pose_real[1, :] = [x_init, y_init, yaw_init]
reference = np.zeros((k_end + 1, 3))
command = np.zeros((k_end + 1, 2))
command_random = np.zeros((k_end + 1, 2))
command_pid = np.zeros((k_end + 1, 2))
command_dnn = np.zeros((k_end + 1, 2))
command_inverse = np.zeros((k_end + 1, 2))
t = np.linspace(0, dt * k_end, num=(k_end + 1))

# Generate trajectory

if trajectory == 1:  # circular
    reference = np.column_stack([2 * np.cos(2 * t), -2 * np.sin(2 * t), np.zeros((k_end + 1, 1))])
if trajectory == 2:  # 8-shaped
    # t -= np.pi/2
    reference = np.column_stack([
        4 / (3 - np.cos(2 * t)) * np.cos(t),
        4 / (3 - np.cos(2 * t)) * np.sin(-2 * t) / np.sqrt(2),
        np.zeros((k_end + 1, 1))])
    # t += np.pi / 2
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
if not collect_data:
    dnn = DNN(dt, dnn_name)
    inverse = Inverse(unicycle)

animation = AnimationUnicycle(reference)

# Main loop

time_pid = 0
time_inverse = 0
time_dnn0 = 0
time_dnn = 0
time_animation = dt
for k in range(1, k_end - 1):

    # Unicycle control
    if collect_data:
        reference[k, 0:2] = reference[k - 1, 0:2] + 100 * np.random.randn(1, 2) * dt  # random displacement
        # reference[k, 2] = reference[k - 1, 2] + 90 * np.random.randn(1, 1) * dt  # random orientation
        command_pid[k, :] = pid.control(pose[k, :], reference[k, :])
        command_pid[k, 0] /= 100
        command_pid[k, 1] /= 100

        command_pid[k, :] = np.clip(command_pid[k, :], -100, 100)
    else:
        time_start = time.time()
        command_pid[k, :] = pid.control(pose[k, :], reference[k, :])
        time_pid += (time.time() - time_start)

        time_start = time.time()
        command_dnn[k, :] = dnn.control(pose[k, :], reference[k + 2, :])
        time_dnn0 += (time.time() - time_start)
        if controller == 4 and k > 0:
            time_start = time.time()
            dnn.learn(pose[k, :], pose[k - 1, :], command[k - 1, :])
            time_dnn += (time.time() - time_start)

        time_start = time.time()
        command_inverse[k, :] = inverse.control(pose[k, :], reference[k + 2, :])
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
    # else:
    #     pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], uncertainty, disturbance, noise)

    # Animation
    if show_animation:
        if True:#k % round(time_animation / dt) == 0:
            time_start = time.time()
            animation.update(pose[:k + 1, :], reference[k, :], command[k, :])
            time_animation = time.time() - time_start
            #print('Animation time: %.3f ms' % (time_animation * 1000))
        #time.sleep(0.1)

    print('Progress: %.1f%%' % (k/(k_end - 1)*100))

# Save online DNN

if controller == 4:
    dnn.save('new_' + dnn_name)

# Computational time

print('PID time: %.3f ms' % (time_pid / k_end * 1000))
print('Inverse time: %.3f ms' % (time_inverse / k_end * 1000))
print('DNN0 time: %.3f ms' % (time_dnn0 / k_end * 1000))
print('DNN time: %.3f ms' % ((time_dnn + time_dnn0) / k_end * 1000))

# Save results

if collect_data:
    save_data_dynamics(t, reference, pose, command, 'unicycle_dynamics_random')

# Plot results

if not collect_data:
    show_plots_dynamics(t, pose_real, reference, command, command_dnn, command_inverse)
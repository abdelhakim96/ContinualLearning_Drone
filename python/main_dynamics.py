# MAIN

import numpy as np

from dynamics.pid_dynamics import PID
from dynamics.dnn_dynamics import DNN
from dynamics.inverse_dynamics import Inverse
from dynamics.unicycle_dynamics import Unicycle
from utils.save_data import save_data_dynamics
from utils.show import show_plots, show_animation

# Parameters

collect_data = False  # True - collect data, False - test performance
controller = 2  # 0 - random, 1 - PID, 2 - DNN, 3 - inverse
online_learning = 1  # For DNN (controller = 2): True (1) - enable online learning, False (0) - disable online learning
trajectory = 1  # 0 - random points, 1 - circular, 2 - 8-shaped, 3 - set-point, 4 - square-wave
uncertainty = 0  # internal uncertainty: 0 - no uncertainty, 1 - all parameters double, -1 - all parameters half;
# default = 1
disturbance = 0  # external disturbance: 0 - no disturbance, >0 - positive disturbance, <0 - negative disturbance
# default = -100
noise = 0  # measurement noise standard deviation: 0 - no noise, >0 - white noise
# default = 0.01

animation = False  # True - enable online animation, False - disable online animation

k_end = 10000
dt = 0.001

if collect_data:
    controller = 1
    online_learning = False
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
reference = np.zeros((k_end + 1, 3))
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
    reference = np.column_stack([-2 * np.cos(2 * t), 2 * np.sin(2 * t), np.zeros((k_end + 1, 1))])
if trajectory == 2:  # 8-shaped
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
dnn = DNN(dt, 'dnn_dynamics_512x512x512')
inverse = Inverse(unicycle)

# Main loop

for k in np.arange(1, k_end - 1):

    # Unicycle control
    command_pid[k, :] = pid.control(pose[k, :], reference[k, :])
    if not collect_data:
        command_dnn[k, :] = dnn.control(pose[k, :], reference[k + 2, :])
        if online_learning and k > 0:
            dnn.learn(pose[k, :], pose[k - 1, :], command[k - 1, :])
        command_inverse[k, :] = inverse.control(pose[k, :], reference[k + 2, :])

    if controller == 0:
        command[k, :] = command_random[k, :]
    else:
        if controller == 1:
            command[k, :] = command_pid[k, :]
        else:
            if controller == 2:
                command[k, :] = command_dnn[k, :]
            else:
                if controller == 3:
                    command[k, :] = command_inverse[k, :]

    # Simulate unicycle
    if k < k_end / 2:
        pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], 0, 0, noise)
    else:
        pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :], uncertainty, disturbance, noise)

    # Animation
    if animation:
        show_animation(pose[:k + 1], reference[:k + 1])

# Save results

if collect_data:
    save_data_dynamics(t, reference, pose, command, 'unicycle_dynamics_random')

# Plot results

if not collect_data:
    show_plots(t, pose_real, reference, command, command_dnn, command_inverse)


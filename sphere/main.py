# MAIN

import numpy as np

from pid import PID
from dnn import DNN
from sphere import Sphere
from save_data import save_data
from show import show_plots, show_animation

# Parameters

controller = 2  # 1 - PID controller, 2 - DNN controller
trajectory = 2  # 1 - circular, 2 - set-point, 3 - random points, 4 - square-wave
uncertainty = 0  # internal uncertainty: 0 - no uncertainty, 1 - all parameters double, -1 - all parameters half;
# default = 1
disturbance = 0  # external disturbance: 0 - no disturbance, >0 - positive disturbance, <0 - negative disturbance
# default = -100
noise = 0  # measurement noise standard deviation: 0 - no noise, >0 - white noise
# default = 0.01

animation = False  # True - enable online animation, False - disable online animation

k_end = 1000
dt = 0.01

# Initial pose

x_init = 0
y_init = 0
yaw_init = 0

# Initialise arrays

pose = np.zeros((k_end + 1, 3))
pose[0, :] = [x_init, y_init, yaw_init]
command = np.zeros((k_end + 1, 3))
t = np.arange(0, dt * (k_end + 1), dt)

# Generate trajectory

if 1 == trajectory:  # circular
    trajectory = np.column_stack([2 * np.cos(t), -2 * np.sin(t), np.zeros((k_end + 1, 1))])
else:
    if 2 == trajectory:  # set-point
        trajectory = np.column_stack(
            [2 * np.ones((k_end + 1, 1)), -2 * np.ones((k_end + 1, 1)), np.zeros((k_end + 1, 1))])
    else:
        if 3 == trajectory:  # random points (for collecting training data)
            trajectory = np.random.randn(k_end + 1, 3)
        else:
            if 4 == trajectory:  # square-wave
                d = np.fix(t)
                d = d % 4
                b1 = np.fix(d / 2)
                b0 = d - 2 * b1
                trajectory = np.column_stack([b1, b0, np.zeros((k_end + 1, 1))])

pid = PID(dt)
dnn = DNN(dt, 'dnn_64x64')
sphere = Sphere(dt, [x_init, y_init, yaw_init])

# Main loop

for k in np.arange(1, k_end - 2):

    # Unicycle control
    if 1 == controller:
        command[k, :] = pid.control(pose[k, :], trajectory[k, :])
    else:
        if 2 == controller:
            command[k, :] = dnn.control(pose[k, :], trajectory[k + 1:k + 4, :])
    #command[k, 2] = 0  # no heading control (spinning)
    #command[k, :] = [0, 1, 0]

    # Simulate unicycle
    if k < k_end / 2:
        pose[k + 1, :] = sphere.simulate(command[k, :], 0, 0, noise)
    else:
        pose[k + 1, :] = sphere.simulate(command[k, :], uncertainty, disturbance, noise)  # default: [1, -100, 0.01]

    # Animation
    if animation:
        show_animation(pose[:k + 1], trajectory[:k + 1])

# Save results

#save_data(t, trajectory, pose, command, 'sphere_random')

# Plot results

show_plots(t, pose, trajectory, command)

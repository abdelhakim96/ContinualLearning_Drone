# MAIN

import numpy as np

from pid import PID
# from dnn import DNN
from unicycle import Unicycle
from save_data import save_data
from show import show


# Parameters

controller = 1  # 1 - PID controller, 2 - DNN controller
trajectory = 1  # 1 - circular, 2 - set-point, 3 - random points, 4 - square-wave
uncertainty = 0  # internal uncertainty: 0 - no uncertainty, 1 - all parameters double, -1 - all parameters half
disturbance = 0  # external disturbance: 0 - no disturbance, >0 - positive disturbance, <0 - negative disturbance
noise = 0  # measurement noise standard deviation: 0 - no noise, >0 - white noise

k_end = 100000
dt = 0.01

# Initial pose

x_init = 0
y_init = 0
yaw_init = 0

# Initialise arrays

pose = np.zeros((k_end, 3))
pose[0, :] = [x_init, y_init, yaw_init]
command = np.zeros((k_end, 2))
t = np.arange(0, dt * k_end, dt)

# Generate trajectory

if 1 == trajectory:  # circular
    trajectory = np.column_stack([-2 * np.cos(t), 2 * np.sin(t), np.zeros((k_end, 1))])
else:
    if 2 == trajectory:  # set-point
        trajectory = np.column_stack([2 * np.ones((k_end, 1)), -2 * np.ones((k_end, 1)), np.zeros((k_end, 1))])
    else:
        if 3 == trajectory:  # random points (for collecting training data)
            trajectory = np.random.randn(k_end, 3)
        else:
            if 4 == trajectory:  # square-wave
                d = np.rint(t)
                d = np.rem(d, 4)
                b1 = np.rint(d / 2)
                b0 = d - 2 * b1
                trajectory = np.column_stack([b1, b0, np.zeros((k_end, 1))])

pid = PID(dt)
# dnn = DNN(dt)
unicycle = Unicycle(dt, [x_init, y_init, yaw_init])

# Main loop

for k in np.arange(k_end - 1):

    # Unicycle control

    if 1 == controller:
        command[k, :] = pid.control(pose[k, :], trajectory[k, :])
    else:
        if 2 == controller:
            pass  # command[k,:] = dnn.control(pose[k,:], trajectory[np.arange(k + 1,k + 3),:])
    #command[k, 1] = -1000 # no heading control (spinning)

    if k < k_end/2:
        pose[k + 1, :] = unicycle.simulate(command[k, :], 0, 0, noise)
    else:
        pose[k + 1, :] = unicycle.simulate(command[k, :], uncertainty, disturbance, noise)  # default: [1, -100, 0.01]

# Save results

save_data(t, trajectory, pose, command, 'random')

# Plot results

show(t, pose, trajectory, command)

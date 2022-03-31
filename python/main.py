# MAIN

import numpy as np
import csv

from show import show
from pid import PID
# from dnn import DNN
from unicycle import Unicycle

# Parameters

controller = 1  # 1 - PID controller, 2 - DNN controller
trajectory = 1  # 1 - circular, 2 - set-point, 3 - random points, 4 - square-wave

k_end = 1000
dt = 0.01

# Initial pose

x_init = 0
y_init = 0
yaw_init = 0

# Initialise arrays

pose = np.zeros((k_end, 3))
command = np.zeros((k_end, 2))
t = np.arange(0, dt * k_end, dt)

# Generate trajectory

if 1 == trajectory:  # circular
    trajectory = np.column_stack([-2 * np.cos(t), 2 * np.sin(t), np.zeros((k_end, 1))])
else:
    if 2 == trajectory:  # set-point
        trajectory = np.array([2 * np.ones((k_end, 1)), -2 * np.ones((k_end, 1)), np.zeros((k_end, 1))])
    else:
        if 3 == trajectory:  # random points (for collecting training data)
            trajectory = np.array([np.random.randn(k_end, 1), np.random.randn(k_end, 1), np.random.randn(k_end, 1)])
        else:
            if 4 == trajectory:  # square-wave
                d = np.rint(t)
                d = np.rem(d, 4)
                b1 = np.rint(d / 2)
                b0 = d - 2 * b1
                trajectory = np.array([b1, b0, np.zeros((k_end, 1))])

pid = PID(dt)
# dnn = DNN(dt)
unicycle = Unicycle(dt, x_init, y_init, yaw_init)

# Main loop

for k in np.arange(k_end - 1):

    # Unicycle control

    if 1 == controller:
        command[k, :] = pid.control(pose[k, :], trajectory[k, :])
    else:
        if 2 == controller:
            pass  # command[k,:] = dnn.control(pose[k,:], trajectory[np.arange(k + 1,k + 3),:])
    # command(k,2) = 1; # no heading control (spinning)

    # Unicycle model

    pose[k + 1, :] = unicycle.simulate(command[k, :])

# Plot results

# show(t, pose, trajectory, command)

# Save results

file = open("data\samples_step.txt", "w")
file.write("%s = %s\n" % ("t", t))
file.write("%s = %s\n" % ("pose", pose))
file.write("%s = %s\n" % ("command", command))
file.close()
# save('data/samples_step', 't', 'pose', 'command')

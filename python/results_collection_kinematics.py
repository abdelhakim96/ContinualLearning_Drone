# Collects results

import numpy as np
import time
import matplotlib.pyplot as plt

from kinematics.pid_kinematics import PID
from kinematics.inverse_kinematics import Inverse
from kinematics.dfnn_kinematics import DFNN
from kinematics.dnn_er_kinematics import DNN as DNN_ER
from kinematics.dnn_agem_kinematics import DNN as DNN_AGEM
from kinematics.dnn_ewc_kinematics import DNN as DNN_EWC
from kinematics.dnn_lwf_kinematics import DNN as DNN_LWF
from kinematics.unicycle_kinematics import Unicycle
from utils.save_data import save_experiment_kinematics, save_experiment_kinematics1
from utils.show import show_trajectory


# Functions

def get_conditions(experiment, k):
    if experiment == 0:
        return [0, 0, 0]
    if experiment == 1:
        if k < k_end * 1 / 3:
            return [0, 0, 0]
        if k_end * 1 / 3 <= k < k_end * 2 / 3:
            return [uncertainty, 0, 0]
        if k_end * 2 / 3 <= k:
            return [0, 0, 0]
    if experiment == 2:
        if k < k_end * 1 / 3:
            return [0, 0, 0]
        if k_end * 1 / 3 <= k < k_end * 2 / 3:
            return [0, disturbance, 0]
        if k_end * 2 / 3 <= k:
            return [0, 0, 0]
    if experiment == 3:
        if k < k_end * 1 / 3:
            return [0, 0, 0]
        if k_end * 1 / 3 <= k < k_end * 2 / 3:
            return [0, 0, noise]
        if k_end * 2 / 3 <= k:
            return [0, 0, 0]
    if experiment == 4:
        if k < k_end * 1 / 7:
            return [0.0, 0.0, 0.0]
        if k_end * 1 / 7 <= k < k_end * 2 / 7:
            return [0.0, 0.0, noise_combined]
        if k_end * 2 / 7 <= k < k_end * 3 / 7:
            return [uncertainty, 0.0, noise_combined]
        if k_end * 3 / 7 <= k < k_end * 4 / 7:
            return [uncertainty_combined, disturbance_combined, noise_combined]
        if k_end * 4 / 7 <= k < k_end * 5 / 7:
            return [uncertainty_combined, 0.0, noise_combined]
        if k_end * 5 / 7 <= k < k_end * 6 / 7:
            return [0.0, 0.0, noise_combined]
        if k_end * 6 / 7 <= k:
            return [0.0, 0.0, 0.0]


def get_changes(experiment):
    if experiment == 0:
        return [[], []]
    if experiment == 1 or experiment == 2 or experiment == 3:
        return [[int(k_end * 1 / 3)], [int(k_end * 2 / 3)]]
    if experiment == 4:
        return [[int(k_end * 1 / 7), int(k_end * 2 / 7), int(k_end * 3 / 7)],
                [int(k_end * 4 / 7), int(k_end * 5 / 7), int(k_end * 6 / 7)]]


# Parameters

dt = 0.001

uncertainty = -1  # default = -2
disturbance = 40  # default = 20
noise = 0.1  # default = 0.0002
uncertainty_combined = -1  # default = -2
disturbance_combined = 10  # default = 20
noise_combined = 0.01  # default = 0.0002

dnnName = 'dnn_kinematics_32'
dfnnName = 'dfnn_kinematics_32'

# Initialise metric

experiments = [4]  # [0, 1, 2, 3, 4]
controllers = [1, 2, 3, 4, 5, 6, 7]  # [0, 1, 2, 3, 4, 5, 6, 7]

mae = np.zeros((5, 8))
times = np.zeros(8)

for experiment in experiments:

    # Generate trajectory
    if experiment == 4:
        scale = 10
        speed = 5
    else:
        scale = 1
        speed = 2

    k_end = int(2 * np.pi / dt * scale / speed)
    t = np.linspace(0, dt * k_end, num=(k_end + 1))

    reference = np.column_stack([
        4 * scale * np.cos(speed * t / scale) / (3 - np.cos(2 * speed * t / scale)),
        -2 * np.sqrt(2) * scale * np.sin(2 * speed * t / scale) / (3 - np.cos(2 * speed * t / scale)),
        np.zeros((k_end + 1, 1))])

    # Initial pose

    x_init = 1 * scale
    y_init = 0
    yaw_init = 0

    for controller in controllers:

        # Initialise arrays

        pose = np.zeros((k_end + 1, 3))
        pose[0, :] = [x_init, y_init, yaw_init]
        pose[1, :] = [x_init, y_init, yaw_init]
        pose_real = np.zeros((k_end + 1, 3))
        pose_real[0, :] = [x_init, y_init, yaw_init]
        pose_real[1, :] = [x_init, y_init, yaw_init]
        command = np.zeros((k_end + 1, 2))
        command_inverse = np.zeros((k_end + 1, 2))

        unicycle = Unicycle(dt, [x_init, y_init, yaw_init])
        # unicycle.max_w_y = 60
        # if controller != 1:
        #     unicycle.max_w_y = 200

        inverse = Inverse(unicycle)
        if controller == 1:
            pid = PID(dt)
        if controller == 2 or controller == 4:
            dnn = DNN_LWF(dt, dnnName)
        if controller == 3:
            dfnn = DFNN(unicycle, dfnnName)
        if controller == 5:
            dnn = DNN_ER(dt, dnnName)
        if controller == 6:
            dnn = DNN_AGEM(dt, dnnName)
        if controller == 7:
            dnn = DNN_EWC(dt, dnnName)

        # Main loop

        computational_time = 0
        for k in range(1, k_end):

            # Unicycle control
            time_start = time.time()
            # 0 - inverse, 1 - PID, 2 - DNN0, 3 - DFNN, 4 - DNN+LwF, 5 - DNN+ER, 6 - DNN+AGEM, 7 - DNN+EWC
            if controller == 0:
                command[k, :] = inverse.control(pose[k, :], reference[k + 1, :])
                command_inverse[k, :] = command[k, :]
            if controller == 1:
                command[k, :] = pid.control(pose[k, :], reference[k, :])
            if controller == 2 or controller == 4 or controller == 5 or controller == 6 or controller == 7:
                command[k, :] = dnn.control(pose[k, :], reference[k + 1, :])
            if controller == 3:
                command[k, :] = dfnn.control(pose[k, :], reference[k + 1, :])

            # if controller == 2:
            #     command[k, 0] = 2 * command[k, 0]
            # if controller == 3:
            #     command[k, 0] = 2 * command[k, 0]
            # if controller == 4 or controller == 5 or controller == 6 or controller == 7:
            #     command[k, 0] = 3 * command[k, 0]

            if controller == 4 or controller == 5 or controller == 6 or controller == 7:
                dnn.learn(pose[k, :], pose[k - 1, :], command[k - 1, :])
            computational_time += (time.time() - time_start)

            # Simulate unicycle
            conditions = get_conditions(experiment, k)
            pose[k + 1, :], pose_real[k + 1, :] = unicycle.simulate(command[k, :],
                                                                    conditions[0], conditions[1], conditions[2])

        # Save online DNN

        if controller == 4 or controller == 5 or controller == 6 or controller == 7:
            dnn.save('new_' + dnnName)

        # Comparison metrics

        e_x = reference[:, 0] - pose_real[:, 0]
        e_y = reference[:, 1] - pose_real[:, 1]
        mae[experiment, controller] = np.mean(np.sqrt(e_x ** 2 + e_y ** 2))

        times[controller] = computational_time / k_end * 1000

        # Save results

        save_experiment_kinematics(t, reference, pose_real, command, command_inverse,
                                   'unicycle_kinematics_' + str(experiment) + '_' + str(controller))

        # Plot results

        show_trajectory(experiment, controller, pose_real, reference, get_changes(experiment))

print(np.transpose(mae))
print(times)

plt.show()

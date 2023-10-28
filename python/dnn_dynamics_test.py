import math

import numpy as np
import matplotlib.pyplot as plt

from dynamics.dnn_dynamics import DNN
from dynamics.inverse_dynamics import Inverse
from dynamics.unicycle_dynamics import Unicycle

dt = 0.001

dnn = DNN(dt, 'dnn_dynamics_8x4')
inverse = Inverse(Unicycle(dt, [0, 0, 0]))

data_min = dnn.get_data_min()
data_max = dnn.get_data_max()

command_x_dnn = np.zeros((5, 99, 2))
command_x_inverse = np.zeros((5, 99, 2))
command_y_dnn = np.zeros((5, 99, 2))
command_y_inverse = np.zeros((5, 99, 2))
command_yaw_dnn = np.zeros((4, 99, 2))
command_yaw_inverse = np.zeros((4, 99, 2))

##########

print(dnn.control([-10, 0, 1], [0, 0]))
exit()

x_d = np.linspace(2 * data_min[3], 2 * data_max[3], 99)
y_d = np.linspace(2 * data_min[4], 2 * data_max[4], 99)
yaw = np.linspace(-np.pi, np.pi, 5)

for i in np.arange(0, 5):
    for k in np.arange(0, 99):
        command_x_dnn[i, k, :] = np.clip(dnn.control([0, 0, yaw[i]], [x_d[k], 0]), -100, 100)
        command_x_inverse[i, k, :] = np.clip(inverse.control(np.array([0, 0, yaw[i]]), np.array([x_d[k], 0, 0])), -100, 100)
        command_y_dnn[i, k, :] = np.clip(dnn.control([0, 0, yaw[i]], [0, y_d[k]]), -100, 100)
        command_y_inverse[i, k, :] = np.clip(inverse.control(np.array([0, 0, yaw[i]]), np.array([0, y_d[k], 0])), -100, 100)

plt.figure(1)
plt.title('x control')
plt.plot(x_d, command_x_dnn[0, :, 0], 'b-', label="-pi")
plt.plot(x_d, command_x_inverse[0, :, 0], 'b:')
plt.plot(x_d, command_x_dnn[1, :, 0], 'g-', label="-pi/2")
plt.plot(x_d, command_x_inverse[1, :, 0], 'g:')
plt.plot(x_d, command_x_dnn[2, :, 0], 'r-', label="0")
plt.plot(x_d, command_x_inverse[2, :, 0], 'r:')
plt.plot(x_d, command_x_dnn[3, :, 0], 'c-', label="pi/2")
plt.plot(x_d, command_x_inverse[3, :, 0], 'c:')
plt.plot(x_d, command_x_dnn[4, :, 0], 'm-', label="pi")
plt.plot(x_d, command_x_inverse[4, :, 0], 'm:')
plt.xlabel('diff_x [m]')
plt.ylabel('tau_y [rad/s]')
plt.ylim([-100, 100])
plt.legend()
plt.grid()

plt.figure(2)
plt.title('y control')
plt.plot(y_d, command_y_dnn[0, :, 0], 'b-', label="-pi")
plt.plot(y_d, command_y_inverse[0, :, 0], 'b:')
plt.plot(y_d, command_y_dnn[1, :, 0], 'g-', label="-pi/2")
plt.plot(y_d, command_y_inverse[1, :, 0], 'g:')
plt.plot(y_d, command_y_dnn[2, :, 0], 'r-', label="0")
plt.plot(y_d, command_y_inverse[2, :, 0], 'r:')
plt.plot(y_d, command_y_dnn[3, :, 0], 'c-', label="pi/2")
plt.plot(y_d, command_y_inverse[3, :, 0], 'c:')
plt.plot(y_d, command_y_dnn[4, :, 0], 'm-', label="pi")
plt.plot(y_d, command_y_inverse[4, :, 0], 'm:')
plt.xlabel('diff_y [m]')
plt.ylabel('tau_y [rad/s]')
plt.ylim([-100, 100])
plt.legend()
plt.grid()

#########################

# yaw = np.linspace(-2 * np.pi, 2 * np.pi, 99)
#
# for k in np.arange(0, 99):
#     dnn.old_pose = [0, 0, yaw[k]]
#     inverse.old_pose = [0, 0, yaw[k]]
#     command_yaw_dnn[0, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [-1, 0]), -100, 100)
#     command_yaw_inverse[0, k, :] = np.clip(inverse.control(np.array([0, 0, yaw[k]]), np.array([-1, 0, 0])), -100, 100)
#     command_yaw_dnn[1, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [1, 0]), -100, 100)
#     command_yaw_inverse[1, k, :] = np.clip(inverse.control(np.array([0, 0, yaw[k]]), np.array([1, 0, 0])), -100, 100)
#     command_yaw_dnn[2, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [0, -1]), -100, 100)
#     command_yaw_inverse[2, k, :] = np.clip(inverse.control(np.array([0, 0, yaw[k]]), np.array([0, -1, 0])), -100, 100)
#     command_yaw_dnn[3, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [0, 1]), -100, 100)
#     command_yaw_inverse[3, k, :] = np.clip(inverse.control(np.array([0, 0, yaw[k]]), np.array([0, 1, 0])), -100, 100)
#
# plt.figure(5)
# plt.title('yaw control')
# plt.plot(yaw/np.pi*180, command_yaw_dnn[0, :, 1], 'b-', label="yaw_d = 180")
# plt.plot(yaw/np.pi*180, command_yaw_inverse[0, :, 1], 'b:')
# plt.plot(yaw/np.pi*180, command_yaw_dnn[1, :, 1], 'g-', label="yaw_d = 0")
# plt.plot(yaw/np.pi*180, command_yaw_inverse[1, :, 1], 'g:')
# # plt.plot(yaw/np.pi*180, command_yaw_dnn[2, :, 1], 'r-', label="yaw_d = -90")
# # plt.plot(yaw/np.pi*180, command_yaw_inverse[2, :, 1], 'r:')
# # plt.plot(yaw/np.pi*180, command_yaw_dnn[3, :, 1], 'c-', label="yaw_d = 90")
# # plt.plot(yaw/np.pi*180, command_yaw_inverse[3, :, 1], 'c:')
# plt.xlabel('yaw [deg]')
# plt.ylabel('tau_z [rad/s]')
# # plt.ylim([-100, 100])
# plt.legend()
# plt.grid()
#
# yaw = np.linspace(-2 * np.pi, 2 * np.pi, 99)
#
# for k in np.arange(0, 99):
#     dnn.old_pose = [0, 0, yaw[k] + 90 * 0.001]
#     inverse.old_pose = [0, 0, yaw[k] + 90 * 0.001]
#     command_yaw_dnn[0, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [1, 0]), -100, 100)
#     command_yaw_inverse[0, k, :] = np.clip(inverse.control(np.array([0, 0, yaw[k]]), np.array([1, 0, 0])), -100, 100)
#     dnn.old_pose = [0, 0, yaw[k]]
#     inverse.old_pose = [0, 0, yaw[k]]
#     command_yaw_dnn[1, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [1, 0]), -100, 100)
#     command_yaw_inverse[1, k, :] = np.clip(inverse.control(np.array([0, 0, yaw[k]]), np.array([1, 0, 0])), -100, 100)
#     dnn.old_pose = [0, 0, yaw[k] - 90 * 0.001]
#     inverse.old_pose = [0, 0, yaw[k] - 90 * 0.001]
#     command_yaw_dnn[2, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [1, 0]), -100, 100)
#     command_yaw_inverse[2, k, :] = np.clip(inverse.control(np.array([0, 0, yaw[k]]), np.array([1, 0, 0])), -100, 100)
#
# plt.figure(6)
# plt.title('yaw control')
# plt.plot(yaw/np.pi*180, command_yaw_dnn[0, :, 1], 'b-', label="w = -90")
# plt.plot(yaw/np.pi*180, command_yaw_inverse[0, :, 1], 'b:')
# plt.plot(yaw/np.pi*180, command_yaw_dnn[1, :, 1], 'g-', label="w = 0")
# plt.plot(yaw/np.pi*180, command_yaw_inverse[1, :, 1], 'g:')
# plt.plot(yaw/np.pi*180, command_yaw_dnn[2, :, 1], 'r-', label="w = 90")
# plt.plot(yaw/np.pi*180, command_yaw_inverse[2, :, 1], 'r:')
# plt.xlabel('yaw [deg]')
# plt.ylabel('tau_z [rad/s]')
# # plt.ylim([-100, 100])
# plt.legend()
# plt.grid()

plt.show()



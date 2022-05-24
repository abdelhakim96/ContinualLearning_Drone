# TEST

import numpy as np
import matplotlib.pyplot as plt

from kinematics.dnn_kinematics import DNN
from kinematics.inverse_kinematics import Inverse
from kinematics.unicycle_kinematics import Unicycle

dt = 0.001

dnn = DNN(dt, 'dnn_kinematics_32x32')
inverse = Inverse(Unicycle(dt, [0, 0, 0]))

##########

print(np.clip(dnn.control([0, 0, -np.pi], [-1, 0]), -10, 10), np.clip(dnn.control([0, 0, np.pi], [-1, 0]), -10, 10))
print(np.clip(dnn.control([0, 0, -np.pi + 0.1], [1, 0]), -10, 10), np.clip(dnn.control([0, 0, np.pi + 0.1], [1, 0]), -10, 10))
print(np.clip(dnn.control([0, 0, -np.pi], [0, -1]), -10, 10), np.clip(dnn.control([0, 0, np.pi], [0, -1]), -10, 10))
print(np.clip(dnn.control([0, 0, -np.pi], [0, 1]), -10, 10), np.clip(dnn.control([0, 0, np.pi], [0, 1]), -10, 10))

##########

data_min = dnn.get_data_min()
data_max = dnn.get_data_max()

x = np.linspace(2 * data_min[2], 2 * data_max[2], 10)
y = np.linspace(2 * data_min[3], 2 * data_max[3], 10)
yaw = np.linspace(-2 * np.pi, 2 * np.pi, 10)

mae = 0
for i in np.arange(10):
    for j in np.arange(10):
        for k in np.arange(10):
            command_dnn = np.clip(dnn.control([x[i], y[j], yaw[k]], [0, 0]), -10, 10)
            command_inverse = np.clip(inverse.control([x[i], y[j], yaw[k]], [0, 0]), -10, 10)
            mae += np.abs(command_dnn - command_inverse)
mae /= 1000
print(mae)

##########

x_d = np.linspace(2 * data_min[2], 2 * data_max[2], 100)
y_d = np.linspace(2 * data_min[3], 2 * data_max[3], 100)
yaw = np.linspace(-np.pi, np.pi, 5)

command_x_dnn = np.zeros((5, 100, 2))
command_x_inverse = np.zeros((5, 100, 2))
command_y_dnn = np.zeros((5, 100, 2))
command_y_inverse = np.zeros((5, 100, 2))
command_yaw_dnn = np.zeros((4, 100, 2))
command_yaw_inverse = np.zeros((4, 100, 2))

for i in np.arange(0, 5):
    for k in np.arange(0, 100):
        command_x_dnn[i, k, :] = np.clip(dnn.control([0, 0, yaw[i]], [x_d[k], 0]), -10, 10)
        command_x_inverse[i, k, :] = np.clip(inverse.control([0, 0, yaw[i]], [x_d[k], 0]), -10, 10)
        command_y_dnn[i, k, :] = np.clip(dnn.control([0, 0, yaw[i]], [0, y_d[k]]), -10, 10)
        command_y_inverse[i, k, :] = np.clip(inverse.control([0, 0, yaw[i]], [0, y_d[k]]), -10, 10)

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
plt.xlabel('x_d [m]')
plt.ylabel('w_y [rad/s]')
plt.ylim([-11, 11])
plt.legend()
plt.grid()

plt.figure(2)
plt.title('x control')
plt.plot(x_d, command_x_dnn[0, :, 1], 'b-', label="-pi")
plt.plot(x_d, command_x_inverse[0, :, 1], 'b:')
plt.plot(x_d, command_x_dnn[1, :, 1], 'g-', label="-pi/2")
plt.plot(x_d, command_x_inverse[1, :, 1], 'g:')
plt.plot(x_d, command_x_dnn[2, :, 1], 'r-', label="0")
plt.plot(x_d, command_x_inverse[2, :, 1], 'r:')
plt.plot(x_d, command_x_dnn[3, :, 1], 'c-', label="pi/2")
plt.plot(x_d, command_x_inverse[3, :, 1], 'c:')
plt.plot(x_d, command_x_dnn[4, :, 1], 'm-', label="pi")
plt.plot(x_d, command_x_inverse[4, :, 1], 'm:')
plt.xlabel('x_d [m]')
plt.ylabel('w_z [rad/s]')
plt.ylim([-11, 11])
plt.legend()
plt.grid()

plt.figure(3)
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
plt.xlabel('y_d [m]')
plt.ylabel('w_y [rad/s]')
plt.ylim([-11, 11])
plt.legend()
plt.grid()

plt.figure(4)
plt.title('y control')
plt.plot(y_d, command_y_dnn[0, :, 1], 'b-', label="-pi")
plt.plot(y_d, command_y_inverse[0, :, 1], 'b:')
plt.plot(y_d, command_y_dnn[1, :, 1], 'g-', label="-pi/2")
plt.plot(y_d, command_y_inverse[1, :, 1], 'g:')
plt.plot(y_d, command_y_dnn[2, :, 1], 'r-', label="0")
plt.plot(y_d, command_y_inverse[2, :, 1], 'r:')
plt.plot(y_d, command_y_dnn[3, :, 1], 'c-', label="pi/2")
plt.plot(y_d, command_y_inverse[3, :, 1], 'c:')
plt.plot(y_d, command_y_dnn[4, :, 1], 'm-', label="pi")
plt.plot(y_d, command_y_inverse[4, :, 1], 'm:')
plt.xlabel('y_d [m]')
plt.ylabel('w_z [rad/s]')
plt.ylim([-11, 11])
plt.legend()
plt.grid()

yaw = np.linspace(-2 * np.pi, 2 * np.pi, 100)

for k in np.arange(0, 100):
    command_yaw_dnn[0, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [-1, 0]), -10, 10)
    command_yaw_inverse[0, k, :] = np.clip(inverse.control([0, 0, yaw[k]], [-1, 0]), -10, 10)
    command_yaw_dnn[1, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [1, 0]), -10, 10)
    command_yaw_inverse[1, k, :] = np.clip(inverse.control([0, 0, yaw[k]], [1, 0]), -10, 10)
    command_yaw_dnn[2, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [0, -1]), -10, 10)
    command_yaw_inverse[2, k, :] = np.clip(inverse.control([0, 0, yaw[k]], [0, -1]), -10, 10)
    command_yaw_dnn[3, k, :] = np.clip(dnn.control([0, 0, yaw[k]], [0, 1]), -10, 10)
    command_yaw_inverse[3, k, :] = np.clip(inverse.control([0, 0, yaw[k]], [0, 1]), -10, 10)

plt.figure(5)
plt.title('yaw control')
plt.plot(yaw, command_yaw_dnn[0, :, 1], 'b-', label="pi")
plt.plot(yaw, command_yaw_inverse[0, :, 1], 'b:')
plt.plot(yaw, command_yaw_dnn[1, :, 1], 'g-', label="0")
plt.plot(yaw, command_yaw_inverse[1, :, 1], 'g:')
plt.plot(yaw, command_yaw_dnn[2, :, 1], 'r-', label="-pi/2")
plt.plot(yaw, command_yaw_inverse[2, :, 1], 'r:')
plt.plot(yaw, command_yaw_dnn[3, :, 1], 'c-', label="pi/2")
plt.plot(yaw, command_yaw_inverse[3, :, 1], 'c:')
plt.xlabel('yaw [rad]')
plt.ylabel('w_z [rad/s]')
plt.ylim([-11, 11])
plt.legend()
plt.grid()

plt.show()



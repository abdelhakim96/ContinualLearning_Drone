# MAIN

import numpy as np

from dnn import DNN

dnn = DNN('dnn_32x32')

pose = np.array([[-1, -1, 0],
                [0, 0, 0]])
trajectory = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])

print(trajectory.shape)

#for k in np.arange(1, k_end - 1):
k = 0
print(trajectory[k:k + 2, :])
print(dnn.control(pose[k, :], trajectory[k:k + 3, :]))

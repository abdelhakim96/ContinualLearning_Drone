# Hierarchical DNN-based controller
import numpy as np


class DNN:

    def __init__(self, dt):
        self.dt = dt

        scipy.io.loadmat('data/dnn_32x32', 'net', 'typeScaling', 'data_min', 'data_max', 'mu', 'sigma')

        self.numIputs = 6

    def control(self, pose, trajectory):

        # Actual state

        x = pose(1)
        y = pose(2)
        yaw = pose(3)

        # Reference values

        x_ref1 = trajectory(1, 1)
        y_ref1 = trajectory(1, 2)
        x_ref2 = trajectory(2, 1)
        y_ref2 = trajectory(2, 2)

        # Compute inputs to DNN

        dnnInput = np.array([x_ref2 - x, y_ref2 - y, np.cos(yaw), np.sin(yaw), x_ref1 - x, y_ref1 - y])
        dnnInput = np.amin(np.amax(dnnInput, self.data_min(np.arange(1, self.numIputs + 1))), self.data_max(np.arange(1, self.numIputs + 1)))

        if self.typeScaling == 1:
            dnnInput = (dnnInput - self.mu(np.arange(1, self.numIputs + 1)))/self.sigma(np.arange(1, self.numIputs + 1))
        else:
            if self.typeScaling == 2:
                dnnInput = 2*(dnnInput - self.data_min(np.arange(1, self.numIputs + 1)))/(self.data_max(np.arange(1, self.numIputs + 1)) - self.data_min(np.arange(1, self.numIputs + 1))) - 1

        # Predict the commands

        command = np.transpose(predict(self.net, np.transpose(dnnInput), 'MiniBatchSize', 1))

        # unscale data
        if self.typeScaling == 1:
            command = np.multiply(command, self.sigma(np.arange(self.numIputs + 1, end() + 1))) + self.mu(np.arange(self.numIputs + 1, end() + 1))
        else:
            if self.typeScaling == 2:
                command = np.multiply((command + 1), (self.data_max(np.arange(self.numIputs + 1, end() + 1)) - self.data_min(np.arange(self.numIputs + 1, end() + 1)))) / 2 + self.data_min(np.arange(self.numIputs + 1, end() + 1))

        #command[2] = 0

        commands = [0, 0]
        return commands

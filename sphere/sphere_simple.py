# Unicycle dynamics

import numpy as np
import math


class Sphere:

    def __init__(self, dt, init=None):
        if init is None:
            init = [0, 0, 0]

        self.dt = dt

        # Local parameters
        self.r = 0.5  # wheel radius

        # Bounds
        self.max_input = 1000  # input

        # Initialize state
        self.state = np.zeros(3)
        self.state[0] = init[0]
        self.state[1] = init[0]
        self.state[2] = init[0]

    def __getstate__(self):
        return self.state

    def simulate(self, command=None, uncertainty=0, disturbance=0, noise=0):
        if command is None:
            command = [0, 0, 0]

        # Get parameters
        r = self.r * 2 ** uncertainty

        # Bound commands
        w_x = min(max(command[0], -self.max_input / 2), self.max_input) + disturbance
        w_y = min(max(command[1], -self.max_input / 2), self.max_input) + disturbance
        w_z = min(max(command[2], -self.max_input / 2), self.max_input) + disturbance

        # System dynamics
        dstate = np.zeros(3)
        v_x = r * w_y
        v_y = -r * w_x
        dstate[0] = np.cos(self.state[2]) * v_x - np.sin(self.state[2]) * v_y
        dstate[1] = np.sin(self.state[2]) * v_x + np.cos(self.state[2]) * v_y
        dstate[2] = w_z

        # update state
        self.state = self.state + self.dt * dstate

        # normalise orientation between [-pi and pi]
        if np.abs(self.state[2]) > math.pi:
            self.state[2] -= 2 * math.pi * np.sign(self.state[2])

        # add Gaussian noise
        pose = self.state[:3]

        return pose + np.random.normal(0, noise, 3), pose

# Unicycle dynamics

import numpy as np
import math


class Unicycle:

    def __init__(self, dt, init=None):
        if init is None:
            init = [0, 0, 0]

        self.dt = dt

        # Local parameters
        self.init_r = 0.5   # initial wheel radius
        self.r = self.init_r  # actual wheel radius

        # Bounds
        self.max_input = 100  # input

        # Initialize state
        self.state = np.zeros(3)
        self.state[0] = init[0]
        self.state[1] = init[0]
        self.state[2] = init[0]

        self.disturbance = 0
        self.noise = [0, 0, 0]

    def get_state(self):
        return self.state

    def simulate(self, command=None, uncertainty=0, disturbance=0, noise=0):
        if command is None:
            command = [0, 0]

        self.disturbance = disturbance

        # Get parameters
        self.r = self.init_r * 2 ** uncertainty

        # Bound commands
        w_y = min(max(command[0], -self.max_input/2), self.max_input) + disturbance
        w_z = min(max(command[1], -self.max_input), self.max_input) + disturbance

        # System dynamics
        dstate = np.zeros(3)
        dstate[0] = np.cos(self.state[2]) * self.r * w_y
        dstate[1] = np.sin(self.state[2]) * self.r * w_y
        dstate[2] = w_z

        # update state
        self.state = self.state + self.dt * dstate

        # normalise orientation between [-pi and pi]
        if np.abs(self.state[2]) > np.pi:
            self.state[2] -= 2 * math.pi * np.sign(self.state[2])

        pose = self.state[:3]
        self.noise = np.random.normal(0, noise, 3)
        return pose + self.noise, pose

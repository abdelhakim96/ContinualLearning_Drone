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
        self.i_x = 1  # wheel rotational inertia around x-axis
        self.i_y = 1  # wheel rotational inertia around y-axis
        self.i_z = 1  # wheel rotational inertia around z-axis

        # Bounds
        self.max_input = 1000  # input
        self.max_acceleration = 100  # acceleration
        self.max_velocity = 10  # velocity

        # Initialize state
        self.state = np.zeros(6)
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
        i_x = self.i_x * 2 ** uncertainty
        i_y = self.i_y * 2 ** uncertainty
        i_z = self.i_z * 2 ** uncertainty

        # Bound commands
        tau_x = min(max(command[0], -self.max_input / 2), self.max_input) + disturbance
        tau_y = min(max(command[1], -self.max_input / 2), self.max_input) + disturbance
        tau_z = min(max(command[2], -self.max_input / 2), self.max_input) + disturbance

        # System dynamics
        dstate = np.zeros(6)
        v_x = r * self.state[4]
        v_y = -r * self.state[3]
        dstate[0] = np.cos(self.state[2]) * v_x - np.sin(self.state[2]) * v_y
        dstate[1] = np.sin(self.state[2]) * v_x + np.cos(self.state[2]) * v_y
        dstate[2] = self.state[5]
        dstate[3] = 1 / i_x * tau_x
        dstate[4] = 1 / i_y * tau_y
        dstate[5] = 1 / i_z * tau_z

        # bound accelerations
        dstate[3] = min(max(dstate[3], -self.max_acceleration / 2), self.max_acceleration)
        dstate[4] = min(max(dstate[4], -self.max_acceleration / 2), self.max_acceleration)
        dstate[5] = min(max(dstate[5], -self.max_acceleration / 2), self.max_acceleration)

        # update state
        self.state = self.state + self.dt * dstate

        # normalise orientation between [-pi and pi]
        if np.abs(self.state[2]) > math.pi:
            self.state[2] -= 2 * math.pi * np.sign(self.state[2])

        # bound velocities
        self.state[3] = min(max(self.state[3], -self.max_velocity / 2), self.max_velocity)
        self.state[4] = min(max(self.state[4], -self.max_velocity / 2), self.max_velocity)
        self.state[5] = min(max(self.state[5], -self.max_velocity / 2), self.max_velocity)

        pose = self.state[:3] + np.random.normal(0, noise, 3)
        return pose

# Unicycle dynamics

import numpy as np
import math


class Unicycle:

    def __init__(self, dt, init=[0, 0, 0]):
        self.dt = dt

        # Local parameters
        self.r = 0.5   # wheel radius
        self.i_y = 1   # wheel rotational inertia around y-axis
        self.i_z = 10  # wheel rotational inertia around z-axis

        # Bounds
        self.max_input = 1000  # input
        self.max_acceleration = 100  # acceleration
        self.max_velocity = 10  # velocity

        # Initialize state
        self.state = np.zeros(5)
        self.state[0] = init[0]
        self.state[1] = init[0]
        self.state[2] = init[0]
        self.state[3] = 0
        self.state[4] = 0

    def __getstate__(self):
        return self.state

    def simulate(self, command=[0, 0], uncertainty=0, disturbance=0, noise=0):

        # Get parameters
        r = self.r * 2 ** uncertainty
        i_y = self.i_y * 2 ** uncertainty
        i_z = self.i_z * 2 ** uncertainty

        # Bound commands
        tau_y = min(max(command[0], -self.max_input), self.max_input) + disturbance
        tau_z = min(max(command[1], -self.max_input), self.max_input) + disturbance

        # System dynamics
        dstate = np.zeros(5)
        dstate[0] = np.cos(self.state[2]) * self.state[3]
        dstate[1] = np.sin(self.state[2]) * self.state[3]
        dstate[2] = self.state[4]
        dstate[3] = r / i_y * tau_y
        dstate[4] = 1 / i_z * tau_z

        # bound accelerations
        dstate[3] = min(max(dstate[3], -self.max_acceleration), self.max_acceleration)
        dstate[4] = min(max(dstate[4], -self.max_acceleration), self.max_acceleration)

        # update state
        self.state = self.state + self.dt * dstate

        # normalise orientation between [-pi and pi]
        if np.abs(self.state[2]) > math.pi:
            self.state[2] -= 2 * math.pi * np.sign(self.state[2])

        # bound velocities
        self.state[3] = min(max(self.state[3], -self.max_velocity), self.max_velocity)
        self.state[4] = min(max(self.state[4], -self.max_velocity), self.max_velocity)

        pose = self.state[:3] + np.random.normal(0, noise, 3)
        return pose

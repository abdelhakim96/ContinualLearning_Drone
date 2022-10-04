# Unicycle dynamics

import numpy as np
import math


class Unicycle:

    def __init__(self, dt, init=None):
        if init is None:
            init = [0, 0, 0]

        self.dt = dt

        # Local parameters
        self.init_r = 0.3                                   # initial wheel radius
        self.init_m = 20                                    # initial mass
        self.init_i_y = self.init_m * self.init_r ** 2      # initial wheel rotational inertia around y-axis
        self.init_i_z = self.init_m * self.init_r ** 2 / 2  # initial wheel rotational inertia around z-axis
        self.r = self.init_r                                # actual wheel radius
        self.i_y = self.init_i_y                            # actual wheel rotational inertia around y-axis
        self.i_z = self.init_i_z                            # actual wheel rotational inertia around z-axis

        # Bounds
        self.max_tau_y = 100  # for input 1
        self.max_tau_z = 100  # for input 2
        self.max_w_y = 20
        self.max_w_z = 90

        # Initialize state
        self.state = np.zeros(5)
        self.state[0] = init[0]
        self.state[1] = init[1]
        self.state[2] = init[2]
        self.state[3] = 0
        self.state[4] = 0

        self.disturbance = [0, 0, 0]
        self.noise = [0, 0, 0]

    def __getstate__(self):
        return self.state

    def simulate(self, command=None, uncertainty=0, disturbance=0, noise=0):
        if command is None:
            command = [0, 0]

        self.disturbance = [disturbance * self.max_w_y * self.init_r / 100,
                            -disturbance * self.max_w_y * self.init_r / 100,
                            disturbance * self.max_w_z / 100]

        # Get parameters
        self.r = self.init_r * 2 ** uncertainty
        self.i_y = self.init_i_y * 2 ** uncertainty
        self.i_z = self.init_i_z * 2 ** uncertainty

        # Get commands
        tau_y = command[0]
        tau_z = command[1]

        # Bound commands
        tau_y = min(max(tau_y, -self.max_tau_y/2), self.max_tau_y)
        tau_z = min(max(tau_z, -self.max_tau_z), self.max_tau_z)

        # System dynamics
        dstate = np.zeros(5)
        dstate[0] = self.r * np.cos(self.state[2]) * self.state[3]
        dstate[1] = self.r * np.sin(self.state[2]) * self.state[3]
        dstate[2] = self.state[4]
        dstate[3] = 1 / self.i_y * tau_y
        dstate[4] = 1 / self.i_z * tau_z
        dstate[0:3] += self.disturbance

        # bound accelerations
        # dstate[3] = min(max(dstate[3], -self.max_alpha_y/2), self.max_alpha_y)
        # dstate[4] = min(max(dstate[4], -self.max_alpha_z), self.max_alpha_z)

        # update state
        self.state = self.state + self.dt * dstate

        # normalise orientation between [-pi and pi]
        if np.abs(self.state[2]) > math.pi:
            self.state[2] -= 2 * math.pi * np.sign(self.state[2])

        # bound velocities
        self.state[3] = min(max(self.state[3], -self.max_w_y/2), self.max_w_y)
        self.state[4] = min(max(self.state[4], -self.max_w_z), self.max_w_z)

        pose = self.state[:3]
        self.noise = np.random.normal(0, noise, 3)
        return pose + self.noise, pose

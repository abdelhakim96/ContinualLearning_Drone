# Unicycle dynamics
import numpy as np
import math


class Unicycle:

    def __init__(self, dt, x_init, y_init, yaw_init):
        self.dt = dt

        # Local parameters
        self.r = 0.5  # wheel radius
        self.i_y = 2  # wheel rotational inertia around y-axis
        self.i_z = 4  # wheel rotational inertia around z-axis

        # Bounds
        self.max_input = 1000  # input
        self.max_acceleration = 100  # acceleration
        self.max_velocity = 10  # velocity

        # Initialize state
        self.state = np.zeros(5)
        self.state[0] = x_init
        self.state[1] = y_init
        self.state[2] = yaw_init
        self.state[3] = 0
        self.state[4] = 0

    def simulate(self, command):

        # Bound commands

        command[0] = min(max(command[0], -self.max_input), self.max_input)
        command[1] = min(max(command[1], -self.max_input), self.max_input)

        # System dynamics

        dstate = np.zeros(5)
        dstate[0] = np.cos(self.state[2])*self.state[3]
        dstate[1] = np.sin(self.state[2])*self.state[3]
        dstate[2] = self.state[4]
        dstate[3] = self.r/self.i_y*command[0]
        dstate[4] = 1/self.i_z*command[1]

        # bound accelerations
        dstate[3] = min(max(dstate[3], -self.max_acceleration), self.max_acceleration)
        dstate[4] = min(max(dstate[4], -self.max_acceleration), self.max_acceleration)

        # update state
        self.state = self.state + self.dt*dstate

        # normalise orientation between [-pi and pi]
        if np.abs(self.state[2]) > math.pi:
            self.state[2] = self.state[2] - 2*math.pi*np.sign(self.state[2])

        # bound velocities
        self.state[3] = min(max(self.state[3], -self.max_velocity), self.max_velocity)
        self.state[4] = min(max(self.state[4], -self.max_velocity), self.max_velocity)

        return self.state[:3]

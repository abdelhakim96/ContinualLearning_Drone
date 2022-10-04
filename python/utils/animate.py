import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Arrow

from utils.data_engineering import get_velocity2


class AnimationUnicycle:

    def __init__(self, reference):

        # You probably won't need this if you're embedding things in a tkinter plot...
        # plt.ion()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([-2.2, 2.2])
        self.ax.set_ylim([-2.1, 1.1])
        self.path_d, = self.ax.plot(reference[:, 0], reference[:, 1], 'g--', zorder=1)  # Returns a tuple of line objects, thus the comma
        self.path, = self.ax.plot([], [], 'b-', zorder=1)
        self.pose, = self.ax.plot([], [], 'b', lw=10, zorder=1)
        self.reference = self.ax.scatter([], [], s=100, c='g', zorder=2)
        v = Arrow(0, 0, 0, 0, color='r')
        self.a_v = self.ax.add_patch(v)
        w = Arrow(0, 0, 0, 0, color='r')
        self.a_w = self.ax.add_patch(w)
        self.tau_y, = self.ax.plot([0, 0], [-1, -1], 'r', lw=10, zorder=3)
        self.tau_z, = self.ax.plot([0, 0], [-1, -1], 'r', lw=10, zorder=3)
        plt.show(block=False)

    def update(self, pose, reference, command):
        self.path.set_xdata(pose[:, 0])
        self.path.set_ydata(pose[:, 1])
        self.pose.set_xdata([pose[-1, 0] + 0.1 * np.cos(pose[-1, 2]), pose[-1, 0] - 0.1 * np.cos(pose[-1, 2])])
        self.pose.set_ydata([pose[-1, 1] + 0.1 * np.sin(pose[-1, 2]), pose[-1, 1] - 0.1 * np.sin(pose[-1, 2])])
        self.reference.set_offsets(reference[:2])

        velocity = get_velocity2(0.001, pose[-2:, :])
        self.a_v.remove()
        v = Arrow(pose[-1, 0], pose[-1, 1],
                  velocity[0] / 6 * np.cos(pose[-1, 2]), velocity[0] / 6 * np.sin(pose[-1, 2]),
                  width=0.1, color='r', zorder=3)
        self.a_v = self.ax.add_patch(v)
        self.a_w.remove()
        w = Arrow(pose[-1, 0], pose[-1, 1],
                  velocity[1] / 9 * np.cos(pose[-1, 2] + np.pi/2), velocity[1] / 9 * np.sin(pose[-1, 2] + np.pi/2),
                  width=0.1, color='r', zorder=3)
        self.a_w = self.ax.add_patch(w)

        self.tau_y.set_ydata([-1, -1 + np.clip(command[0], -100, 100) / 100])
        self.tau_z.set_xdata([0, np.clip(command[1], -100, 100) / 100])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

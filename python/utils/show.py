# Plot results

import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored


def show_plots(t, pose, trajectory, command, command_dnn, command_inverse):

    # Compute position the error

    e_x = trajectory[:, 0] - pose[:, 0]
    e_y = trajectory[:, 1] - pose[:, 1]
    e = np.sqrt(e_x**2 + e_y**2)
    print(colored('Tracking MAE: %.3f m' % np.mean(e), 'red'))

    # Compute orientation error

    e_yaw = trajectory[:, 2] - pose[:, 2]
    e_yaw -= (np.abs(e_yaw) > np.pi) * 2 * np.pi * np.sign(e_yaw)  # normalise angles
    # print(np.mean(np.abs(e_yaw)))

    # Max control inputs

    # print(np.max(np.abs(command[:, 0])), np.max(np.abs(command[:, 1])))

    # Plot 2D trajectory

    plt.figure(1)
    plt.title('2D Trajectory')
    plt.scatter(trajectory[int(len(trajectory) * 1 / 4), 0], trajectory[int(len(trajectory) * 1 / 4), 1], c='r', marker='X')
    plt.scatter(trajectory[int(len(trajectory) * 2 / 4), 0], trajectory[int(len(trajectory) * 2 / 4), 1], c='r', marker='X')
    plt.scatter(trajectory[int(len(trajectory) * 3 / 4), 0], trajectory[int(len(trajectory) * 3 / 4), 1], c='r', marker='X')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'g--', label="desired")
    plt.plot(pose[:-1, 0], pose[:-1, 1], 'b', label="actual")
    #plt.scatter(rotate_x, rotate_y, c='r', marker='X')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    # Plot x trajectory

    plt.figure(2)
    plt.title('x Tracking')
    plt.plot([t[-1] * 1 / 4, t[-1] * 1 / 4], [-2, 2], 'r:')
    plt.plot([t[-1] * 2 / 4, t[-1] * 2 / 4], [-2, 2], 'r:')
    plt.plot([t[-1] * 3 / 4, t[-1] * 3 / 4], [-2, 2], 'r:')
    plt.plot(t, trajectory[:, 0], 'g--', label="desired")
    plt.plot(t[:-1], pose[:-1, 0], 'b', label="actual")
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    # Plot y trajectory

    plt.figure(3)
    plt.title('y Tracking')
    plt.plot([t[-1] * 1 / 4, t[-1] * 1 / 4], [-2, 2], 'r:')
    plt.plot([t[-1] * 2 / 4, t[-1] * 2 / 4], [-2, 2], 'r:')
    plt.plot([t[-1] * 3 / 4, t[-1] * 3 / 4], [-2, 2], 'r:')
    plt.plot(t, trajectory[:, 1], 'g--', label="desired")
    plt.plot(t[:-1], pose[:-1, 1], 'b', label="actual")
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    # Plot yaw trajectory

    # plt.figure(2)
    # plt.title('Yaw Tracking')
    # plt.plot(t, (trajectory[:, 2] / np.pi * 180) % 180 * np.sign(trajectory[:, 2]), 'g--', label="desired")
    # plt.plot(t, (pose[:, 2] / np.pi * 180) % 180 * np.sign(pose[:, 2]), 'b', label="actual")
    # plt.xlabel('t [s]')
    # plt.ylabel('yaw [deg]')
    # plt.legend()
    # plt.grid()
    # plt.show(block=False)

    # Plot Euclidean error

    # plt.figure('Name','Euclidean error','NumberTitle','off')
    # hold('on')
    # grid('on')
    # h1 = plt.plot(t,e_x,'g','linewidth',2)
    # h2 = plt.plot(t,e_y,'b','linewidth',2)
    # h3 = plt.plot(t,e,'r','linewidth',2)
    # plt.legend(np.array([h1,h2,h3]),'$e_x$','$e_y$','$e$','Interpreter','latex','Location','north','Orientation','horizontal','FontSize',15)
    # set(gca,'fontsize',15)
    # set(gca,'TickLabelInterpreter','latex')
    # plt.xlabel('$t$ [s]','interpreter','latex','fontsize',15)
    # plt.ylabel('Euclidean error [m]','interpreter','latex','fontsize',15)

    # Plot control inputs

    command_inverse = np.clip(command_inverse, -10, 10)
    command_dnn = np.clip(command_dnn, -10, 10)

    plt.figure(31)
    plt.title('Control Input')
    plt.plot(t[:-2], command_inverse[:-2, 0], 'g', label="inverse")
    plt.plot(t[:-2], command_dnn[:-2, 0], 'b', label="DNN")
    plt.plot([t[-1] * 1 / 4, t[-1] * 1 / 4], [-10, 10], 'r:')
    plt.plot([t[-1] * 2 / 4, t[-1] * 2 / 4], [-10, 10], 'r:')
    plt.plot([t[-1] * 3 / 4, t[-1] * 3 / 4], [-10, 10], 'r:')
    plt.xlabel('t [s]')
    plt.ylabel('w_y [rad/s]')
    plt.legend()
    plt.grid()

    plt.figure(32)
    plt.title('Control Input')
    plt.plot(t[:-2], command_inverse[:-2, 1], 'g', label="inverse")
    plt.plot(t[:-2], command_dnn[:-2, 1], 'b', label="DNN")
    plt.plot([t[-1] * 1 / 4, t[-1] * 1 / 4], [-10, 10], 'r:')
    plt.plot([t[-1] * 2 / 4, t[-1] * 2 / 4], [-10, 10], 'r:')
    plt.plot([t[-1] * 3 / 4, t[-1] * 3 / 4], [-10, 10], 'r:')
    plt.xlabel('t [s]')
    plt.ylabel('w_z [rad/s]')
    plt.legend()
    plt.grid()

    plt.show()


def show_animation(pose, trajectory):

    # Plot 2D trajectory

    plt.title('2D Trajectory')
    plt.plot(trajectory[-2:, 0], trajectory[-2:, 1], 'g--')
    plt.plot(pose[-2:, 0], pose[-2:, 1], 'bo')
    plt.pause(0.00001)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

# Plot results

import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

import kinematics.unicycle_kinematics as unicycle_kinematics
import dynamics.unicycle_dynamics as unicycle_dynamics
from utils.data_engineering import get_velocity2


def show_trajectory(experiment, controller, pose, trajectory, conditions):
    if controller == 0:
        color = '#556b2f'
    if controller == 1:
        color = '#191970'
    if controller == 2:
        color = '#ff4500'
    if controller == 3:
        color = '#ffd700'
    if controller == 4:
        color = '#00ff7f'
    if controller == 5:
        color = '#00bfff'
    if controller == 6:
        color = '#0000ff'
    if controller == 7:
        color = '#ff1493'

    # Plot 2D trajectory

    plt.figure(experiment)
    plt.title('Experiment' + str(experiment))
    plt.scatter(pose[0, 0], pose[0, 1], c=color, marker='o')
    plt.scatter(pose[-1, 0], pose[-1, 1], c=color, marker='o')
    plt.scatter(pose[conditions[0], 0], pose[conditions[0], 1], c='r', marker='X')
    plt.scatter(pose[conditions[1], 0], pose[conditions[1], 1], c='g', marker='X')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'k--')
    plt.plot(pose[:, 0], pose[:, 1], color, label='controller ' + str(controller))
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)


def show_plots_kinematics(t, pose, trajectory, command, command_dnn, command_inverse):

    # Compute position error

    e_x = trajectory[:, 0] - pose[:, 0]
    e_y = trajectory[:, 1] - pose[:, 1]
    e = np.sqrt(e_x**2 + e_y**2)
    print(colored('Tracking MAE: %.3f m' % np.mean(e), 'red'))

    # Compute orientation error

    e_yaw = trajectory[:, 2] - pose[:, 2]
    e_yaw -= (np.abs(e_yaw) > np.pi) * 2 * np.pi * np.sign(e_yaw)  # normalise angles
    # print(np.mean(np.abs(e_yaw)))

    # Compute difference to the analytical inverse

    unicycle = unicycle_kinematics.Unicycle(0)
    command1_min = -unicycle.max_w_y / 2
    command1_max = unicycle.max_w_y
    command2_min = -unicycle.max_w_z
    command2_max = unicycle.max_w_z

    command[:, 0] = np.clip(command[:, 0], command1_min, command1_max)
    command[:, 1] = np.clip(command[:, 1], command2_min, command2_max)
    command_dnn[:, 0] = np.clip(command_dnn[:, 0], command1_min, command1_max)
    command_dnn[:, 1] = np.clip(command_dnn[:, 1], command2_min, command2_max)
    command_inverse[:, 0] = np.clip(command_inverse[:, 0], command1_min, command1_max)
    command_inverse[:, 1] = np.clip(command_inverse[:, 1], command2_min, command2_max)

    approximation_difference1 = np.mean(np.abs(command_dnn[:-2, 0] - command_inverse[:-2, 0]))
    approximation_difference2 = np.mean(np.abs(command_dnn[:-2, 1] - command_inverse[:-2, 1]))
    print(colored('Approximation difference: [%.1f, %.1f] rad/s' % (approximation_difference1, approximation_difference2), 'blue'))

    # Plot 2D trajectory

    plt.figure(1)
    plt.title('2D Trajectory')
    plt.scatter(pose[0, 0], pose[0, 1], c='b', marker='o')
    plt.scatter(pose[-1, 0], pose[-1, 1], c='b', marker='o')
    plt.scatter(pose[int(len(pose) * 1 / 7), 0], pose[int(len(pose) * 1 / 7), 1], c='r', marker='X')
    plt.scatter(pose[int(len(pose) * 2 / 7), 0], pose[int(len(pose) * 2 / 7), 1], c='r', marker='X')
    plt.scatter(pose[int(len(pose) * 3 / 7), 0], pose[int(len(pose) * 3 / 7), 1], c='r', marker='X')
    plt.scatter(pose[int(len(pose) * 4 / 7), 0], pose[int(len(pose) * 4 / 7), 1], c='g', marker='X')
    plt.scatter(pose[int(len(pose) * 5 / 7), 0], pose[int(len(pose) * 5 / 7), 1], c='g', marker='X')
    plt.scatter(pose[int(len(pose) * 6 / 7), 0], pose[int(len(pose) * 6 / 7), 1], c='g', marker='X')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'k--', label="desired")
    plt.plot(pose[:, 0], pose[:, 1], 'b', label="actual")
    #plt.scatter(rotate_x, rotate_y, c='r', marker='X')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    plt.show()
    return

    # Plot x trajectory

    plt.figure(2)
    plt.title('x Tracking')
    plt.plot(t, trajectory[:, 0], 'k--', label="desired")
    plt.plot(t, pose[:, 0], 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [-2, 2], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [-2, 2], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [-2, 2], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [-2, 2], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [-2, 2], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [-2, 2], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    # Plot y trajectory

    plt.figure(3)
    plt.title('y Tracking')
    plt.plot(t, trajectory[:, 1], 'k--', label="desired")
    plt.plot(t, pose[:, 1], 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [-1, 1], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [-1, 1], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [-1, 1], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [-1, 1], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [-1, 1], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [-1, 1], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    # Plot yaw trajectory

    plt.figure(4)
    plt.title('Yaw Tracking')
    plt.plot(t, trajectory[:, 2] / np.pi * 180, 'k--', label="desired")
    plt.plot(t, pose[:, 2] / np.pi * 180, 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [-180, 180], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [-180, 180], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [-180, 180], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [-180, 180], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [-180, 180], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [-180, 180], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('yaw [deg]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

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

    plt.figure(31)
    plt.title('Control Input')
    plt.plot(t[:-1], command_inverse[:-1, 0], 'c', label="inverse")
    plt.plot(t[:-1], command[:-1, 0], 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [command1_min, command1_max], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [command1_min, command1_max], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [command1_min, command1_max], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [command1_min, command1_max], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [command1_min, command1_max], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [command1_min, command1_max], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('w_y [rad/s]')
    plt.legend()
    plt.grid()

    plt.figure(32)
    plt.title('Control Input')
    plt.plot(t[:-1], command_inverse[:-1, 1], 'c', label="inverse")
    plt.plot(t[:-1], command[:-1, 1], 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [command2_min, command2_max], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [command2_min, command2_max], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [command2_min, command2_max], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [command2_min, command2_max], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [command2_min, command2_max], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [command2_min, command2_max], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('w_z [rad/s]')
    plt.legend()
    plt.grid()

    plt.show()


def show_plots_dynamics(t, pose, trajectory, command, command_dnn, command_inverse):

    # Compute position error

    e_x = trajectory[:, 0] - pose[:, 0]
    e_y = trajectory[:, 1] - pose[:, 1]
    e = np.sqrt(e_x**2 + e_y**2)
    print(colored('Tracking MAE: %.3f m' % np.mean(e), 'red'))

    # Compute orientation error

    e_yaw = trajectory[:, 2] - pose[:, 2]
    e_yaw -= (np.abs(e_yaw) > np.pi) * 2 * np.pi * np.sign(e_yaw)  # normalise angles
    # print(np.mean(np.abs(e_yaw)))

    # Compute difference to the analytical inverse

    unicycle = unicycle_dynamics.Unicycle(0)
    command1_min = -unicycle.max_tau_y
    command1_max = unicycle.max_tau_y
    command2_min = -unicycle.max_tau_z
    command2_max = unicycle.max_tau_z

    command[:, 0] = np.clip(command[:, 0], command1_min, command1_max)
    command[:, 1] = np.clip(command[:, 1], command2_min, command2_max)
    command_dnn[:, 0] = np.clip(command_dnn[:, 0], command1_min, command1_max)
    command_dnn[:, 1] = np.clip(command_dnn[:, 1], command2_min, command2_max)
    command_inverse[:, 0] = np.clip(command_inverse[:, 0], command1_min, command1_max)
    command_inverse[:, 1] = np.clip(command_inverse[:, 1], command2_min, command2_max)

    approximation_difference1 = np.mean(np.abs(command_dnn[:-2, 0] - command_inverse[:-2, 0]))
    approximation_difference2 = np.mean(np.abs(command_dnn[:-2, 1] - command_inverse[:-2, 1]))
    print(colored('Approximation difference: [%.1f, %.1f] rad/s' % (approximation_difference1, approximation_difference2), 'blue'))

    # Plot 2D trajectory

    plt.figure(1)
    plt.title('2D Trajectory')
    plt.scatter(pose[0, 0], pose[0, 1], c='b', marker='o')
    plt.scatter(pose[-1, 0], pose[-1, 1], c='b', marker='o')
    plt.scatter(pose[int(len(pose) * 1 / 7), 0], pose[int(len(pose) * 1 / 7), 1], c='r', marker='X')
    plt.scatter(pose[int(len(pose) * 2 / 7), 0], pose[int(len(pose) * 2 / 7), 1], c='r', marker='X')
    plt.scatter(pose[int(len(pose) * 3 / 7), 0], pose[int(len(pose) * 3 / 7), 1], c='r', marker='X')
    plt.scatter(pose[int(len(pose) * 4 / 7), 0], pose[int(len(pose) * 4 / 7), 1], c='g', marker='X')
    plt.scatter(pose[int(len(pose) * 5 / 7), 0], pose[int(len(pose) * 5 / 7), 1], c='g', marker='X')
    plt.scatter(pose[int(len(pose) * 6 / 7), 0], pose[int(len(pose) * 6 / 7), 1], c='g', marker='X')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'k--', label="desired")
    plt.plot(pose[:, 0], pose[:, 1], 'b', label="actual")
    #plt.scatter(rotate_x, rotate_y, c='r', marker='X')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    # Plot x trajectory

    plt.figure(2)
    plt.title('x Tracking')
    plt.plot(t, trajectory[:, 0], 'k--', label="desired")
    plt.plot(t, pose[:, 0], 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [-2, 2], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [-2, 2], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [-2, 2], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [-2, 2], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [-2, 2], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [-2, 2], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    # Plot y trajectory

    plt.figure(3)
    plt.title('y Tracking')
    plt.plot(t, trajectory[:, 1], 'k--', label="desired")
    plt.plot(t, pose[:, 1], 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [-1, 1], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [-1, 1], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [-1, 1], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [-1, 1], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [-1, 1], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [-1, 1], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    # Plot yaw trajectory

    plt.figure(4)
    plt.title('Yaw Tracking')
    plt.plot(t, trajectory[:, 2] / np.pi * 180, 'k--', label="desired")
    plt.plot(t, pose[:, 2] / np.pi * 180, 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [-180, 180], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [-180, 180], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [-180, 180], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [-180, 180], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [-180, 180], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [-180, 180], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('yaw [deg]')
    plt.legend()
    plt.grid()
    plt.show(block=False)

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

    plt.figure(31)
    plt.title('Control Input')
    plt.plot(t[:-1], command_inverse[:-1, 0], 'c', label="inverse")
    plt.plot(t[:-1], command[:-1, 0], 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [command1_min, command1_max], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [command1_min, command1_max], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [command1_min, command1_max], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [command1_min, command1_max], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [command1_min, command1_max], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [command1_min, command1_max], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('tau_y [N m]')
    plt.legend()
    plt.grid()

    plt.figure(32)
    plt.title('Control Input')
    plt.plot(t[:-1], command_inverse[:-1, 1], 'c', label="inverse")
    plt.plot(t[:-1], command[:-1, 1], 'b', label="actual")
    plt.plot([t[-1] * 1 / 7, t[-1] * 1 / 7], [command2_min, command2_max], 'r:')
    plt.plot([t[-1] * 2 / 7, t[-1] * 2 / 7], [command2_min, command2_max], 'r:')
    plt.plot([t[-1] * 3 / 7, t[-1] * 3 / 7], [command2_min, command2_max], 'r:')
    plt.plot([t[-1] * 4 / 7, t[-1] * 4 / 7], [command2_min, command2_max], 'g:')
    plt.plot([t[-1] * 5 / 7, t[-1] * 5 / 7], [command2_min, command2_max], 'g:')
    plt.plot([t[-1] * 6 / 7, t[-1] * 6 / 7], [command2_min, command2_max], 'g:')
    plt.xlabel('t [s]')
    plt.ylabel('tau_z [N m]')
    plt.legend()
    plt.grid()

    plt.show()


def show_animation(pose, trajectory, command):

    # Plot 2D trajectory

    plt.title('2D Trajectory')
    plt.plot(trajectory[-2:, 0], trajectory[-2:, 1], 'g--')
    plt.plot(pose[-2:, 0], pose[-2:, 1], 'bo')
    plt.pause(0.00001)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    velocity = get_velocity2(0.001, pose[-2:, :])
    print('tau_y = %.1f, \t tau_z = %.1f, \t v = %.1f, \t w = %.1f' %
          (command[0], command[1], velocity[0][0], velocity[0][1]))

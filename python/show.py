# Plot results

import numpy as np
import matplotlib.pyplot as plt


def show(t, pose, trajectory, command):
    # Compute the error

    e_x = trajectory[:, 0] - pose[:, 0]
    e_y = trajectory[:, 1] - pose[:, 1]
    e = np.sqrt(e_x**2 + e_y**2)
    print(np.mean(e))

    # Plot 3D trajectory

    plt.title('2D Trajectory')
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="desired")
    plt.plot(pose[:, 0], pose[:, 1], label="actual")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.show()

    # Plot x trajectory fuzzy

    # plt.figure('Name','Trajectories','NumberTitle','off')
    # subplot(3,1,1)
    # hold('on')
    # grid('on')
    # h1 = plt.plot(t,trajectory(:,1),'k--','linewidth',2)
    # h2 = plt.plot(t,pose(:,1),'g','linewidth',2)
    # axis([0 max(t) -3 3]);
    # set(gca,'fontsize',15)
    # set(gca,'TickLabelInterpreter','latex')
    # plt.ylabel('$x$ [m]','interpreter','latex','fontsize',15)

    # Plot y trajectory fuzzy

    # subplot(3,1,2)
    # hold('on')
    # grid('on')
    # h1 = plt.plot(t,trajectory(:,2),'k--','linewidth',2)
    # h2 = plt.plot(t,pose(:,2),'g','linewidth',2)
    # # axis([0 max(t) -3 3]);
    # set(gca,'fontsize',15)
    # set(gca,'TickLabelInterpreter','latex')
    # plt.ylabel('$y$ [m]','interpreter','latex','fontsize',15)

    # Plot yaw trajectory fuzzy

    # subplot(3,1,3)
    # hold('on')
    # grid('on')
    # yaw_ref = atan2(e_y,e_x)
    # plt.plot(t,yaw_ref / pi * 180,'k--','linewidth',2)
    # plt.plot(t,(pose(:,3) - 2 * pi * np.rint(pose(:,3) / pi)) / pi * 180,'g','linewidth',2)
    # #plot(t, command(:,2)/100, 'b', 'linewidth', 2);
    # plt.axis(np.array([0,np.amax(t),- 180,180]))
    # set(gca,'fontsize',15)
    # set(gca,'TickLabelInterpreter','latex')
    # plt.xlabel('$t$ [s]','interpreter','latex','fontsize',15)
    # plt.ylabel('$\theta$ [deg]','interpreter','latex','fontsize',15)

    # Plot Euclidean error fuzzy

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

    # plt.figure('Name','Control inputs','NumberTitle','off')
    # hold('on')
    # grid('on')
    # h1 = plt.plot(t,command(:,1),'g','linewidth',2)
    # h2 = plt.plot(t,command(:,2),'b','linewidth',2)
    # plt.legend(np.array([h1,h2]),'$\tau_y$','$\tau_z$','Interpreter','latex','Location','north','Orientation','horizontal','FontSize',15)
    # set(gca,'fontsize',15)
    # set(gca,'TickLabelInterpreter','latex')
    # plt.xlabel('$t$ [s]','interpreter','latex','fontsize',15)
    # plt.ylabel('Euclidean error [m]','interpreter','latex','fontsize',15)

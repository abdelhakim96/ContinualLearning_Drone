function show(t, pose, trajectory, command)

%% Compute the error

e_x = trajectory(:, 1) - pose(:, 1);
e_y = trajectory(:, 2) - pose(:, 2);
e = sqrt(e_x.^2 + e_y.^2);

mean(e)

%% Plot 3D trajectory

figure('Name', '2D Trajectory', 'NumberTitle', 'off');
hold on;
grid on;
h1 = plot(trajectory(:, 1), trajectory(:, 2), 'k--', 'linewidth', 2);
h2 = plot(pose(:, 1), pose(:, 2), 'g', 'linewidth', 2);
set(gca, 'fontsize', 15);
% axis ([-3 3 -3 3]);
legend([h1, h2], 'desired', 'actual', 'Orientation', 'horizontal', 'Location', 'north', 'FontSize', 15);
xlabel('$x$ [m]', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$y$ [m]', 'interpreter', 'latex', 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex');

%% Plot x trajectory fuzzy
figure('Name', 'Trajectories', 'NumberTitle', 'off');
subplot(3,1,1);
hold on;
grid on;
h1 = plot(t, trajectory(:, 1), 'k--', 'linewidth', 2);
h2 = plot(t, pose(:, 1), 'g', 'linewidth', 2);
% axis([0 max(t) -3 3]);
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex')
ylabel('$x$ [m]', 'interpreter', 'latex', 'fontsize', 15);

%% Plot y trajectory fuzzy
subplot(3,1,2);
hold on;
grid on;
h1 = plot(t, trajectory(:, 2), 'k--', 'linewidth', 2);
h2 = plot(t, pose(:, 2), 'g', 'linewidth', 2);
% axis([0 max(t) -3 3]);
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex')
ylabel('$y$ [m]', 'interpreter', 'latex', 'fontsize', 15);

%% Plot yaw trajectory fuzzy
subplot(3,1,3);
hold on;
grid on;
yaw_ref = atan2(e_y, e_x);
plot(t, yaw_ref/pi*180, 'k--', 'linewidth', 2);
plot(t, (pose(:, 3) - 2*pi*fix(pose(:, 3)/pi))/pi*180, 'g', 'linewidth', 2);
%plot(t, command(:,2)/100, 'b', 'linewidth', 2);
axis([0 max(t) -180 180]);
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex')
xlabel('$t$ [s]', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$\theta$ [deg]', 'interpreter', 'latex', 'fontsize', 15);

%% Plot Euclidean error fuzzy
figure('Name', 'Euclidean error', 'NumberTitle', 'off');
hold on;
grid on;
h1 = plot(t, e_x, 'g', 'linewidth', 2);
h2 = plot(t, e_y, 'b', 'linewidth', 2);
h3 = plot(t, e, 'r', 'linewidth', 2);
legend([h1, h2, h3], '$e_x$', '$e_y$', '$e$', 'Interpreter', 'latex', 'Location', 'north', 'Orientation', 'horizontal', 'FontSize', 15);
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex')
xlabel('$t$ [s]', 'interpreter', 'latex', 'fontsize', 15);
ylabel('Euclidean error [m]', 'interpreter', 'latex', 'fontsize', 15);

%% Plot control inputs
figure('Name', 'Control inputs', 'NumberTitle', 'off');
hold on;
grid on;
h1 = plot(t, command(:,1), 'g', 'linewidth', 2);
h2 = plot(t, command(:,2), 'b', 'linewidth', 2);
legend([h1, h2], '$\tau_y$', '$\tau_z$', 'Interpreter', 'latex', 'Location', 'north', 'Orientation', 'horizontal', 'FontSize', 15);
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex')
xlabel('$t$ [s]', 'interpreter', 'latex', 'fontsize', 15);
ylabel('Euclidean error [m]', 'interpreter', 'latex', 'fontsize', 15);
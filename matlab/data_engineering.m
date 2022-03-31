clear all
close all
clc

%% Parameters



%% Preallocate variables



%% Load data

load('data/samples_step'); % obtained from simulation

%% Process data

time = t;
state = pose;
input = command;

dt = median(diff(time));

%% Plot states

colors = lines(7);

figure('Name', 'States', 'NumberTitle', 'off');
hold on;
grid on;
h1 = plot(time, state(:,1), '-', 'color', colors(2,:), 'linewidth', 1);
h2 = plot(time, state(:,2), '-', 'color', colors(5,:), 'linewidth', 1);
h3 = plot(time, state(:,3), '-', 'color', colors(4,:), 'linewidth', 1);
h4 = plot(time, input(:,1), '-', 'color', colors(6,:), 'linewidth', 2);
h5 = plot(time, input(:,2), '-', 'color', colors(7,:), 'linewidth', 2);

xlim([0 1]);
legend([h1 h2 h3 h4 h5], '$x$', '$y$', '$\theta$', '$v$', '$\omega$', 'Interpreter', 'latex', 'Location', 'southeast', 'Orientation', 'horizontal', 'FontSize', 15);
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('$t$ [s]', 'interpreter', 'latex', 'fontsize', 15);

%% Calculate derived and delayed states

state_diff = zeros(size(state, 1), size(state, 2), 10);
state_diff(:,:,1) = state;

for r = 1:9
    state_diff(1:end - r,:,r + 1) = diff(state, r, 1)/dt;
end

%% Plot correlation

figure('Name', 'omega vs. theta', 'NumberTitle', 'off');
hold on;
grid on;
scatter(input(:,2), state_diff(:,3,2), 'filled');
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('$\omega$ [rad/s]', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$\theta$ [rad]', 'interpreter', 'latex', 'fontsize', 15);

figure('Name', 'v vs. x', 'NumberTitle', 'off');
hold on;
grid on;
scatter(input(:,1), state_diff(:,1,2), 'filled');
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('$v$ [m/s]', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$x$ [m]', 'interpreter', 'latex', 'fontsize', 15);

figure('Name', 'v vs. y', 'NumberTitle', 'off');
hold on;
grid on;
scatter(input(:,1), state_diff(:,2,2), 'filled');
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('$v$ [m/s]', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$y$ [m]', 'interpreter', 'latex', 'fontsize', 15);

figure('Name', 'theta vs. x', 'NumberTitle', 'off');
hold on;
grid on;
scatter(state(:,3), state_diff(:,1,2), 'filled');
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('$\theta$ [m/s]', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$y$ [m]', 'interpreter', 'latex', 'fontsize', 15);

figure('Name', 'theta vs. y', 'NumberTitle', 'off');
hold on;
grid on;
scatter(state(:,3), state_diff(:,2,2), 'filled');
set(gca, 'fontsize', 15);
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('$\theta$ [m/s]', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$y$ [m]', 'interpreter', 'latex', 'fontsize', 15);

%% Calculate correlation

state_input = [state input];

correlation = zeros(size(state, 2), size(state_input, 2), 10);

for r = 0:9
    for i = 1:size(state_input, 2)

        
        for j = 1:size(state, 2)
            X = state_diff(1:end - r,j,r + 1);
            Y = state_input(1:end - r,i);
            
            correlation(j,i,r + 1) = corr(X, Y, 'Type', 'Kendall')'; % ['Pearson', 'Kendall', 'Spearman']

            [fitobject,gof] = fit(X, Y, 'fourier8'); % 'poly1', 'poly2', 'smoothingspline'
            correlation(j,i,r + 1) = gof.rsquare;
        end
    end
end

%% Calculate relative degree for derived states

relative_degree = zeros(size(state, 2), size(state_input, 2));
confidence = zeros(size(state, 2), size(state_input, 2));

for j = 1:size(state, 2)
    for i = 1:size(state_input, 2)
        corr(:) = correlation(j,i,:);
        [~,relative_degree(j,i)] = find(abs(corr) == max(abs(corr)));
        confidence(j,i) = corr(abs(corr) == max(abs(corr)));
    end
end

relative_degree(abs(confidence) < 0.1) = NaN;
relative_degree = relative_degree - 1
relative_degree(:,size(state, 2) + 1:end);

confidence
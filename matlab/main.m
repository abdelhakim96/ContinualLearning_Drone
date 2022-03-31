%% Clear the workspace
clc
clear all
close all

%% Parameters

controller = 1; % 1 - PID controller, 2 - DNN controller
trajectory = 1; % 1 - circular, 2 - set-point, 3 - random points, 4 - square-wave

k_end = 1000;

%% Global variables

global dt;
dt = 0.01;

%% Initialise arrays

pose = zeros(k_end, 5);
command = zeros(k_end, 2);

t = dt*(1:k_end)';

%% Generate trajectory

switch trajectory
    case 1 % circular
        trajectory = [-2*cos(t) 2*sin(t) zeros(k_end, 1)];
    case 2 % set-point
        trajectory = [2*ones(k_end, 1) -2*ones(k_end, 1) zeros(k_end, 1)];
    case 3 % random points (for collecting training data)
        trajectory = [randn(k_end, 1) randn(k_end, 1) randn(k_end, 1)];
    case 4 % square-wave
        d = fix(t);
        d = rem(d, 4);
        b1 = fix(d/2);
        b0 = d - 2*b1;
        trajectory = [b1 b0 zeros(k_end, 1)]; %
end

%% Main loop
for k = 1:k_end - 2

    %% Unicycle control

    switch controller
        case 1 % PID controller
            command(k,:) = controller_pid(pose(k,:), trajectory(k,:));
        case 2 % DNN controller
            command(k,:) = controller_dnn(pose(k,:), trajectory(k + 1:k + 2,:));
    end
    %    command(k,2) = 1; % no heading control (spinning)

    %% Unicycle model

    pose(k + 1,:) = unicycle_dynamic(command(k,:));

end

%% Plot results

show(t(1:end-1), pose(1:end - 1,:), trajectory(1:end - 1,:), command(1:end - 1,:));

%% Save results

save('data/samples_step', 't', 'pose', 'command');
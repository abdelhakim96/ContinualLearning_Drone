function pose = unicycle_dynamic(command)

%% Global parameters

global dt;

%% Local parameters
R   = 0.5;  % wheel radius
I_y = 2;    % wheel rotational inertia around y-axis
I_z = 4;    % wheel rotational inertia around z-axis

max_input = 1000; % input bounds
max_acceleration = 100; % acceleration bounds
max_velocity = 10; % velocity bounds

%% Initial state

x_init = 0;
y_init = 0;
yaw_init = 0;

%% Initialize state

persistent state;

if isempty(state)
    state = zeros(1, 5);
    state(1) = x_init;
    state(2) = y_init;
    state(3) = yaw_init;
end

%% Bound commands

command(1) = min(max(command(1), -max_input), max_input);     % torque around y-axis
command(2) = min(max(command(2), -max_input), max_input);     % torque around z-axis

%% System dynamics

dstate(1) = cos(state(3))*state(4);     % linear velocity along x-axis
dstate(2) = sin(state(3))*state(4);     % linear velocity along y-axis
dstate(3) = state(5);                   % angular velocity around z-axis
dstate(4) = R/I_y*command(1);           % linear acceleration along x-axis
dstate(5) = 1/I_z*command(2);           % angular acceleration around z-axis

% bound accelerations
dstate(4) = min(max(dstate(4), -max_acceleration), max_acceleration);     % linear acceleration along x-axis
dstate(5) = min(max(dstate(5), -max_acceleration), max_acceleration);     % angular acceleration around z-axis

state = state + dt*dstate;

% normalise orientation between [-pi and pi]
if(abs(state(3)) > pi)
    state(3) = state(3) - 2*pi*sign(state(3));
end

% bound velocities
state(4) = min(max(state(4), -max_velocity), max_velocity);     % linear velocity along x-axis
state(5) = min(max(state(5), -max_velocity), max_velocity);     % angular velocity around z-axis

pose = state(1:5);
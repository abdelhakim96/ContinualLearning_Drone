function pose = unicycle(command)

%% Parameters

global dt;

%% Initial state

x_init = 0;
y_init = 0;
yaw_init = 0;

%% Initialize state

persistent state;

if isempty(state)
    state = zeros(1, 3);
    state(1) = x_init;
    state(2) = y_init;
    state(3) = yaw_init;
end

%% Bound commands

command(1) = min([max([command(1) -1000*dt]) 1000*dt]); %
command(2) = min([max([command(2) -1000*2*pi*dt]) 1000*2*pi*dt]); %

%% System dynamics

dstate(1) = cos(state(3))*command(1);     % x_dot
dstate(2) = sin(state(3))*command(1);     % y_dot
dstate(3) = command(2);                   % yaw_dot

state = state + dt*dstate;

% normalise orientation between [-pi and pi]
if(abs(state(3)) > pi)
    state(3) = state(3) - 2*pi*sign(state(3));
end

pose = state;
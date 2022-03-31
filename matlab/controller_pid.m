%% Hierarchical PID controller
function commands = controller_pid(pose, trajectory)

global dt;
global pid_debug;
persistent old_pose i_distance ie_yaw old_velocity ie_v ie_w;

if isempty(old_pose)
    old_pose = zeros(1, 3);
    i_distance = 0;
    ie_yaw = 0;
    old_velocity = zeros(1, 3);
    ie_v = 0;
    ie_w = 0;
end

%% Gains

% position
Kp_p = 10;
Kp_i = 0;
Kp_d = 0;
% orientation
Ko_p = 10;
Ko_i = 0;
Ko_d = 0;
% linear velocity
Kv_p = 100;
Kv_i = 0;
Kv_d = 0;
% angular velocity
Kw_p = 100;
Kw_i = 0;
Kw_d = 0;

%% Actual state

x = pose(1);
y = pose(2);
yaw = pose(3);

%% Reference values

x_ref = trajectory(1);
y_ref = trajectory(2);

%% Compute pose errors

e_x = x_ref - x;
e_y = y_ref - y;

yaw_ref = atan2(e_y, e_x);
e_yaw = yaw_ref - yaw;
if(abs(e_yaw) > pi)
    e_yaw = yaw_ref - yaw - sign(e_yaw)*2*pi;
end

%% Pose controller

distance = cos(e_yaw)*sqrt(e_x^2 + e_y^2);
i_distance = min(max(i_distance + distance*dt, -1), 1);
ie_yaw = min(max(ie_yaw + e_yaw*dt, -1), 1);

vx = (pose(1) - old_pose(1))/dt;
vy = (pose(2) - old_pose(2))/dt;
if (abs(atan2(vy, vx) - yaw) < pi/4) || (abs(atan2(vy, vx) - yaw) > 7/4*pi), direction = 1; else, direction = -1; end % 
v = direction*sqrt(vx^2 + vy^2);
if abs(pose(3) - old_pose(3)) > pi, old_pose(3) = old_pose(3) - 2*pi*sign(old_pose(3)); end % normalise yaw
w = (pose(3) - old_pose(3))/dt;
old_pose = pose;

v_ref = Kp_p*distance + Kp_i*i_distance + Kp_d*v; %
w_ref = Ko_p*e_yaw    + Ko_i*ie_yaw     + Ko_d*w; %

%% Compute velocity error

e_v = v_ref - v;
e_w = w_ref - w;

%% Velocity controller

ie_v = min(max(ie_v + e_v*dt, -1), 1);
ie_w = min(max(ie_w + e_w*dt, -1), 1);

a_v = (v - old_velocity(1))/dt;
a_w = (w - old_velocity(2))/dt;
old_velocity = [v w];

tau_y = Kv_p*e_v + Kv_i*ie_v + Kv_d*a_v; %
tau_z = Kw_p*e_w + Kw_i*ie_w + Kw_d*a_w; %

commands = [tau_y tau_z];
pid_debug = [v, w, v_ref, w_ref, vx, vy];
%% PD controller
function command = controller_dnn(pose, trajectory)

persistent net typeScaling data_min data_max mu sigma;

if isempty(net)
    load('data/dnn_32x32', 'net', 'typeScaling', 'data_min', 'data_max', 'mu', 'sigma');
end

%% Initialize parameters

numIputs = 6;

%% Actual state

x = pose(1);
y = pose(2);
yaw = pose(3);

%% Reference values

x_ref1 = trajectory(1,1);
y_ref1 = trajectory(1,2);

x_ref2 = trajectory(2,1);
y_ref2 = trajectory(2,2);

%% Compute inputs to DNN

dnnInput = [x_ref2 - x y_ref2 - y cos(yaw) sin(yaw) x_ref1 - x y_ref1 - y];

dnnInput = min(max(dnnInput, data_min(1:numIputs)), data_max(1:numIputs)); % bound inputs

if typeScaling == 1
    dnnInput  = (dnnInput - mu(1:numIputs))./sigma(1:numIputs);
end
if typeScaling == 2
    dnnInput = 2 * (dnnInput - data_min(1:numIputs))./(data_max(1:numIputs) - data_min(1:numIputs)) - 1;
end

%% Predict the commands

command = predict(net, dnnInput', 'MiniBatchSize', 1)';

% unscale data
if typeScaling == 1
    command = command.*sigma(numIputs + 1:end) + mu(numIputs + 1:end);
end
if typeScaling == 2
    command = (command + 1).*(data_max(numIputs + 1:end) - data_min(numIputs + 1:end))/2 + data_min(numIputs + 1:end);
end

% command(2) = 0;
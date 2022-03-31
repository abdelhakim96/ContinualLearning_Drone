% clear all
close all
clc
if length(findall(0)) > 1
    delete(findall(0)); % closes the last Training Progress window
end
    
%% Parameters

typeAction = 1; % 1 - train, 2 - test, 3 - train more
typeScaling = 1; % 0 - no scaling, 1 - standardization, 2 - normalization

r = 2;

numHiddenUnits = [32 32];
maxEpochs = 10000; % 1000 -> 18min, 10000 -> 3h
miniBatchSize = 1000;

networkName = 'dnn_32x32';
datasetName = 'samples_random';

trainingPercentage = 0.7;
validationPercentage = 0.1;
testPercentage = 0.2;

%% Allocate space

if exist('checkpoint_networks', 'dir')
    rmdir('checkpoint_networks', 's');
end
mkdir('checkpoint_networks'); % to store temporary DNNs during training

%% Load data

load(['data/', datasetName]); % obtained from simulation
input = pose;
output = command(:,1:2);

%% Process data

% translation invariant: input = [x(k + r_x) - x(k), y(k + r_y) - y(k)];
input(1:end - 1,5:6) = input(1 + 1:end,1:2) - input(1:end - 1,1:2);
input(1:end - r,1:2) = input(1 + r:end,1:2) - input(1:end - r,1:2);

% make orientation periodic with sin and cos
input(:,4) = sin(input(:,3));
input(:,3) = cos(input(:,3));

input(end - max(r):end,:) = [];
output(end - max(r):end,:) = [];

numSamples = size(input, 1);
numInputs = size(input, 2);
numOutputs = size(output, 2);

%% Check input balancing

figure('Name', 'Training Input Histogram', 'NumberTitle', 'off');
subplot(3,1,1);
histogram(input(:,1));
xlabel('$x$ [m]', 'interpreter', 'latex');
subplot(3,1,2);
histogram(input(:,2));
xlabel('$y$ [m]', 'interpreter', 'latex');
subplot(3,1,3);
histogram(acos(input(:,3))/pi*180);
xlabel('$\theta$ [deg]', 'interpreter', 'latex');

%% Split traning, validation and testing samples

index = randperm(numSamples); % randomly split

trainingInput = input(index(1:round(trainingPercentage*numSamples)),:);
trainingOutput = output(index(1:round(trainingPercentage*numSamples)),:);

validationInput = input(index(round(trainingPercentage*numSamples) + 1:round((trainingPercentage + validationPercentage)*numSamples)),:);
validationOutput = output(index(round(trainingPercentage*numSamples) + 1:round((trainingPercentage + validationPercentage)*numSamples)),:);

testInput = input(index(round((1 - testPercentage)*numSamples) + 1:end),:);
testOutput = output(index(round((1 - testPercentage)*numSamples) + 1:end),:);

%% Check output ballancing

figure('Name', 'Outputs Histogram', 'NumberTitle', 'off');
subplot(3,1,1);
histogram(trainingOutput(:,1));
ylabel('training', 'interpreter', 'latex');
subplot(3,1,2);
histogram(validationOutput(:,1));
ylabel('validation', 'interpreter', 'latex');
subplot(3,1,3);
histogram(testOutput(:,1));
ylabel('test', 'interpreter', 'latex');
xlabel('$v$ [m/s]', 'interpreter', 'latex');

figure('Name', 'Outputs Histogram', 'NumberTitle', 'off');
subplot(3,1,1);
histogram(trainingOutput(:,2));
ylabel('training', 'interpreter', 'latex');
subplot(3,1,2);
histogram(validationOutput(:,2));
ylabel('validation', 'interpreter', 'latex');
subplot(3,1,3);
histogram(testOutput(:,2));
ylabel('test', 'interpreter', 'latex');
xlabel('$w$ [rad/s]', 'interpreter', 'latex');

%% Scale data

mu = mean([trainingInput trainingOutput]);
sigma = std([trainingInput trainingOutput]);
sigma(sigma == 0) = 1; % to avoid division for 0
data_min = min([trainingInput trainingOutput]);
data_max = max([trainingInput trainingOutput]);
if typeScaling == 1
    trainingInput  = (trainingInput - mu(1:numInputs))./sigma(1:numInputs);
    trainingOutput  = (trainingOutput - mu(numInputs + 1:end))./sigma(numInputs + 1:end);

    validationInput  = (validationInput - mu(1:numInputs))./sigma(1:numInputs);
    validationOutput  = (validationOutput - mu(numInputs + 1:end))./sigma(numInputs + 1:end);
    
    testInput  = (testInput - mu(1:numInputs))./sigma(1:numInputs);
    testOutput  = (testOutput - mu(numInputs + 1:end))./sigma(numInputs + 1:end);
end
if typeScaling == 2
    trainingInput = 2*(trainingInput - data_min(1:numInputs))./(data_max(1:numInputs) - data_min(1:numInputs)) - 1;
    trainingOutput = 2*(trainingOutput - data_min(numInputs + 1:end))./(data_max(numInputs + 1:end) - data_min(numInputs + 1:end)) - 1;
    
    validationInput = 2*(validationInput - data_min(1:numInputs))./(data_max(1:numInputs) - data_min(1:numInputs)) - 1;
    validationOutput = 2*(validationOutput - data_min(numInputs + 1:end))./(data_max(numInputs + 1:end) - data_min(numInputs + 1:end)) - 1;
    
    testInput = 2*(testInput - data_min(1:numInputs))./(data_max(1:numInputs) - data_min(1:numInputs)) - 1;
    testOutput = 2*(testOutput - data_min(numInputs + 1:end))./(data_max(numInputs + 1:end) - data_min(numInputs + 1:end)) - 1;
end

%% Define DNN

layers = [ ...
    sequenceInputLayer(numInputs)
%     lstmLayer(numHiddenUnits(1)) % can work when the samples are in sequence (not shufled)
    fullyConnectedLayer(numHiddenUnits(1))
    reluLayer
    fullyConnectedLayer(numHiddenUnits(2))
%     reluLayer
%     fullyConnectedLayer(numHiddenUnits(3))
    tanhLayer
    fullyConnectedLayer(numOutputs)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
...%     'InitialLearnRate',0.01, ...
...%     'GradientThreshold',1, ...
    'Plots','training-progress', ...
...% 	'LearnRateDropPeriod',round(maxEpochs/2), ...
...% 	'LearnRateDropFactor',0.2, ...
    'Verbose',1, ...
    'ValidationData',{validationInput',validationOutput'}, ...
    'ValidationFrequency',10, ...
...%    'CheckpointPath','checkpoint_networks', ... % to save checkpoit networks
    'ExecutionEnvironment','auto');

maxNumCompThreads('automatic'); % to leave one core for other stuff

%% Train DNN

if typeAction == 1
    net = trainNetwork(trainingInput', trainingOutput', layers, options);

    save(['data/', networkName], 'net', 'r', 'typeScaling', 'data_min', 'data_max', 'mu', 'sigma');
end

%% Load DNN

if typeAction == 2
    load(['data/', networkName]);   
end

%% Continue training DNN

if typeAction == 3
    load(['data/', networkName]);
    
    net = trainNetwork(trainingInput', trainingOutput', net.Layers, options);
    
    save(['data/', networkName], 'net', 'r', 'typeScaling', 'data_min', 'data_max', 'mu', 'sigma');
end

%% Testing DNN

testPrediction = predict(net, testInput', 'MiniBatchSize', 1)';

% unscale data
if typeScaling == 1
    testOutput = testOutput.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
    testPrediction = testPrediction.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
end
if typeScaling == 2
    testOutput = (testOutput + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
    testPrediction = (testPrediction + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
end

deviation = std(testPrediction - testOutput)
mae = mean(abs(testPrediction - testOutput))

% scale data
if typeScaling == 1
    testOutput  = (testOutput - mu(numInputs + 1:end))./sigma(numInputs + 1:end);
end
if typeScaling == 2
    testOutput = 2*(testOutput - data_min(numInputs + 1:end))./(data_max(numInputs + 1:end) - data_min(numInputs + 1:end)) - 1;
end

%% Check outputs

trainingIndexes = [find(trainingOutput == min(trainingOutput), 1) find(trainingOutput == median(trainingOutput), 1) find(trainingOutput == max(trainingOutput), 1)];
validationIndexes = [find(validationOutput == min(validationOutput), 1) find(validationOutput == median(validationOutput), 1) find(validationOutput == max(validationOutput), 1)];
testIndexes = [find(testOutput == min(testOutput), 1) find(testOutput == median(testOutput), 1) find(testOutput == max(testOutput), 1)];
checkInput = [trainingInput(trainingIndexes,:); validationInput(validationIndexes,:); testInput(testIndexes,:)];
checkOutput = [trainingOutput(trainingIndexes,:); validationOutput(validationIndexes,:); testOutput(testIndexes,:)];

checkPrediction = predict(net, checkInput', 'MiniBatchSize', 1)';

% unscale data
if typeScaling == 1
    checkInput = checkInput.*sigma(1:numInputs) + mu(1:numInputs);
    checkOutput = checkOutput.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
    checkPrediction = checkPrediction.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
end
if typeScaling == 2
    checkInput = (checkInput + 1).*(data_max(1:numInputs) - data_min(1:numInputs))/2 + data_min(1:numInputs);
    checkOutput = (checkOutput + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
    checkPrediction = (checkPrediction + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
end

disp(num2str([checkPrediction, checkOutput, checkInput]));

%% Plot performance

trainingPrediction = predict(net, trainingInput', 'MiniBatchSize', 1)';
validationPrediction = predict(net, validationInput', 'MiniBatchSize', 1)';
testPrediction = predict(net, testInput', 'MiniBatchSize', 1)';

% unscale data
if typeScaling == 1
    trainingOutput = trainingOutput.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
    validationOutput = validationOutput.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
    testOutput = testOutput.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
    trainingPrediction = trainingPrediction.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
    validationPrediction = validationPrediction.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
    testPrediction = testPrediction.*sigma(numInputs + 1:end) + mu(numInputs + 1:end);
end
if typeScaling == 2
    trainingOutput = (trainingOutput + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
    validationOutput = (validationOutput + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
    testOutput = (testOutput + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
    trainingPrediction = (trainingPrediction + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
    validationPrediction = (validationPrediction + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
    testPrediction = (testPrediction + 1).*(data_max(numInputs + 1:end) - data_min(numInputs + 1:end))/2 + data_min(numInputs + 1:end);
end

figure('Name', 'Output', 'NumberTitle', 'off');
hold on;
grid on;
line([data_min(numInputs + 1) data_max(numInputs + 1)], [data_min(numInputs + 1) data_max(numInputs + 1)], 'Color', 'k', 'LineWidth', 1);
scatter(trainingOutput(:,1), trainingPrediction(:,1), 1, 'r', 'filled');
scatter(validationOutput(:,1), validationPrediction(:,1), 1, 'b', 'filled');
scatter(testOutput(:,1), testPrediction(:,1), 1, 'g', 'filled');
xlim([data_min(numInputs + 1) data_max(numInputs + 1)]);
legend('Ideal', 'Training', 'Validation', 'Test', 'Location', 'northwest');
xlabel('$v$ [m/s]', 'interpreter', 'latex');
ylabel('$\tilde{v}$ [m/s]', 'interpreter', 'latex');

figure('Name', 'Output', 'NumberTitle', 'off');
hold on;
grid on;
line([data_min(numInputs + 2) data_max(numInputs + 2)], [data_min(numInputs + 2) data_max(numInputs + 2)], 'Color', 'k', 'LineWidth', 1);
scatter(trainingOutput(:,2), trainingPrediction(:,2), 1, 'r', 'filled');
scatter(validationOutput(:,2), validationPrediction(:,2), 1, 'b', 'filled');
scatter(testOutput(:,2), testPrediction(:,2), 1, 'g', 'filled');
xlim([data_min(numInputs + 2) data_max(numInputs + 2)]);
legend('Ideal', 'Training', 'Validation', 'Test', 'Location', 'northwest');
xlabel('$w$ [rad/s]', 'interpreter', 'latex');
ylabel('$\tilde{w}$ [rad/s]', 'interpreter', 'latex');

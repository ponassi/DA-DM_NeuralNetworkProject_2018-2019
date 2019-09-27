clear 
clc
close all;

%% Settings
% General
IMAGE_DIM = [50 50];
TRANING_SET_PERCENT = 0.6;
TEST_SET_PERCENT = 0.2;
ALPHAS_NUM = 3;

% Directory
normalDir = dir('./Dataset/NORMAL/*.jpeg');
pneumoniaDir = dir('./Dataset/PNEUMONIA/*.jpeg');

%% Reading and manipulating dataset
fprintf("Reading dataset...\n")

nfiles = length(normalDir) + length(pneumoniaDir);

% Datasets sizes
trainingSetSize = floor(TRANING_SET_PERCENT .* nfiles);
testSetSize = floor(TEST_SET_PERCENT .* nfiles);
validationSetSize = nfiles - testSetSize - trainingSetSize;

% Dataset initializing
datasetImages = zeros(IMAGE_DIM(1), IMAGE_DIM(2), nfiles);
datasetAnswers = zeros(nfiles, 1);
randomizer = randperm(nfiles);

% Index to cycle on pneumonia directory
i = 1;

for k = 1 : nfiles
    if(k <= length(normalDir))
        % Read files
        currentFileName = normalDir(k).name;
        currentDir = normalDir(k).folder;
        
        % Read images and convert to gray scale
        currentImage = imread(strcat(currentDir, '/', currentFileName));
        if size(currentImage, 3) == 3
            currentImage = rgb2gray(currentImage);
        end

        % Resize
        currentImage = imresize(currentImage, IMAGE_DIM);
        
        %currentImage = mat2gray(currentImage, [0 255]);
        
        currentImage = im2double(currentImage);
        
        % Standardization
        %currentImage = (currentImage - mean2(currentImage)) ./ std2(currentImage);
        
        % Fill datasets
        datasetImages(:,:,randomizer(k)) = currentImage;
        datasetAnswers(randomizer(k)) = 0;
    else
        % Read files
        currentFileName = pneumoniaDir(i).name;
        currentDir = pneumoniaDir(i).folder;
        
        % Read images and convert to gray scale
        currentImage = imread(strcat(currentDir, '/', currentFileName));
        if size(currentImage, 3) == 3
            currentImage = rgb2gray(currentImage);
        end

        % Resize
        currentImage = imresize(currentImage, IMAGE_DIM);
        
        %currentImage = mat2gray(currentImage, [0 255]);
        
        currentImage = im2double(currentImage);
        
        % Standardization
        %currentImage = (currentImage - mean2(currentImage))./std2(currentImage);
        
        % Fill datasets
        datasetImages(:,:,randomizer(k)) = currentImage;
        datasetAnswers(randomizer(k)) = 1;
        
        i = i + 1;
    end
end

fprintf("Dataset read, standardized and randomized successfully\n")

%% Generating training, test and validation sets
fprintf("Generating training set...\n")

% Traning set
trainingSet = datasetImages(:, :, 1 : trainingSetSize);
X_Train = zeros(trainingSetSize, IMAGE_DIM(1) * IMAGE_DIM(2));
for k = 1 : trainingSetSize
    singleRow = reshape(trainingSet(:, :, k), [1 IMAGE_DIM(1) * IMAGE_DIM(2)]);
    X_Train(k, :) = singleRow;
end
Y_Train = datasetAnswers(1 : trainingSetSize, :);

fprintf("Generating test set...\n")

% Test set
testSet = datasetImages(:, :, trainingSetSize + 1 : testSetSize + trainingSetSize);
X_Test = zeros(testSetSize, IMAGE_DIM(1) * IMAGE_DIM(2));
for k = 1 : testSetSize
    singleRow = reshape(testSet(:, :, k), [1 IMAGE_DIM(1) * IMAGE_DIM(2)]);
    X_Test(k, :) = singleRow;
end

Y_Test = datasetAnswers(trainingSetSize + 1 : testSetSize + trainingSetSize);

fprintf("Generating validation set...\n")

% Validation set
valSet = datasetImages(:, :, testSetSize + trainingSetSize + 1 : nfiles);
X_Val = zeros(validationSetSize, IMAGE_DIM(1) * IMAGE_DIM(2));
for k = 1 : validationSetSize
    singleRow = reshape(valSet(:, :, k), [1 IMAGE_DIM(1) * IMAGE_DIM(2)]);
    X_Val(k, :) = singleRow;
end

Y_Val = datasetAnswers(testSetSize + trainingSetSize + 1 : nfiles);

fprintf("Train, test and validation sets generated successfully\n")

%% Neural Network with backpropagation and sigmoid activation
etha = 0.5;
nEpochs = 25000;

wh = (0.5+0.5) * rand(2500, 4) - 0.5;
wo = (0.5+0.5) * rand(4, 1) - 0.5;
bh = (0.5+0.5) * rand(1, 4) - 0.5;
bo = (0.5+0.5) * rand(1, 1) - 0.5;
lr = 0.1;
momentum = 0.9;

epochAccuracies = zeros(nEpochs, 1);
epochAccuracies_val = zeros(nEpochs, 1);
epochLosses = zeros(nEpochs, 1);
epochLosses_val = zeros(nEpochs, 1);

temp_dcost_wh = 0;
temp_dcost_wo = 0;
temp_dcost_bh = 0;
temp_dcost_bo = 0;

earlyStoppingFlag = true;
bestLossVal = Inf;

for epoch = 1 : nEpochs
    % Forward
    zh = X_Train * wh + bh;
    ah = sigmoid(zh);

    zo = ah * wo + bo;
    ao = sigmoid(zo);
    
    % Validation forward
    zh_val = X_Val * wh + bh;
    ah_val = sigmoid(zh_val);

    zo_val = ah_val * wo + bo;
    ao_val = sigmoid(zo_val);
    
    predicted_val = round(ao_val);
    ypred_val = Y_Val - predicted_val;
    correct_predictions_val = sum(ypred_val == 0);
    
    error_out_val = ((1 / 2) .* (ao_val - Y_Val).^2);
    
    epoch_accuracy_val = (correct_predictions_val .* 100) ./ validationSetSize;
    epochAccuracies_val(epoch) = epoch_accuracy_val;
    epochLosses_val(epoch) = (sum(error_out_val) / validationSetSize) .* 100;
    
    %EarlyStopping
    
    if(mod(epoch, 50) == 0 && earlyStoppingFlag)
        if(epochLosses_val(epoch) < bestLossVal)
            bestLossVal = epochLosses_val(epoch);
            bestWh = wh;
            bestWo = wo;
            bestBh = bh;
            bestBo = bo;
            bestEpoch = epoch;
        else
            earlyStoppingFlag = false;
        end
    end

    % Phase1 =======================

    error_out = ((1 / 2) .* (ao - Y_Train).^2);
    %fprintf("Error: %d\n", sum(error_out) / trainingSetSize);
    
    predicted = round(ao);
    ypred = Y_Train - predicted;
    correct_predictions = sum(ypred == 0);
    
    epoch_accuracy = (correct_predictions .* 100) ./ trainingSetSize;
    fprintf("Accuracy after epoch %d: %.2f with error %.2f\n", epoch, epoch_accuracy, sum(error_out) / trainingSetSize);
    
    epochAccuracies(epoch) = epoch_accuracy;
    epochLosses(epoch) = (sum(error_out) / trainingSetSize) .* 100;
    
    dcost_dao = ao - Y_Train;
    dao_dzo = dsigmoid(zo); 
    dzo_dwo = ah;

    dcost_wo = (dzo_dwo' * (dcost_dao .* dao_dzo)) / trainingSetSize;
    dcost_bo = (ones(trainingSetSize, 1)' * (dcost_dao .* dao_dzo)) / trainingSetSize;

    % Phase 2 =======================

    % dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
    % dcost_dah = dcost_dzo * dzo_dah
    dcost_dzo = dcost_dao .* dao_dzo;
    dzo_dah = wo;
    dcost_dah = dcost_dzo * dzo_dah';
    dah_dzh = dsigmoid(zh);
    dzh_dwh = X_Train;
    dcost_wh = (dzh_dwh' * (dah_dzh .* dcost_dah)) / trainingSetSize;
    dcost_bh = (ones(trainingSetSize, 1)' * (dah_dzh .* dcost_dah)) / trainingSetSize;
    % Update Weights ================

    wh = wh - lr .* dcost_wh - momentum .* temp_dcost_wh;
    wo = wo - lr .* dcost_wo - momentum .* temp_dcost_wo;
    bh = bh - lr .* dcost_bh - momentum .* temp_dcost_bh;
    bo = bo - lr .* dcost_bo - momentum .* temp_dcost_bo;
    
    temp_dcost_wh = + lr .* dcost_wh + momentum .* temp_dcost_wh;
    temp_dcost_wo = + lr .* dcost_wo + momentum .* temp_dcost_wo;
    temp_dcost_bh = + lr .* dcost_bh + momentum .* temp_dcost_bh;
    temp_dcost_bo = + lr .* dcost_bo + momentum .* temp_dcost_bo;
end

%% Test phase
zh_test = X_Test * bestWh + bestBh;
ah_test = sigmoid(zh_test);

zo_test = ah_test * bestWo + bestBo;
ao_test = sigmoid(zo_test);

predicted_test = round(ao_test);
ypred_test = Y_Test - predicted_test;
correct_predictions_test = sum(ypred_test == 0);

error_out_test = ((1 / 2) .* (ao_test - Y_Test).^2);

loss_test = (sum(error_out_test) / testSetSize) .* 100;
accuracy_test = (correct_predictions_test .* 100) ./ testSetSize;

fprintf("Test accuracy %f \n", accuracy_test);
fprintf("Test Loss %f \n", loss_test);

%% Plot Results
%fprintf("Best reached accuracy: %f \n", max(epochAccuracies));
%fprintf("best reached validation accuracy: %f \n", max(accuracyPerEpochVal));
%fprintf("test final accuracy: %f \n", testFinalAccuracy);
figure(1)
box on
hold on
title(['Test accuracy ' num2str(accuracy_test)]);
plot(1:nEpochs,1);
%plot(1:nEpochs,max(epochAccuracies).*ones(nEpochs,1),'r--');
plot(1:nEpochs,epochAccuracies,'b');
plot(1:nEpochs,epochAccuracies_val,'r');
%plot(1:nEpochs,accuracyPerEpochVal,'g');
xlabel("Epoch");
ylabel("Accuracy");
hold off

figure(2)
box on
hold on
title(['Test Loss ' num2str(loss_test)]);
plot(1:nEpochs,1);
plot(1:nEpochs,epochLosses,'b');
plot(1:nEpochs,epochLosses_val,'r');
plot(bestEpoch,bestLossVal,'g*')
xlabel("Epoch");
ylabel("Loss");
hold off


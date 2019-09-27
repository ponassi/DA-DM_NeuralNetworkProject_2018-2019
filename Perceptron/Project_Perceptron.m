clear 
clc
close all;

%% Settings
% General
IMAGE_DIM = [100 100];
TRANING_SET_PERCENT = 0.6;
TEST_SET_PERCENT = 0.2;
ALPHAS_NUM = 1;

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
        
        % Standardization
        currentImage = (currentImage - mean2(currentImage)) ./ std2(currentImage);
        
        % Fill datasets
        datasetImages(:,:,randomizer(k)) = imresize(currentImage, IMAGE_DIM);
        datasetAnswers(randomizer(k)) = -1;
    else
        % Read files
        currentFileName = pneumoniaDir(i).name;
        currentDir = pneumoniaDir(i).folder;
        
        % Read images and convert to gray scale
        currentImage = imread(strcat(currentDir, '/', currentFileName));
        if size(currentImage, 3) == 3
            currentImage = rgb2gray(currentImage);
        end

        %currentImage = mat2gray(currentImage);

        % Standardization
        currentImage = (currentImage - mean2(currentImage))./std2(currentImage);
        
        % Fill datasets
        datasetImages(:,:,randomizer(k)) = imresize(currentImage, IMAGE_DIM);
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

%% Perceptron Algorithm
clear bestW bestAlpha bestAccuracy bestB currentAccuracy w b loops finalWeights finalAccuracy f i;
fprintf("Starting perceptron algorithm...\n")

bestW = zeros(IMAGE_DIM(1) * IMAGE_DIM(2), 1)';
bestAlpha = 1;
bestAccuracy = 0;
bestB = 0;

alphas = logspace(-1, 0, ALPHAS_NUM);

idx = 1;

% 2-D Plot values initialized
xs = alphas;
ys = zeros(size(alphas));

for alpha = alphas
    w = rand(IMAGE_DIM(1) * IMAGE_DIM(2), 1)';
    b = rand;
    i = 1;
    loops = 0;
    
    xs(idx) = alpha;
    
    % TRAINING PHASE
    exit_flag = is_there_error(w, b, Y_Train, X_Train, trainingSetSize);
    count = 0;
    while exit_flag
        f = sign(dot(w, X_Train(i, :)) + b);
        
        if Y_Train(i) * f <= 0
            w = w + alpha * (Y_Train(i) - f) * X_Train(i, :);
            b = b + alpha * (Y_Train(i) - f);
            exit_flag = is_there_error(w, b, Y_Train, X_Train, trainingSetSize);
        end
            i = i + 1;
        if(i > trainingSetSize)
            i = 1;
        end
        loops = loops + 1;
        fprintf("(ALPHA: %d) Learning loops: %d \n", alpha, loops);
    end

    currentAccuracy = calculate_accuracy(X_Val, Y_Val, validationSetSize, w, b);
    
    % VALIDATION PHASE: alpha tuning
    if(currentAccuracy > bestAccuracy)
        bestAccuracy = currentAccuracy;
        bestW = w;
        bestAlpha = alpha;
        bestB = b;
    end
    
    y = currentAccuracy;
    ys(idx) = y;
    idx = idx + 1;
end

% Plot alpha/accuracy result
figure('Name', 'Alpha/Accuracy');
plot(xs, ys, 'b');

%% TEST PHASE: find accuracy
finalAccuracy = calculate_accuracy(X_Test, Y_Test, testSetSize, bestW, bestB);
fprintf("Final accuracy: %f \n", finalAccuracy);

%% Print final weights
figure('Name', 'Final weights');
finalWeights = zeros(IMAGE_DIM(1), IMAGE_DIM(2));
finalWeights(:, :) = reshape(bestW(:), IMAGE_DIM)';

imagesc(finalWeights(:, :)')

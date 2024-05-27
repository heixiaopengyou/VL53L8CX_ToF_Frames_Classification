%% Creating an Augmented DataSet 
% Augmenting Raw Data and Zonal Data from the ToF Sensor, COncatenating Raw and Zonal Data (8x8x2)
Augmented_table = Augmentations(); % takes about 3 mins
%% SHUFFLE AND SPLIT THE DATA TABLE 
% Splitting the data into train, validation and test sets
numRows = size(Augmented_table, 1);
perm = randperm(numRows);
Augmented_table = Augmented_table(perm, :);
[XTrain, TTrain, XVal, TVal, XTest, TTest] = trainWithTabularDataEval(Augmented_table);
DsTrain ={XTrain, TTrain};
DsVal = {XVal, TVal};
DsTest ={XTest, TTest};
%% Creating the Neural Network
% Network with input size 8x8x2
net = createLayers();
analyzeNetwork(net);
%% Save the Unrained Model as ONNX For Deployability Analysis on STM32Cube AI (https://stm32ai-cs.st.com/home)
exportONNXNetwork(net, 'untrained_model.onnx')
%% Training Process
dlnet = Trainingloop(net, DsTrain, DsVal, DsTest);
%% Save the Trained Model as ONNX For Benchmarking on STM32Cube AI
exportONNXNetwork(dlnet, 'trained_model.onnx')
%% Hyper Parameters tuning 
% Switching through several network configurations and number of iterations
params = [
    optimizableVariable('numInitialFilters',[1 8],'Type','integer')
    optimizableVariable('numFinalFilters',[1 16],'Type','integer')
    optimizableVariable('PoolSize',[1 4],'Type','integer')
    optimizableVariable('Stride',[1 4],'Type','integer')
    optimizableVariable('MaxIterations',[100 120], 'Type','integer')
    ];
obj_function = hyperparameterstuning(DsTrain, DsVal, DsTest);
BayesObject = bayesopt(obj_function,params, ...
    'MaxTime',15*60, ...
    'MaxObjectiveEvaluations', 30, ...
    'IsObjectiveDeterministic',false, ...
    'UseParallel',false);
%% Loading The Best HPO Model
 load('0.061278.mat');
%% Save the HPO Model as ONNX For Benchmarking on STM32Cube AI
analyzeNetwork(trainedNet);
exportONNXNetwork(trainedNet, 'HPOtrained_model.onnx')
%% Model Pruning
%% 1. Magnitude Masking Estimation
% Estimating the number of Learnables to prune using Magnitude Masking
% Uses minibatches, thus use the minibatch custom function (minibatch)
[mbqTrain, mbqValidation, mbqTest] = minibatch(DsTrain, DsVal, DsTest);
Magnitude_Pruning(trainedNet, mbqValidation, DsVal, DsTest)
%% 2. Taylor Score Pruning
% Two steps, Pruning the model (based on the Magnitude Pruning Graph) and training the pruned model
[prunedDlNet, AccuracyPrunedNet] = TaylorScore(trainedNet, DsTrain, DsVal, DsTest, 2, 2)
% Visualizing the results from Pruning
statistics = Pruning_Results(trainedNet, prunedDlNet, DsTest)
%% Save the Pruned Model as ONNX For Benchmarking on STM32Cube AI
analyzeNetwork(prunedDlNet);
exportONNXNetwork(PrunedDlNet, 'Pruned_model.onnx')
%% HPO Pruned Model Quantization
qNet = quantizeWorkflow(prunedDlNet, DsTrain)
summary(qNet) % Model Summary
quantizationDetails(qNet)
%% Validation of the Quantized Model
qNet_accuracy = validateQuantizedNet(qNet, DsTest)
%% Save the Quantized Model as ONNX For Benchmarking on STM32Cube AI
analyzeNetwork(qNet);
exportONNXNetwork(qNet, 'Pruned_model.onnx')
%% Classification in Simulink (Image Classifier block)

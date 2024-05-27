function [XTrain, TTrain, XVal, TVal, XTest, TTest] = trainWithTabularDataEval(tbl)
numObservations = size(tbl, 1); 
[idxTrain,idxEval] = trainingPartitions(numObservations,[0.8 0.20]);
tblTrain = tbl(idxTrain,:);
tblEval = tbl(idxEval,:);
% XTrainR = table2array(tblTrain(:, "ToFReading")); 
XTrainR = table2array(tblTrain(:, "Augmented_data"));
XTrain = zeros(8, 8, 2, length(XTrainR));
for idxT = 1:length(XTrainR)
    XTrain(:, :, :, idxT) = XTrainR{idxT}; %#ok<SAGROW>
end
% TTrain = tblTrain.Gesture;
TTrain = tblTrain.Labels;
numObservations = size(tblEval, 1); 
[idxVal,idxTest] = trainingPartitions(numObservations,[0.50 0.50]);
tblVal = tbl(idxVal,:);
tblTest = tbl(idxTest,:);
% XTestR = table2array(tblTest(:, "ToFReading")); 
XValR = table2array(tblVal(:, "Augmented_data"));
XVal = zeros(8, 8, 2, length(XValR));
for idxR = 1:length(XValR)
    XVal(:, :, :, idxR) = XValR{idxR}; %#ok<SAGROW>
end
% TTest = tblTest.Gesture; 
TVal = tblVal.Labels;
% XTestR = table2array(tblTest(:, "ToFReading")); 
XTestR = table2array(tblTest(:, "Augmented_data"));
XTest = zeros(8, 8, 2, length(XTestR));
for idxR = 1:length(XTestR)
    XTest(:, :, :, idxR) = XTestR{idxR}; %#ok<SAGROW>
end
% TTest = tblTest.Gesture; 
TTest = tblTest.Labels;

function Magnitude_Pruning(dlnet, mbqValidation, DsVal, DsTest)
TTest = DsTest{2};
classNames = categories(TTest);
scores = minibatchpredict(dlnet,DsTest{1});
YTest = scores2label(scores,classNames);
accuracyOfTrainedNet = mean(YTest == TTest)*100;
layerG = layerGraph(dlnet);
layerG = removeLayers(layerG,layerG.OutputNames);
net = dlnetwork(layerG);
prunableNet = taylorPrunableNetwork(net);
maxPrunableFilters = prunableNet.NumPrunables;
numTotalParams = sum(cellfun(@numel,net.Learnables.Value));
numNonZeroPerParam = cellfun(@(w)nnz(extractdata(w)),net.Learnables.Value);
initialSparsity = (1-(sum(numNonZeroPerParam)/numTotalParams))*100;
numIterations = 10; 
targetSparsity = 0.90;
iterationScheme = linspace(0,targetSparsity,numIterations);
pruningMaskMagnitude = cell(1,numIterations); 
pruningMaskMagnitude{1} = dlupdate(@(p)true(size(p)), net.Learnables);
figure
plot(100*iterationScheme([1,end]),accuracyOfTrainedNet*[1 1],'*-b','LineWidth',2,"Color","b")
ylim([0 100])
xlim(100*iterationScheme([1,end]))
xlabel("Sparsity (%)")
ylabel("Accuracy (%)")
legend("Original Accuracy","Location","southwest")
title("Pruning Accuracy")    
grid on
lineAccuracyPruningMagnitude = animatedline('Color','g','Marker','o','LineWidth',1.5);
legend("Original Accuracy","Magnitude Pruning Accuracy","Location","southwest")
% Compute magnitude scores
scoresMagnitude = calculateMagnitudeScore(net);
for idx = 1:numel(iterationScheme)
    prunedNetMagnitude = net;
    
    % Update the pruning mask
    pruningMaskMagnitude{idx} = calculateMask(scoresMagnitude,iterationScheme(idx));
    
    % Check the number of zero entries in the pruning mask
    numPrunedParams = sum(cellfun(@(m)nnz(~extractdata(m)),pruningMaskMagnitude{idx}.Value));
    sparsity = numPrunedParams/numTotalParams;
    
    % Apply pruning mask to network parameters
    prunedNetMagnitude.Learnables = dlupdate(@(W,M)W.*M, prunedNetMagnitude.Learnables, pruningMaskMagnitude{idx});
    
    % Compute validation accuracy on pruned network
    accuracyMagnitude = modelAccuracy(prunedNetMagnitude,mbqValidation,classNames,length(DsVal{1}));
    
    % Display the pruning progress
    addpoints(lineAccuracyPruningMagnitude,100*sparsity,accuracyMagnitude)
    drawnow
end
disp('Number of Filters in the Original Model (Max to Prune)')
disp(maxPrunableFilters)
disp(' ')
disp('Initial Sparsity')
disp(initialSparsity)
end

function [statistics] = analyzeNetworkMetrics(originalNet,prunedNet,accuracyOriginal,accuracyPruned)
originalNetMetrics = estimateNetworkMetrics(originalNet);
prunedNetMetrics = estimateNetworkMetrics(prunedNet);
% Accuracy of original network and pruned network
perChangeAccu = 100*(accuracyPruned - accuracyOriginal)/accuracyOriginal;
AccuracyForNetworks = [accuracyOriginal;accuracyPruned;perChangeAccu];
% Total learnables in both networks
originalNetLearnables = sum(originalNetMetrics(1:end,"NumberOfLearnables").NumberOfLearnables);
prunedNetLearnables = sum(prunedNetMetrics(1:end,"NumberOfLearnables").NumberOfLearnables);
percentageChangeLearnables = 100*(prunedNetLearnables - originalNetLearnables)/originalNetLearnables;
LearnablesForNetwork = [originalNetLearnables;prunedNetLearnables;percentageChangeLearnables];
% Approximate parameter memory
approxOriginalMemory = sum(originalNetMetrics(1:end,"ParameterMemory (MB)").("ParameterMemory (MB)"));
approxPrunedMemory = sum(prunedNetMetrics(1:end,"ParameterMemory (MB)").("ParameterMemory (MB)"));
percentageChangeMemory = 100*(approxPrunedMemory - approxOriginalMemory)/approxOriginalMemory;
NetworkMemory = [ approxOriginalMemory; approxPrunedMemory; percentageChangeMemory];
% Create the summary table
statistics = table(LearnablesForNetwork,NetworkMemory,AccuracyForNetworks, ...
    'VariableNames',["Network Learnables","Approx. Network Memory (MB)","Accuracy"], ...
    'RowNames',{'Original Network','Pruned Network','Percentage Change'});
end

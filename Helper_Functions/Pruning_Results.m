function [statistics] = Pruning_Results(dlnet, prunedDlNet, DsTest)
TTest = DsTest{2};
classNames = categories(TTest);
scores = minibatchpredict(dlnet,DsTest{1});
YTest = scores2label(scores,classNames);
accuracyOfTrainedNet = mean(YTest == TTest)*100;
scorespruned = minibatchpredict(prunedDlNet,DsTest{1});
YTestpruned = scores2label(scorespruned,classNames);
AccuracyPrunedNet = mean(YTestpruned == TTest)*100;
[originalNetFilters,layerNames] = numConvLayerFilters(dlnet);
prunedNetFilters = numConvLayerFilters(prunedDlNet);
figure("Position",[10,10,900,900])
bar([originalNetFilters,prunedNetFilters])
xlabel("Layer")
ylabel("Number of Filters")
title("Number of Filters Per Layer")
xticks(1:(numel(layerNames)))
xticklabels(layerNames)
xtickangle(90)
ax = gca;
ax.TickLabelInterpreter = "none";
legend("Original Network Filters","Pruned Network Filters","Location","southoutside")
figure
confusionchart(TTest,YTest,Normalization = "row-normalized");
title("Original Network")
figure
confusionchart(TTest,YTestpruned,Normalization = "row-normalized");
title("Pruned Network")
statistics = analyzeNetworkMetrics(dlnet,prunedDlNet,accuracyOfTrainedNet,AccuracyPrunedNet);
end

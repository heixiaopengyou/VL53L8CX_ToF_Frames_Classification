function [loss,pruningGradient,pruningActivations,netGradients,state] = modelLossPruning(prunableNet, X, T)
[dlYPred,state,pruningActivations] = forward(prunableNet,X);
loss = crossentropy(dlYPred,T);
[pruningGradient,netGradients] = dlgradient(loss,pruningActivations,prunableNet.Learnables);
end

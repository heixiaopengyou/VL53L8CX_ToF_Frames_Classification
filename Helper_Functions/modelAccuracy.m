function accuracy = modelAccuracy(net, mbq, classes, numObservations)
% This function computes the model accuracy of a net(dlnetwork) on the minibatchque 'mbq'.
totalCorrect = 0;
classes = int32(categorical(classes));
reset(mbq);
while hasdata(mbq)
    [dlX, Y] = next(mbq);
    dlYPred = extractdata(predict(net, dlX));
    YPred = onehotdecode(dlYPred,classes,1)';
    YReal = onehotdecode(Y,classes,1)';
    miniBatchCorrect = nnz(YPred == YReal);
    totalCorrect = totalCorrect + miniBatchCorrect;
end
accuracy = totalCorrect / numObservations * 100;
end

function qNet = quantizeWorkflow(net, DsTrain)
xTrain = DsTrain{1};
qLocal = dlquantizer(net, ExecutionEnvironment="MATLAB");
calResults = calibrate(qLocal, xTrain); %#ok<NASGU>
qNet = qLocal.quantize();
end

function ObjFcn = hyperparameterstuning(imdsTrain,imdsValidation, imdsTest)
    ObjFcn = @Test_loss;
    function [Test_loss,cons,fileName] = Test_loss(optVars)
    numInitialFilters = optVars.numInitialFilters;
    numFinalFilters = optVars.numFinalFilters;
    MaxIterations = optVars.MaxIterations;
    MaxIterations = MaxIterations / 10;
    MaxIterations = round(MaxIterations);
    MaxIterations = 10*MaxIterations;
    PoolSize = optVars.PoolSize;
    Stride = optVars.Stride;
    Input_Size = [8 8 2];
    options = trainingOptions("lbfgs", ...
        MaxIterations= MaxIterations, ...
        ExecutionEnvironment="cpu", ...
        ValidationData=imdsValidation, ...
        ValidationFrequency=5, ...
        Verbose=false ...
        );
    net = createLayers(Input_Size, numInitialFilters, numFinalFilters, PoolSize, Stride);
    trainedNet = trainnet(imdsTrain,net,'crossentropy',options);
    close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'));
    Test_loss = test_loss(trainedNet, imdsTest);
    fileName = num2str(Test_loss) + ".mat";
    save(fileName,'trainedNet','Test_loss','options')
    cons = [];
    end
end

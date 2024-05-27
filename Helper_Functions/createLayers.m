function net = createLayers(Input_Size, numInitialFilters, numFinalFilters, PoolSize, Stride)
if nargin < 2
        Input_Size = [8 8 2];
end
if nargin < 2
        numInitialFilters = 8;
end
if nargin < 3
        numFinalFilters = 16;
end
if nargin < 4
        PoolSize = 2;
end
if nargin < 5
        Stride = 2;
end
net = dlnetwork;
tempNet = [
    imageInputLayer(Input_Size,"Name","input")
    convolution2dLayer(3,numInitialFilters,"Name","conv1","BiasLearnRateFactor",0,"Padding","same","WeightsInitializer","he")
    maxPooling2dLayer(PoolSize,'Stride',Stride, 'Name','maxp')
    batchNormalizationLayer("Name","bn1")
    reluLayer("Name","relu1")];
net = addLayers(net,tempNet);
tempNet = [    
    convolution2dLayer(3,numFinalFilters,"Name","conv2","BiasLearnRateFactor",0,"Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","bn2")
    % maxPooling2dLayer(2,'Stride',2, 'Name','maxp1')
    ];
net = addLayers(net,tempNet);
tempNet = [
    convolution2dLayer(3,numFinalFilters,"Name","conv0","BiasLearnRateFactor",0,"Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","bn0")
    % maxPooling2dLayer(2,'Stride',2, 'Name','maxp0')
    ];
net = addLayers(net,tempNet);
tempNet = [
    additionLayer(2,"Name","addition_layer")
    reluLayer("Name","relu4")
    fullyConnectedLayer(4,"Name","fc","WeightsInitializer","he")
    softmaxLayer("Name","softmax")];
net = addLayers(net,tempNet);
net = connectLayers(net,"relu1","conv2");
net = connectLayers(net,"relu1","conv0");
net = connectLayers(net,"bn2","addition_layer/in1");
net = connectLayers(net,"bn0","addition_layer/in2");
clear tempNet;
% analyzeNetwork(net)

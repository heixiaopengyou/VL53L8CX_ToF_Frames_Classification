function [nFilters, convNames] = numConvLayerFilters(net)
numLayers = numel(net.Layers);
convNames = [];
nFilters = [];
% Check for convolution layers and extract the number of filters.
for cnt = 1:numLayers
    if isa(net.Layers(cnt),"nnet.cnn.layer.Convolution2DLayer")
        sizeW = size(net.Layers(cnt).Weights);
        nFilters = [nFilters; sizeW(end)];
        convNames = [convNames; string(net.Layers(cnt).Name)];
    end
end
end

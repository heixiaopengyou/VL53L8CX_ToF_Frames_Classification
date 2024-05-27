function Test_loss = test_loss(trainedNet, imdsTest)
classNames = categories(imdsTest{2});
scores = minibatchpredict(trainedNet,imdsTest{1});
YTest = scores2label(scores,classNames);
TTest = imdsTest{2};
Test_accuracy = mean(YTest == TTest);
Test_loss = 1 - Test_accuracy;

function accuracy = validateQuantizedNet(qNet, DsTest)
XTest = DsTest{1};
TTest = DsTest{2};
classNames = categories(TTest);
scoresTest = minibatchpredict(qNet,XTest);
YTest = onehotdecode(scoresTest,classNames,2);
confusionchart(TTest,YTest)
accuracy = 100* mean(YTest == TTest); 
end

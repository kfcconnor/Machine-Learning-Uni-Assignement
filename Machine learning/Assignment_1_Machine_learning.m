examples_dataset = csvread('examples_2.csv', 1);
x = examples_dataset(1:11, 1:end);
t = examples_dataset(end, 1:end);
inputs = x;
targets = t;

net = fitnet([10 10]);
net.divideParam.trainRatio = 65/100;
net.divideParam.testRatio = 35/100;

[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
performance = perform(net,targets,outputs);
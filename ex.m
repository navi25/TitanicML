trainf = csvread('train.csv');
trainf(1,:) = [];
Y = trainf(:,9);
X = trainf(:,2:8);
[m,n] = size(X);
X = [ones(m, 1) X];
plot(X,Y);
[m,n] = size(X);
theta = zeros(n,1);
alpha = 0.00433;
iter = 80000;

[theta,cost]= gradientDescent(X,Y,
theta,alpha,iter);

% Print theta to screen

fprintf('Cost at theta found by gradient Descent: %f\n', cost);

fprintf('theta: \n');

fprintf(' %f \n', theta);

% Load Test data Setup

testF = csvread('test1.csv');
XT = testF;
[mt,nt] = size(XT);
XT = [ones(mt, 1) XT];
YT = predict(theta,XT);
f = 'o2.txt';
open(f);
csvwrite(f, YT);

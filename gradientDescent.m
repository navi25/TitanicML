function [theta,cost] = gradientDescent(X, y, theta, alpha, num_iters)

%GRADIENTDESCENT Performs gradient descent to learn theta

%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by

%   taking num_iters gradient steps with learning rate alpha



% Initialize some useful values

m = length(y); % number of training examples

J_history = zeros(num_iters, 1);

theta_history = theta;



for iter = 1:num_iters

    h = sigmoid(X*theta);

    grad = (X'*(h - y))/m;

    theta = theta - alpha*grad;

end

[cost,gradient] = computeFunction(theta,X,y);

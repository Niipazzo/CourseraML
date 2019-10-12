function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    t = theta'.*X;
	t = t(:,1)+t(:,2);
	t = t - y
	
	%t1 = theta(1) - alpha / m * sum(t.*X(:,1));
	%t2 = theta(2) - alpha / m * sum(t.*X(:,2));
	%theta = [t1; t2]

	s = alpha / m * sum(t.*X);
	theta = theta - s'

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % simple solution:
    % htheta = X * theta;
    % theta0 = theta(1) - alpha / m * sum((htheta - y) .* X(:,1));
    % theta1 = theta(2) - alpha / m * sum((htheta - y) .* X(:,2)); 
    % theta = [theta0; theta1];
    
    % advanced vectorized solution:

    % X is a 97 x 2 matrix
    % y is a 97 x 1 matrix
    % theta is a 2 x 1 matrix
    delta = (1/m) * sum(X .* repmat((X*theta - y), 1, size(X,2)));
    
    theta = (theta' - (alpha*delta))';
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% simple soluton:
% uses element-wise multiplication operation (.*) and the sum operation sum 

% htheta = sigmoid(X * theta);
% J_2 = 1 / m * sum(-y .* log(htheta) - (1 - y) .* log(1 - htheta)) + lambda / (2 * m) * sum(theta(2:end) .^ 2);
% temp = theta;
% temp(1) = 0;
% grad_2 = 1 / m * (X' * (htheta - y) + lambda * temp);

% advanced solution:
% do regular matrix multiplication by first taking the transpose of y
J = (1/m) * sum((-y' * log(sigmoid(X*theta))) - ((1-y)' * log(1 - sigmoid(X*theta)))) + (lambda/(2*m))*sum(theta(2:end) .* theta(2:end));

% Do not regularize the parameter θ(0)
% X = 5000 x 401 vector
% repmat 5000 x 1 vector by size(x, 2) = 5000 x 401 vector
% grad = 1 x 401 vector
grad = (1/m) * sum(X .* repmat((sigmoid(X*theta) - y), 1, size(X, 2)));

% Regularize starting at index 2
grad(:,2:end) = grad(:,2:end) + (lambda/m)*theta(2:end)';

% =============================================================

grad = grad(:);

end

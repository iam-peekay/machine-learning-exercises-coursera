function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


J = (1/m) * sum((-y' * log(sigmoid(X*theta))) - ((1-y)' * log(1 - sigmoid(X*theta)))) + (lambda/(2*m))*sum(theta(2:end) .* theta(2:end));

fprintf('size: %f', size(J));
% Do not regularize the parameter θ(0)
grad = (1/m) * sum(X .* repmat((sigmoid(X*theta) - y), 1, size(X, 2))); 
% grad is a 1x28 dimensional matrix (a gradient for each feature)

fprintf('size 2: %f', size(grad));
% Regularize only starting at index 2
grad(:, 2:end) = grad(:,2:end) + (lambda/m)*theta(2:end)';

end

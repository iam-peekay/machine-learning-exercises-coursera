function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1
% X is a 5000x400 matrix
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
a1 = [ones(m, 1), X]; % 5000 x 401 matrix
z2 = a1 * Theta1'; % 5000 x 25 matrix
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1), a2]; % 5000 x 26 matrix
z3 = a2 * Theta2'; % 5000 x 10 matrix
a3 = sigmoid(z3);

hThetaX = a3; % 5000 x 10 matrix

yVec = zeros(m, num_labels); % 5000 x 10 matrix

for i = 1:m
  yVec(i, y(i)) = 1;
end; 

% using for-loop

for i = 1:m
    term1 = -yVec(i,:) .* log(hThetaX(i,:));
    term2 = (ones(1,num_labels) - yVec(i,:)) .* log(ones(1,num_labels) - hThetaX(i,:));
    J = J + sum(term1 - term2);
end

J_2 = J / m;

% using vectorized
J = (1/m) * sum(sum(-1 * yVec .* log(hThetaX) - (1 - yVec) .* log(1 - hThetaX)));

% add regularization to the cost
regularization = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J + regularization;


% Part 2

for t = 1:m
  % Set the input layer’s values (a(1)) to the t-th training example x(t)
  a1 = [1; X(t,:)']; % a1 is a 401x1 vector, Theta1 is a 25x401 vector

  % For hidden layers
  z2 = Theta1 * a1; % This will be a 25 x 1 vector

  a2 = [1; sigmoid(z2)];  % Add bias row, so a2 is a 26x1 vector, while Theta2 is a 10x26 vector
  
  z3 = Theta2 * a2; % z3 is now a 10x1 vector
  a3 = sigmoid(z3);  % a3 is now a 10x1 vector

  % Indicates whether the current training example belongs to class k (yk = 1), 
  % or if it belongs to a different class (yk = 0)
  yk = ([1:num_labels]==y(t))';  % size of y is 5000x1 and size of yk is 10x1

  % delta for output layer
  delta_3 = a3 - yk; % delta_3 is a 10x1 vector

  % delta for hidden layer 2
  delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)]; % delta_2 is now a 26x1 vector
  % fprintf('\nsize of delta_2: %f', size(delta_2));
  % fprintf('\nsize of Theta2: %f', size(Theta2));

  % Remove the bias row
  delta_2 = delta_2(2:end); % 25 x 1 matrix

  % NOTE: No need for delta_1 since we do not need to get error for input layer

  % Accumulate the gradient from this example
  % Second part of equation after '+' is regularization
  Theta1_grad = Theta1_grad + delta_2 * a1' + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
	Theta2_grad = Theta2_grad + delta_3 * a2' + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
end

% Obtain the (unregularized) gradient 
Theta1_grad = (1/m) * Theta1_grad 
Theta2_grad = (1/m) * Theta2_grad

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

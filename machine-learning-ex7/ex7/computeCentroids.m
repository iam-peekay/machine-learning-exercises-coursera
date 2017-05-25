function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% size of idx is 300 x 1 or m x 1
% size of centroids in this case is 3 x 2 (K x n)
for i = 1:K
  centroid_i = i == idx; % centroid_i is m x 1 matrix

  num_i = sum(centroid_i);
  centroid_i_matrix = repmat(centroid_i, 1, n); % centroid_i_matrix is m x n matrix

  X_centroid_i = X .* centroid_i_matrix; % X_centroid_i is also an m x n since we do element wise multiplication
  centroids(i, :) = sum(X_centroid_i) / num_i;
end


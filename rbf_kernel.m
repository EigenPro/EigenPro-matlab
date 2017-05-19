function K = rbf_kernel(X, Y, gamma)
% Calculate rbf kernel matrix of form exp(-gamma * ||x-y||^2).
%
%   X: [n_sample, n_raw_feature]: data matrix.
%   Y: [n_center, n_raw_feature]: center matrix.
%   gamma: shape paramter of Gaussian (RBF) kernel.

XY = X * Y';
XX = sum(X.^2, 2);
YY = sum(Y.^2, 2);

D = bsxfun(@plus, bsxfun(@plus, -2 * XY, XX), YY');
K = exp(-gamma * D);

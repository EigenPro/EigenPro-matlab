function K = kernel(X, Y, s, ktype)
% Calculate the specified rbf kernel matrix.
%
% Arguments:
%   X: [n_sample, n_raw_feature]: data matrix.
%   Y: [n_center, n_raw_feature]: center matrix.
%   s: kernel bandwidth.
%   ktype: name of kernel type.
%
% Returns:
%   K: [n_sample, n_center]: kernel matrix.

XY = X * Y';
XX = sum(X.^2, 2);
YY = sum(Y.^2, 2);

D2 = bsxfun(@plus, bsxfun(@plus, -2 * XY, XX), YY');
if strcmp(ktype, 'Gaussian')
    K = exp(-D2 / (2 * s^2));
elseif strcmp(ktype, 'Laplace')
    D = sqrt(max(D2,0));
    K = exp(-D / s);
elseif strcmp(ktype, 'Cauchy')
    K = 1 / (1 + D2 / s^2)
end

function [model, max_S, beta] = setup(model, feat, mG, alpha)
% Compute preconditioner and scale factors for Eigenpro iteration
%
% [input]
%	feat: Feature matrix (normally from training data).
%   max_components: Maximum number of components to be used in EigenPro iteration.
%	mG: minimum batch size to fit in memory
%	alpha: Exponential factor (< 1) for eigenvalue ratio
%
% [output]
%	max_S: top-k eigenvalue vector of phi(X) in desecding order.
%	max_kxx: Maximum of k(x,x) where k is the EigenPro kernel.


% Estimate eigenvalues (S) and eigenvectors (V) of the kernel matrix
[S, V] = nystrom_svd(model, feat);
n_subsamples = size(feat, 1);

% Calculate the number of components to be used such that the
% corresponding batch size is bounded by the subsample size and the
% memory size.
max_bs = min(max(n_subsamples/5, mG), n_subsamples);
n_components = sum((1./S).^alpha < max_bs)-1;
if n_components < 2
    n_components = min(size(S,1), 2);
end

model.V = V(:,1:n_components);
scale = (S(1)./S(n_components+1))^alpha;

% Compute part of the preconditioner for step 2 of gradient descent in
% the eigenpro model
model.Q =(1 - (S(n_components+1) ./ S(1:n_components)).^alpha)...
    ./S(1:n_components);

max_S = S(1)/scale;
kxx = 1 - sum(V.^2,2).* n_subsamples;
beta = max(kxx);
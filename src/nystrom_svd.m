function [S, V] = nystrom_svd(model, X)
% Compute the top eigensystem of a kernel operator using Nystrom method.
%   The eigenvectors of the subsample kernel matrix can be directly used to
%   approximate the eigenfunctions of the kernel operator.
%
% [input]
%   model: The Eigenpro model
%   X (n_subsamples, n_features): Subsample feature matrix.
%   n_components: Number of top eigencomponents to be restored.
% [output]
%   S (k): Top k eigenvalues
%   V (k): Top k eigenvectors 

m = size(X, 1);
K = kernel(X, X, model.bandwidth, model.kernel_name);
W = K./m; 

% Ensure matrix is symmetric and calculate top k eigenvalues
W = (W+W')/2;
W = double(W);
[V, S] = eigs(W, model.n_components, 'largestreal');
V = single(V);
S = single(max(1e-7, diag(S)));
V = V(:,1:model.n_components)./sqrt(m);

function [s, V, lambda] = rsvd(X, phi, M, k)
% Subsample randomized SVD based on
%	Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
%	"Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions."
%	SIAM review 53.2 (2011): 217-288.
%
% [input]
%	X: (n_sample, n_feature): feature matrix.
%	phi: feature map.
%	M: subsample size.
%	k: top eigensystem.
%
% [output]
%	s: top-k eigenvalue vector of phi(X) in desecding order.
%	V: top-k eigenvectors of phi(X).
%	lambda: (k+1)-th largest eigenvalue.

n = size(X, 1);
bs = 512;
PXs = {};

inx = randsample(RandStream('mt19937ar', 'Seed', 1), n, M);
for sindex = 1:bs:M
    eindex = min(sindex + bs -1 , M);
    PXs{length(PXs) + 1} = phi(X(inx(sindex:eindex), :));
end
PX = vertcat(PXs{:});

A = PX;
d = size(A, 2);
p = min(2 * (k+1), d);
R = randn(d, p);
Y = A * R;
W = orth(Y);
B = W' * A; % rank(B) <= 2(k+1)
[~, S1, V1] = svd(B, 'econ');

s = sqrt(n / M) * diag(S1(1:k, 1:k));
V = V1(:, 1:k);
lambda = sqrt(n / M) * S1(k+1, k+1);

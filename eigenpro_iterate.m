function [alpha, t] = eigenpro_iterate(rstream, X, Y, alpha, phi, eta, bs, n_epoch, method, k, M, tau)
% Train least squared regression model with feature map
%	by mini-batch EigenPro iteration.
%
% [input]
%   rstream: random number stream.
%   X: [n_example, n_raw_feature]: raw feature matrix.
%   Y: [n_example, n_label]: label matrix.
%	alpha: [n_feature, n_label]: weight matrix.
%   phi: feature map.
%	eta: step size.
%	bs: mini-batch size.
%   method: training method name (Kernel EigenPro or EigenPro).
%   k: the number of eigendirections.
%   tau: damping factor.
%
% [output]
%   alpha: updated weight matrix.
%   t: iteration time.

% Subsampled randomized SVD.
[Lambda, V, lambda] = rsvd(X, phi, M, k);
if strcmp(method, 'Kernel EigenPro')
    eigenpro_eta= eta * sqrt(Lambda(1) / lambda);
elseif strcmp(method, 'EigenPro')
    % Eigenvalues of the design matrix (in Lambda) are
    %   square root of eigenvalues of covariance matrix.
    eigenpro_eta = eta * Lambda(1) / lambda;
end

% EigenPro iteration.
n = size(X, 1);
st = clock;
for epoch = 1:n_epoch
    inx = randperm(rstream, n);

    for sindex = 1:bs:n
        eindex = min(sindex + bs - 1, n);
        mbinx = inx(sindex: eindex);
        batch_x = X(mbinx, :);
        batch_y = Y(mbinx, :);
        batch_px = phi(batch_x);

        if strcmp(method, 'Kernel EigenPro')
            g = 1./ bs * (batch_px * alpha - batch_y);
            alpha(mbinx, :) = alpha(mbinx, :) - eigenpro_eta * g;
            DgT = g' * batch_px * V * diag((1 - sqrt(tau * lambda ./ Lambda)) ./ Lambda) * V';
            alpha = alpha + eigenpro_eta * DgT';

        elseif strcmp(method, 'EigenPro')
            g = 1./ bs * (batch_px.' * (batch_px * alpha - batch_y));
            alpha = alpha - eigenpro_eta * g;
            DgT = g' * V * diag(1 - tau * lambda ./ Lambda) * V';
            alpha = alpha + eigenpro_eta * DgT';
        end
    end
end
t = etime(clock, st);

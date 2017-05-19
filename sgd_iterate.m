function [alpha, t] = sgd_iterate(rstream, X, Y, alpha, phi, eta, bs, n_epoch, method)
% Train least squared regression model with feature map
%	by mini-batch SGD iteration.
%
% [input]
%   rstream: random number stream.
%   X: [n_example, n_raw_feature]: raw feature matrix.
%   Y: [n_example, n_label]: label matrix.
%	alpha: [n_feature, n_label]: weight matrix.
%   phi: feature map.
%	eta: step size.
%	bs: mini-batch size.
%   method: training method name (Pegasos or Linear).
%
% [output]
%   alpha: updated weight matrix.
%   t: iteration time.

n = size(X, 1);
st = clock;
for epoch = 1:n_epoch
    inx = randperm(rstream, n);
    for sindex = 1:bs:n
        eindex = min(sindex + bs, n);
        mbinx = inx(sindex: eindex);
        batch_x = X(mbinx, :);
        batch_y = Y(mbinx, :);
        batch_px = phi(batch_x);

        if strcmp(method, 'Pegasos')
            g = 1./ bs * (batch_px * alpha - batch_y);
            alpha(mbinx, :) = alpha(mbinx, :) - eta * g;

        elseif strcmp(method, 'Linear')
            g = 1./ bs * (batch_px.' * (batch_px * alpha - batch_y));
            alpha = alpha - eta * g;
        end
    end
end
t = etime(clock, st);

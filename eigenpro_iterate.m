function [model] = eigenpro_iterate(model, X, y, pinx)
% Train least squared regression model with feature map
%	by mini-batch EigenPro iteration.
%
% [input]
%   model: Eigenpro model 
% [output]
%   alpha: updated weight matrix.

% EigenPro iteration.
n = size(X, 1);
if ischar(model.alpha) % Not yet fit
    model.alpha = single(zeros(n, size(y, 2)));
end
step = model.eta ./ single(model.batch_size);

for epoch = 1:model.n_epoch
    inx = randperm(model.random_stream, n);

    for sindex = 1:model.batch_size:n
        eindex = min(sindex + model.batch_size - 1, n);
        mbinx = inx(sindex: eindex);
        batch_x = X(mbinx, :);
        batch_y = y(mbinx, :);
        batch_px = model.phi(batch_x);
     
        g = step * (batch_px * model.alpha - batch_y);
        model.alpha(mbinx, :) = model.alpha(mbinx, :) -  g;
        DgT = bsxfun(@times,model.V, model.Q') * (model.V)' * ...
            batch_px(:, pinx)' * g;
        model.alpha(pinx, :) = model.alpha(pinx, :) + DgT;
    end
end

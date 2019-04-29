function err = evaluate(X, Y, alpha, phi)
% Evaluate classification error.
%
% [input]
%   X: [n_example, n_raw_feature]: raw feature matrix.
%   Y: [n_example, n_label]: label matrix.
%   alpha: [n_example, n_label]: weight matrix.
%   phi: feature map.
%
% [output]
%   err: classification error.

n = size(X, 1);
cerr = 0;
bs = 1024;
for sindex = 1:bs:n
    eindex = min(sindex + bs, n);
    batch_x = X(sindex : eindex, :);
    batch_y = Y(sindex : eindex, :);

    batch_px = phi(batch_x);
    pred = batch_px * alpha;
    [~, pred_c] = max(pred, [], 2);
    [~, batch_yc] = max(batch_y, [], 2);

    cerr = cerr + sum(pred_c ~= batch_yc);
end
err = cerr / n;

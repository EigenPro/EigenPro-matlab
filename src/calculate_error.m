function err = calculate_error(guesses, prediction)
% Evaluate classification error.
%
% [input]
%   guesses: guesses
%   correct: correct labels
%
% [output]
%   err: classification error.
[~, pred_c] = max(guesses, [], 2);
[~, batch_yc] = max(prediction, [], 2);
err = sum(pred_c ~= batch_yc) / size(pred_c, 1);

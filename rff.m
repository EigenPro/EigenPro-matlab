function RF = rff(X, R, use_gpu)
% Calculate random Fourier features according to paper,
%   "Random Features for Large-Scale Kernel Machines".
%
%   X: [n_sample, n_raw_feature]: data matrix.
%   R: [n_raw_feature, n_random_feature]: N(0,1) matrix.
%   use_gpu: boolean flag.

XR = X * R;
d = single(size(XR, 2));
if use_gpu
  RF = [cos(XR) sin(XR)] / sqrt(d);
else
  RF = BlockMatrix(cos(XR)/ sqrt(d), sin(XR)/ sqrt(d));
end

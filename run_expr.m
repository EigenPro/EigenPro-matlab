use_gpu = false; % gpu usage flag.
floatx = @(x) single(x);

% Load MNIST dataset.
mnist_path = './data/mnist.mat';
data = load(mnist_path);

if use_gpu
    train_x = gpuArray(floatx(data.train_x));
    train_y = gpuArray(floatx(data.train_y));
    test_x = gpuArray(floatx(data.test_x));
    test_y = gpuArray(floatx(data.test_y));
    new_weight = @(n, d) gpuArray(floatx(zeros(n, d)));
else
    train_x = floatx(data.train_x);
    train_y = floatx(data.train_y);
    test_x = floatx(data.test_x);
    test_y = floatx(data.test_y);
    new_weight = @(n, d) floatx(zeros(n, d));
end

% Parameters.
bs = 256;           % mini-batch size.
eta = floatx(5);    % step size.
s2 = 25;            % variance of Gaussian kernel.
M = 4800;           % (EigenPro) subsample size.
k = 160;            % (EigenPro) top-k eigensystem.

gamma = 1 /(2*s2);  % shape parameter of Gaussian kernel.
n = size(train_x, 1); % number of training samples.
N = size(train_x, 2); % number of raw features.
d = floor(size(train_x, 1) / 2) * 2; % number of random features.
l = size(train_y, 2); % number of labels.

rs = RandStream('mt19937ar', 'Seed', 1);

% Wrap feature map.
phi_rbf = @(x) rbf_kernel(x, train_x, gamma); % RBF kernel map.
normal = floatx(sqrt(2 * gamma) * randn(rs, N, d/2));
if use_gpu
    normal = gpuArray(normal);
end
phi_rff = @(x) rff(x, normal, use_gpu); % random Fourier feature map.

alpha = new_weight(n, l);
alpha_ep = new_weight(n, l);
beta = new_weight(d, l);
beta_ep = new_weight(d, l);
cur_epoch = 0;
pega_t = 0; epro_t = 0; rf_t = 0; repro_t = 0; % timing .
rs.reset();
for n_epoch = [1 5 10 20 40 80 160 320]
  fprintf('\n');
  fprintf('Pegasos on MNIST\n');
  [alpha, t] = sgd_iterate(rs, train_x, train_y, alpha, phi_rbf, eta, ...
                           bs, n_epoch - cur_epoch, 'Pegasos');
  pega_t = pega_t + t;
  train_err = evaluate(train_x, train_y, alpha, phi_rbf) * 100;
  test_err = evaluate(test_x, test_y, alpha, phi_rbf) * 100;
  fprintf('training error %.2f%%, testing_error %.2f%% (%d epochs, %.2f seconds)\n', ...
          train_err, test_err, n_epoch, pega_t);

  fprintf('Kernel EigenPro on MNIST\n');
  [alpha_ep, t] = eigenpro_iterate(rs, train_x, train_y, alpha_ep, phi_rbf, eta, ...
                                   bs, n_epoch - cur_epoch, 'Kernel EigenPro', k, M, floatx(1.));
  epro_t = epro_t + t;
  train_err = evaluate(train_x, train_y, alpha_ep, phi_rbf) * 100;
  test_err = evaluate(test_x, test_y, alpha_ep, phi_rbf) * 100;
  fprintf('training error %.2f%%, testing_error %.2f%% (%d epochs, %.2f seconds)\n', ...
          train_err, test_err, n_epoch, epro_t);

  fprintf('SGD with random Fourier feature on MNIST\n');
  [beta, t] = sgd_iterate(rs, train_x, train_y, beta, phi_rff, eta, ...
                          bs, n_epoch - cur_epoch, 'Linear');
  rf_t = rf_t + t;
  train_err = evaluate(train_x, train_y, beta, phi_rff) * 100;
  test_err = evaluate(test_x, test_y, beta, phi_rff) * 100;
  fprintf('training error %.2f%%, testing_error %.2f%% (%d epochs, %.2f seconds)\n', ...
          train_err, test_err, n_epoch, rf_t);

  fprintf('EigenPro with random Fourier feature on MNIST\n');
  [beta_ep, t] = eigenpro_iterate(rs, train_x, train_y, beta_ep, phi_rff, eta, ...
                                  bs, n_epoch - cur_epoch, 'EigenPro', k, M, floatx(.25));
  repro_t = repro_t + t;
  train_err = evaluate(train_x, train_y, beta_ep, phi_rff) * 100;
  test_err = evaluate(test_x, test_y, beta_ep, phi_rff) * 100;
  fprintf('training error %.2f%%, testing_error %.2f%% (%d epochs, %.2f seconds)\n', ...
          train_err, test_err, n_epoch, repro_t);

  cur_epoch = n_epoch;
end

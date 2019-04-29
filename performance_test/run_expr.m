use_gpu = false; % gpu usage flag.
ktype = 'gaussian'; % kernel type.
floatx = @(x) single(x);

% Load MNIST dataset.
mnist_path = './data/mnist/mnist.mat';
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
M = 4800;           % (EigenPro) subsample size.
k = 160;            % (EigenPro) top-k eigensystem.

% Set kernel bandwidth.
if strcmp(ktype, 'gaussian')
	s = 5;
elseif strcmp(ktype, 'laplace')
	s = floatx(sqrt(10));
elseif strcmp(ktype, 'cauchy')
	s = floatx(sqrt(40));
end

n = size(train_x, 1); % number of training samples.
N = size(train_x, 2); % number of raw features.
d = floor(size(train_x, 1) / 2) * 2; % number of random features.
l = size(train_y, 2); % number of labels.

rs = RandStream('mt19937ar', 'Seed', 1);

% Wrap feature map.
phi_rbf = @(x) kernel(x, train_x, s, ktype); % kernel map.
normal = floatx(randn(rs, N, d/2) / s);
if use_gpu
    normal = gpuArray(normal);
end
% Random Fourier feature map (for Gaussian kernel).
phi_rff = @(x) rff(x, normal, use_gpu);

% Calculate the step size.
[s, V, lambda] = rsvd(train_x, phi_rbf, 100, 2);
eta = 1.5 / (2 * s(1) / n);

alpha = new_weight(n, l);
alpha_ep = new_weight(n, l);
beta = new_weight(d, l);
beta_ep = new_weight(d, l);
cur_epoch = 0;
pega_t = 0; epro_t = 0; rf_t = 0;% for timing.
rs.reset();
ep_model = Eigenpro('random_stream', rs,...
    'subsample_size', M, 'n_components', k, 'mem_gb', 3);
for n_epoch = [1 2 5 10]
  fprintf('\n');
  fprintf('EigenPro on MNIST\n');
  tic;
  ep_model.n_epoch = n_epoch - cur_epoch;
  ep_model = ep_model.fit(train_x, train_y);
  epro_t = epro_t + toc;
  pred_y_train = ep_model.predict(train_x);
  pred_y = ep_model.predict(test_x);
  
  train_err = calculate_error(pred_y_train, train_y) * 100;
  test_err = calculate_error(pred_y, test_y) * 100;
  fprintf('training error %.2f%%, testing_error %.2f%% (%d epochs, %.2f seconds)\n', ...
          train_err, test_err, n_epoch, epro_t);
  % ------------------------------------  
  fprintf('Pegasos on MNIST\n');
  [alpha, t] = sgd_iterate(rs, train_x, train_y, alpha, phi_rbf, eta, ...
                           bs, n_epoch - cur_epoch, 'Pegasos');
  pega_t = pega_t + t;
  train_err = evaluate(train_x, train_y, alpha, phi_rbf) * 100;
  test_err = evaluate(test_x, test_y, alpha, phi_rbf) * 100;
  fprintf('training error %.2f%%, testing_error %.2f%% (%d epochs, %.2f seconds)\n', ...
          train_err, test_err, n_epoch, pega_t);
  % --------------------------------------
  fprintf('SGD with random Fourier feature on MNIST\n');
  [beta, t] = sgd_iterate(rs, train_x, train_y, beta, phi_rff, eta, ...
                          bs, n_epoch - cur_epoch, 'Linear');
  rf_t = rf_t + t;
  train_err = evaluate(train_x, train_y, beta, phi_rff) * 100;
  test_err = evaluate(test_x, test_y, beta, phi_rff) * 100;
  fprintf('training error %.2f%%, testing_error %.2f%% (%d epochs, %.2f seconds)\n', ...
  train_err, test_err, n_epoch, rf_t);

  cur_epoch = n_epoch;
end

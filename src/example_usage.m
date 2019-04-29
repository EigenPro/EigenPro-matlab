use_gpu = false; % gpu usage flag.
ktype = 'gaussian'; % kernel type.

% Load MNIST dataset.
mnist_path = './data/mnist/mnist.mat';
data = load(mnist_path);

train_x = single(data.train_x);
train_y = single(data.train_y);
train_x = train_x(1:10000, :);
train_y = train_y(1:10000, :);

if(use_gpu)
    train_x = gpuArray(train_x);
    train_y = gpuArray(train_y);
end

ep_model = Eigenpro('random_stream', RandStream('mt19937ar', 'Seed', 1),...
    'n_epoch', 1, 'subsample_size', 800, 'n_components', 500);
ep_model = ep_model.fit(train_x, train_y);
pred_y = ep_model.predict(train_x);
test_err = calculate_error(pred_y, train_y) * 100;
disp("Error: "  + test_err + "%");

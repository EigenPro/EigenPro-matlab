function [model, pinx] = initialize_params(model, X, y)
% Set up all of the parameters to this model
if size(size(X), 2) ~= 2
    error("Train data features are not 2D");
end
n = size(X, 1);
d = size(X, 2);

if size(size(y), 2)~=2
    error("Train data labels are not 2D");
else
    n_label = size(y, 2);
end

if size(y, 1) ~= size(X,1)
    error("Train X and Y have different number of rows");
end

model.phi = @(x) kernel(x, X, model.bandwidth, ...
    model.kernel_name);

% Choose subsample size automatically
if ~isnumeric(model.subsample_size) % model.subsample_size=="auto"
    if n < 100000
        model.subsample_size = min(n, 4000);
    else
        model.subsample_size = 12000;
    end
else
    model.subsample_size = min(model.subsample_size, n);
end

% Fix n_components if it has a unreasonable value
model.n_components = min(model.subsample_size - 1, model.n_components);
model.n_components = max(1, model.n_components);

mem_bytes = model.mem_gb * 1024 * 1024 * 1024;
mem_usages = (d + n_label + 2*(0:model.subsample_size))*n*4;
mG = sum(mem_usages<mem_bytes);

% Choose random subsample to use for iteration
pinx = int32(randsample(model.random_stream, n,...
    model.subsample_size));

% Setup preconditioner 
[model, max_S, beta] = setup(model, X(pinx, :), mG, .95);

% Calculate best batch size
if ~isnumeric(model.batch_size) % model.batch_size=="auto"
    model.batch_size = min(int32(beta./max_S)+1, mG);
end
model.batch_size = min(model.batch_size, n);

% Calculate best step size
if model.batch_size < int32(beta./max_S) + 1
    model.eta = single(model.batch_size)./beta;
elseif model.batch_size < n
    model.eta = 2.0 * single(model.batch_size) ./ ...
        (beta + (single(model.batch_size) - 1) * max_S);
else
    model.eta = .95 * 2.0 / max_S;
end
model.eta = single(model.eta);
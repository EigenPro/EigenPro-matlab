classdef Eigenpro
    %EIGENPRO A class implementing EigenPro iteration for regression
    properties
        random_stream   % random number stream.
        batch_size      % mini-batch size.
        n_epoch         % times to iterate over train data
        n_components    % number of eigndirections to use
        subsample_size  % samples to use in estimating eigenstuff
        mem_gb          % physical device memory
        kernel_name     % The kernel used
        bandwidth       % Bandwidth of the kernel
        phi             % Feature mapping
        eta             % step size
        V               % Eigenvectors for Preconditioner
        Q               % Part of the Preconditioner
        alpha = 'None'  % Coefficient weight matrix
    end
    
    methods
        function model = Eigenpro(varargin)
            p = inputParser;
            is_pos = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            is_str_or_char = @(x) ischar(x) || isstring(x);
            is_auto = @(x) is_str_or_char(x) && x=="auto";
            is_pos_or_auto = @(x) is_pos(x)  || is_auto(x);
            is_valid_kernel = @(x) is_str_or_char(x) && ...
                (x=="gaussian" || x=="laplacian" || x=="cauchy");
            addParameter(p,'random_stream', 1);
            addParameter(p,'batch_size','auto', is_pos_or_auto);
            addParameter(p,'n_epoch', 2, is_pos);
            addParameter(p,'n_components', 1000, is_pos);
            addParameter(p,'subsample_size','auto', is_pos_or_auto);
            addParameter(p,'mem_gb', 6, is_pos);
            addParameter(p,'kernel_name', "gaussian", is_valid_kernel);
            addParameter(p,'bandwidth', 5, is_pos);
            parse(p, varargin{:});
            
            % EIGENPRO Construct an instance of this class
            model.batch_size = p.Results.batch_size;
            model.n_epoch = p.Results.n_epoch;
            model.n_components = p.Results.n_components;
            model.subsample_size = p.Results.subsample_size;
            model.mem_gb = p.Results.mem_gb;
            model.kernel_name = p.Results.kernel_name;
            model.bandwidth = p.Results.bandwidth;
            rs = p.Results.random_stream;
            if isnumeric(rs)
                model.random_stream = RandStream('mt19937ar','Seed',rs);
            else
                model.random_stream = rs;
            end
        end
        
        function model = fit(model, X, y)
           % FIT return a model fit to the given data
           %    model = eigenpro model
           %    X = [n_example, n_raw_feature]: raw feature matrix.
           %    y = [n_example, n_label]: label matrix.       
           %    Return: the fitted model
          
          % Set every parameter except alpha to an appropriate value
          [model, pinx] = initialize_params(model, X, y);
          % Train the model, updating alpha
          model = eigenpro_iterate(model, X, y, pinx);
        end
        
        
        function pred_y = predict(model, X)
            %PREDICT Predict targets for this class
            %   Predict using an eigenpro object. Returns data as a
            %   matrix of shape (n_samples, n_targets)
            %
            % [input]
            %   model: The eigenpro model
            %   X: [n_example, n_raw_feature]: raw feature matrix for test data.
            %   phi: feature map.
            %
            % [output]
            %   pred_y: output labels
            if ischar(model.alpha)
               error("Model has not been fit yet. " + ... 
               "Call model = model.fit(X,y) first."); 
            end
            n = size(X, 1);
            pred_y = zeros(n, size(model.alpha, 2));
            
            for sindex = 1:model.batch_size:n
                eindex = min(sindex + model.batch_size, n);
                batch_x = X(sindex : eindex, :);
                
                batch_px = model.phi(batch_x);
                pred = batch_px * model.alpha;
                pred_y(sindex: eindex, :) = pred;
            end
        end
    end
end


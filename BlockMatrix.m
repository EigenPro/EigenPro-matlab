classdef BlockMatrix
  properties
    blocks % blocks(i) has shape (n_row, n_blk_feature).
    istranspose % flag of transpose.
  end
  methods
    function obj = BlockMatrix(varargin)
      obj.blocks = varargin;
      obj.istranspose = false;
    end

    function ret = size(varargin)
      A = varargin{1};
      n = size(A.blocks{1}, 1);
      d = 0;
      for i = 1:length(A.blocks)
        d = d + size(A.blocks{i}, 2);
      end

      if A.istranspose
        shape = [d, n];
      else
        shape = [n, d];
      end

      if length(varargin) == 1
        ret = shape;
      else
        dim = varargin{2};
        assert(dim <= 2);
        ret = shape(dim);
      end
    end

    function T = transpose(A)
      T = BlockMatrix(A.blocks{:});
      T.istranspose = ~A.istranspose;
    end

    function C = mtimes(A, B)
      if isa(A, 'BlockMatrix')
        assert(~isa(B, 'BlockMatrix'));
      else
        C = mtimes(B.', A.').';
        return
      end
      assert(isa(A, 'BlockMatrix'));
      assert(~isa(B, 'BlockMatrix'));

      if A.istranspose
        n_row = 0;
        for i = 1:length(A.blocks)
          n_row = n_row + size(A.blocks{i}, 2);
        end
        C = zeros(n_row, size(B, 2));
        rinx = 1;
        for i = 1:length(A.blocks)
          blk = A.blocks{i}';
          blk_row = size(blk, 1);
          rinxe = rinx + blk_row - 1;
          C(rinx:rinxe, :) = blk * B;
          rinx = rinxe + 1;
        end

      else
        nd = size(A);
        C = zeros(nd(1), size(B, 2));
        rinx = 1;
        for i = 1:length(A.blocks)
          blk = A.blocks{i};
          rinxe = rinx + size(blk, 2) - 1;
          C = C + blk * B(rinx:rinxe, :);
          %y = y + size(blk, 2);
          rinx = rinxe + 1;
        end
      end % if
    end % mtimes()

    function C = vertcat(varargin)
      assert(~varargin{1}.istranspose); % TODO: handle transposed matrix.
      n_mat = length(varargin);
      n_blk = length(varargin{1}.blocks);
      blocks = cell(1, n_blk);
      for i = 1:n_blk
        tocat = cell(1, n_mat);
        for j = 1:n_mat
          tocat{j} = varargin{j}.blocks{i};
        end
        blocks{i} = vertcat(tocat{:});
      end
      C = BlockMatrix(blocks{:});
    end % vertcat()

  end % methods
end

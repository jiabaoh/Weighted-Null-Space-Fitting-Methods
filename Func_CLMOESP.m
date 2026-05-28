function [A_MOESP, B_MOESP, C_MOESP, D_MOESP, K_MOESP] = Func_CLMOESP(y, u, n_x, f, p)
% Closed-loop MOESP, simple implementation
%
% Inputs:
%   y, u : data matrices, size N x ny and N x nu
%   n_x  : system order
%   f    : future horizon
%   p    : past horizon
%
% Outputs:
%   A_MOESP, B_MOESP, C_MOESP, D_MOESP, K_MOESP
%
% Notes:
%   - This version assumes D = 0.
%   - Step 1 uses Data1 = [Z_p1; U_f1; Y_f1].
%   - Step 3 estimates only x0, B, K.

    [N_total, l] = size(y);
    m = size(u, 2);

    if N_total ~= size(u,1)
        error('y and u must have the same number of samples.');
    end
    if f < 2
        error('f must be at least 2 to estimate A from shift invariance.');
    end
    if N_total - f - p + 1 <= 0
        error('Not enough data for chosen f and p.');
    end

    %% ============================================================
    % STEP 1: Estimate innovation sequence
    % ============================================================
    [Y_f1, Y_p1, U_f1, U_p1] = stackdata(y, u, 1, p);

    % Z_p = [U_p; Y_p]
    Z_p1 = [U_p1; Y_p1];

    % Correct order from the paper: [Z_p1; U_f1; Y_f1]
    Data1 = [Z_p1; U_f1; Y_f1];
    [Q1, R1] = qr(Data1', 0);
    R1 = R1';
    Q1 = Q1';

    n_regressors = size(Z_p1, 1) + size(U_f1, 1);

    % Estimated innovation sequence for f = 1
    E_hat_mat = R1(n_regressors+1:end, n_regressors+1:end) * Q1(n_regressors+1:end, :);
    e_hat_seq = E_hat_mat';   % (N-p) x l

    %% ============================================================
    % STEP 2: Estimate A and C
    % ============================================================
    [Y_f, ~, U_f, ~] = stackdata(y, u, f, p);

    % Innovation Hankel matrix
    E_f = blkhank(e_hat_seq', f, size(Y_f, 2));

    % Z_f = [U_f; E_f]
    Z_f = [U_f; E_f];

    Data2 = [Z_f; Y_f];
    [~, R2] = qr(Data2', 0);
    R2 = R2';

    n_zf = size(Z_f, 1);
    R22 = R2(n_zf+1:end, n_zf+1:end);

    [U_svd, ~, ~] = svd(R22, 'econ');

    % Basis for extended observability matrix
    Un = U_svd(:, 1:n_x);

    % C = first l rows
    C_MOESP = Un(1:l, :);

    % A from shift invariance
    A_top = Un(1:l*(f-1), :);
    A_bot = Un(l+1:l*f, :);
    
%     A_MOESP = A_top \ A_bot;
    A_MOESP = pinv(A_top) * A_bot;
%     A_MOESP = [0.67 0.67 0 0; -0.67 0.67 0 0; 0 0 -0.67 -0.67; 0 0 0.67 -0.67];
%     C_MOESP = [-0.3749 0.0751 -0.5225 0.5830; -0.8977 0.7543 0.1159 0.0982];

    %% ============================================================
    % STEP 3: Estimate B and K, with D fixed to zero
    % ============================================================
    u_ls = u(p+1:end, :);
    y_ls = y(p+1:end, :);
    e_ls = e_hat_seq;
    N_ls = size(y_ls, 1);

    % Theta = [x0; vec(B); vec(K)]
    Phi = zeros(N_ls * l, n_x + n_x*m + n_x*l);
    Y_vec = reshape(y_ls', [], 1);

    for k = 1:N_ls
        row_idx = (k-1)*l + 1 : k*l;

        % x0 term
        Phi_x0 = C_MOESP * (A_MOESP^(k-1));

        % B and K convolution terms
        Phi_B = zeros(l, n_x * m);
        Phi_K = zeros(l, n_x * l);

        for tau = 0:k-2
            CA_pow = C_MOESP * (A_MOESP^(k-tau-2));
            Phi_B = Phi_B + kron(u_ls(tau+1, :), CA_pow);
            Phi_K = Phi_K + kron(e_ls(tau+1, :), CA_pow);
        end

        Phi(row_idx, :) = [Phi_x0, Phi_B, Phi_K];
    end

    % Robust LS solve
    Theta = svd_ls(Phi, Y_vec);

    % Recover parameters
    idx = 0;

    % x0 estimated but not returned
    idx = idx + n_x;

    B_MOESP = reshape(Theta(idx+1 : idx+n_x*m), n_x, m);
    idx = idx + n_x*m;

    K_MOESP = reshape(Theta(idx+1 : idx+n_x*l), n_x, l);

    % D is always zero
    D_MOESP = zeros(l, m);
end


% ============================================================
% Helper: truncated-SVD least-squares solver
% ============================================================
function x = svd_ls(A, b)
    [U, S, V] = svd(A, 'econ');
    s = diag(S);

    if isempty(s)
        x = zeros(size(A,2),1);
        return;
    end

    tol = max(size(A)) * eps(max(s));
    r = sum(s > tol);

    if r == 0
        x = zeros(size(A,2),1);
    else
        x = V(:,1:r) * ((U(:,1:r)' * b) ./ s(1:r));
    end
end


% ============================================================
% Helper: block Hankel matrix
% ============================================================
function H = blkhank(data, rows, cols)
    [l, ~] = size(data);
    H = zeros(l*rows, cols);
    for i = 1:rows
        H((i-1)*l+1:i*l, :) = data(:, i:i+cols-1);
    end
end


% ============================================================
% Helper: stack past/future data
% ============================================================
function [Yf, Yp, Uf, Up] = stackdata(y, u, f, p)
    [N, l] = size(y);
    m = size(u, 2); %#ok<NASGU>

    num_cols = N - f - p + 1;
    if num_cols <= 0
        error('N - f - p + 1 must be positive.');
    end

    Yp = blkhank(y(1:p+num_cols-1, :)', p, num_cols);
    Up = blkhank(u(1:p+num_cols-1, :)', p, num_cols);
    Yf = blkhank(y(p+1:end, :)', f, num_cols);
    Uf = blkhank(u(p+1:end, :)', f, num_cols);
end
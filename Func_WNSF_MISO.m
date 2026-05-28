function [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_MISO(u,y,n_x,n_HOARX)
%FUNC_WNSF_MISO Weighted null-space fitting for MISO state-space models.
%
% [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_MISO(u,y,n_x,n_HOARX)
%
% Estimate a discrete-time multi-input single-output (MISO) innovation-form
% state-space model using weighted null-space fitting (WNSF). A high-order
% ARX model is first estimated by ordinary least squares (OLS). The matrix
% A - K*C is then obtained from a weighted null-space fit, and the input
% matrix B and innovation gain K are estimated using weighted least squares.
%
% The estimated state-space model has the form
%
%   x(t+1) = A_wnsf*x(t) + B_wnsf*u(t) + K_wnsf*e(t)
%   y(t)   = C_wnsf*x(t) + D_wnsf*u(t) + e(t)
%
% Inputs:
% u        - input data, N-by-n_u
% y        - output data, N-by-1 or 1-by-N
% n_x      - system order / number of states
% n_HOARX  - order of the high-order ARX model
%
% Outputs:
% A_wnsf   - estimated state matrix, n_x-by-n_x
% B_wnsf   - estimated input matrix, n_x-by-n_u
% C_wnsf   - estimated output matrix, 1-by-n_x
% D_wnsf   - estimated feedthrough matrix, fixed to zeros(1,n_u)
% K_wnsf   - estimated innovation gain, n_x-by-1
%
% Notes:
% This implementation is intended for MISO systems: multiple inputs and one
% output. The coefficient ordering assumes z = [u y], where all input
% channels appear first and the single output channel appears last.
%
% Compared with the SISO WNSF implementation, the B-estimation step is
% generalized by grouping the high-order ARX coefficients by channel. The
% permutation matrix is constructed for arbitrary n_u, not only for the
% special case n_z = 3.
%
% Reference implementation standardized for public GitHub release.
%
% Jiabao He, 2026
%

%% Input checks and data orientation
if nargin < 4
    error('Func_WNSF_MISO:NotEnoughInputs', ...
        'Four inputs are required: u, y, n_x, and n_HOARX.');
end

if ~isnumeric(u) || ~isnumeric(y)
    error('Func_WNSF_MISO:InvalidDataType', ...
        'Inputs u and y must be numeric arrays.');
end

if ~isvector(y)
    error('Func_WNSF_MISO:SingleOutputOnly', ...
        'This function supports one output only. The output y must be a vector.');
end

y = y(:);

% If a single-input record is supplied as a row vector, convert it to N-by-1.
if isvector(u)
    u = u(:);
end

% If the input matrix appears transposed, orient it as N-by-n_u using the
% length of y as the number of samples.
if size(u,1) ~= length(y) && size(u,2) == length(y)
    u = u.';
end

N = size(u,1);
n_u = size(u,2);
n_y = 1;
n_z = n_u + n_y;

if length(y) ~= N
    error('Func_WNSF_MISO:DataLengthMismatch', ...
        'Input and output data must have the same number of samples.');
end

if n_u < 1
    error('Func_WNSF_MISO:NoInputChannel', ...
        'The input matrix u must contain at least one input channel.');
end

if n_x < 1 || n_HOARX <= n_x
    error('Func_WNSF_MISO:InvalidOrders', ...
        'The orders must satisfy n_x >= 1 and n_HOARX > n_x.');
end

if N <= n_HOARX
    error('Func_WNSF_MISO:InsufficientData', ...
        'The number of samples N must be larger than n_HOARX.');
end

%% Dimensions and regression data
n_data = N - n_HOARX;
Hankel_row = n_x + 1;
Hankel_column = n_HOARX - n_x;

z = [u y].';
Z_hankel = zeros(n_z*n_HOARX,n_data);

% Build the high-order ARX regressor with past inputs and outputs ordered as
% [u_1(t-1); ... ; u_nu(t-1); y(t-1); ... ;
%  u_1(t-n_HOARX); ... ; u_nu(t-n_HOARX); y(t-n_HOARX)].
for i = 1:n_data
    z_col = zeros(n_z*n_HOARX,1);
    for j = 1:n_HOARX
        rows = (j-1)*n_z + (1:n_z);
        z_col(rows) = z(:,i+n_HOARX-j);
    end
    Z_hankel(:,i) = z_col;
end

Y_hankel = y(n_HOARX+1:end).';

%% Step 1: Estimate high-order ARX coefficients by OLS
g_ols = Y_hankel/Z_hankel;

% Regularized covariance estimate of the OLS parameter vector. The small
% diagonal loading improves numerical conditioning for finite data records.
lambda = 1e-6;
R_zz = (Z_hankel*Z_hankel.' + lambda*eye(size(Z_hankel,1)))/n_data;
Cov_g_ols = R_zz\eye(size(R_zz));

%% Step 2: Weighted null-space fitting for A - K*C
% Construct the block Hankel matrix of high-order ARX coefficients. Since
% the system has a single output, the null-space relation gives a scalar
% polynomial whose coefficients define A - K*C in companion form.
G_hankel = zeros(n_y*Hankel_row,n_z*Hankel_column);
for i = 1:Hankel_row
    for j = 1:Hankel_column
        row_idx = (i-1)*n_y + (1:n_y);
        col_idx = (j-1)*n_z + (1:n_z);
        par_idx = (i+j-2)*n_z + (1:n_z);
        G_hankel(row_idx,col_idx) = g_ols(:,par_idx);
    end
end

G_hankel_plus = G_hankel(1:Hankel_row-1,:);
G_hankel_minus = G_hankel(Hankel_row,:);
alpha_ols = -G_hankel_minus/G_hankel_plus;

% Build the Toeplitz weighting matrix associated with the null-space
% polynomial and refine alpha by WLS.
column_vector = [alpha_ols(:); 1; zeros(n_HOARX-n_x-1,1)];
row_vector = [alpha_ols(1), zeros(1,Hankel_column-1)];
Toep_alpha = toeplitz(column_vector,row_vector);
K_alpha = kron(Toep_alpha,eye(n_z));

W_alpha = (K_alpha.'*Cov_g_ols*K_alpha)\eye(size(K_alpha,2));
alpha_wls = -(G_hankel_minus*W_alpha*G_hankel_plus.')/ ...
    (G_hankel_plus*W_alpha*G_hankel_plus.');

% Convert the estimated polynomial coefficients to companion canonical form.
I_n_x = eye(n_x);
A_K_CF = [-flip(alpha_wls).', I_n_x(:,1:n_x-1)];
C_CF = I_n_x(1,:);

%% Step 3: Initial OLS estimation of B and K
% Build the observability matrix associated with the estimated A - K*C and C.
Obs_ols = zeros(n_HOARX,n_x);
for i = 0:n_HOARX-1
    Obs_ols(i+1,:) = C_CF*(A_K_CF^i);
end

% Reshape the high-order ARX coefficients from lag-interleaved ordering to
% channel-grouped ordering:
% [channel 1 lags; ... ; channel n_u lags; output lags].
g_by_lag = reshape(g_ols,n_z,[]).';
g_ols_B_K = g_by_lag(:);

data_B = kron(eye(n_z),Obs_ols);
B_K_CF_ols = data_B\g_ols_B_K;
B_K_ols = reshape(B_K_CF_ols,[],n_z);

B_ols = B_K_ols(:,1:n_u);
K_ols = B_K_ols(:,n_u+1);
B_K_ols = [B_ols, K_ols];

%% Step 4: Weighted least-squares estimation of B and K
% Construct the correction term that accounts for uncertainty in the
% estimated A-polynomial coefficients.
barP = fliplr(eye(n_x));
barI = [eye(n_x), zeros(n_x,(n_x-1)*n_x)];
S_n = zeros(n_HOARX*n_x,n_x);

for k = 1:n_HOARX-1
    S_k = zeros(n_x*n_x,n_x);
    for i = 0:k-1
        S_k = S_k + kron(A_K_CF^(k-i-1),(C_CF*(A_K_CF^i)).');
    end
    rows = k*n_x + (1:n_x);
    S_n(rows,:) = -barP*barI*S_k;
end

I_n = eye(n_HOARX);
Xi = zeros(n_z*n_HOARX,n_x);
for i = 1:n_x
    e_i = I_n_x(:,i);
    I_tmp = kron(I_n,e_i.');
    Xi(:,i) = kron(B_K_ols.'*S_n.',I_n)*I_tmp(:);
end

M_n = G_hankel_plus*W_alpha*G_hankel_plus.';

column_vector = [alpha_wls(:); 1; zeros(n_HOARX-n_x-1,1)];
row_vector = [alpha_wls(1), zeros(1,Hankel_column-1)];
Toep_alpha = toeplitz(column_vector,row_vector);
K_alpha_wls = kron(Toep_alpha,eye(n_z));

Weighting_A = M_n\(G_hankel_plus*W_alpha*K_alpha_wls.');

% Permutation matrix that maps the lag-interleaved ARX coefficients
% [z_1(t-1); ... ; z_nz(t-1); z_1(t-2); ...] to channel-grouped coefficients
% [z_1 all lags; z_2 all lags; ... ; z_nz all lags].
pi_idx = zeros(1,n_z*n_HOARX);
for channel = 1:n_z
    idx = (channel-1)*n_HOARX + (1:n_HOARX);
    pi_idx(idx) = channel:n_z:n_z*n_HOARX;
end

P = eye(n_z*n_HOARX);
P = P(pi_idx,:);

Weighting = P + Xi*Weighting_A;
W_B = (Weighting*Cov_g_ols*Weighting.')\eye(size(Weighting,1));

B_K_CF_wls = (data_B.'*W_B*data_B)\(data_B.'*W_B*g_ols_B_K);
B_K_wls = reshape(B_K_CF_wls,[],n_z);

B_wls = B_K_wls(:,1:n_u);
K_wls = B_K_wls(:,n_u+1);

%% Output state-space matrices
C_wnsf = C_CF;
K_wnsf = K_wls;
B_wnsf = B_wls;
A_wnsf = A_K_CF + K_wnsf*C_wnsf;
D_wnsf = zeros(n_y,n_u);

end

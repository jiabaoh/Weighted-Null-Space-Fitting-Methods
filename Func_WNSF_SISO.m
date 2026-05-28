function [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_SISO(u,y,n_x,n_HOARX)
%FUNC_WNSF_SISO2 Weighted null-space fitting for SISO state-space models.
%
%
% Estimate a discrete-time SISO innovation-form state-space model using
% weighted null-space fitting (WNSF). A high-order ARX model is first
% estimated by ordinary least squares. The state matrix is then recovered
% from a weighted null-space fit, followed by weighted estimation of the
% input and Kalman-gain vectors.
%
% Inputs:
% u        - input data, N-by-1 or 1-by-N
% y        - output data, N-by-1 or 1-by-N
% n_x      - system order / number of states
% n_HOARX  - order of the high-order ARX model
%
% Outputs:
% A_wnsf   - estimated state matrix
% B_wnsf   - estimated input matrix
% C_wnsf   - estimated output matrix
% D_wnsf   - estimated feedthrough matrix, fixed to zero in this version
% K_wnsf   - estimated innovation gain
%
% Notes:
% This implementation is SISO-specific. The coefficient ordering assumes
% z = [u y] and separates the high-order ARX coefficients using odd/even
% indices. For MIMO data, this routine must be generalized before use.
%
% Reference implementation standardized for public GitHub release.
%
% Jiabao He, 2026
%

%% Input checks and data orientation
if nargin < 4
    error('Func_WNSF_SISO2:NotEnoughInputs', ...
        'Four inputs are required: u, y, n_x, and n_HOARX.');
end

if ~isvector(u) || ~isvector(y)
    error('Func_WNSF_SISO2:SISOOnly', ...
        'This function supports SISO data only. Inputs u and y must be vectors.');
end

u = u(:);
y = y(:);

N = size(u,1);
if size(y,1) ~= N
    error('Func_WNSF_SISO2:DataLengthMismatch', ...
        'Input and output data must have the same number of samples.');
end

if n_x < 1 || n_HOARX <= n_x
    error('Func_WNSF_SISO2:InvalidOrders', ...
        'The orders must satisfy n_x >= 1 and n_HOARX > n_x.');
end

if N <= n_HOARX
    error('Func_WNSF_SISO2:InsufficientData', ...
        'The number of samples N must be larger than n_HOARX.');
end

%% Dimensions and regression data
n_y = 1;
n_u = 1;
n_z = n_y + n_u;

n_data = N - n_HOARX;
Hankel_row = n_x + 1;
Hankel_column = n_HOARX - n_x;

z = [u y].';
Z_hankel = zeros(n_z*n_HOARX,n_data);

% Build the high-order ARX regressor with past inputs and outputs ordered as
% [u(t-1); y(t-1); ... ; u(t-n_HOARX); y(t-n_HOARX)].
for i = 1:n_data
    z_col = zeros(n_z*n_HOARX,1);
    for j = 1:n_HOARX
        rows = (j-1)*n_z + (1:n_z);
        z_col(rows) = z(:,i+n_HOARX-j);
    end
    Z_hankel(:,i) = z_col;
end

Y_hankel = y(n_HOARX+1:end).';

%% Weighted null-space fitting for A
% Step 1: Estimate the high-order ARX Markov parameters by OLS.
g_ols = Y_hankel/Z_hankel;

% Regularized covariance estimate of the OLS parameter vector. The small
% diagonal loading improves numerical conditioning for finite data records.
lambda = 1e-6;
R_zz = (Z_hankel*Z_hankel.' + lambda*eye(size(Z_hankel,1)))/n_data;
Cov_g_ols = R_zz\eye(size(R_zz));

% Step 2: Construct the block Hankel matrix of Markov parameters.
G_hankel = zeros(n_y*Hankel_row,n_z*Hankel_column);
for i = 1:Hankel_row
    for j = 1:Hankel_column
        row_idx = (i-1)*n_y + (1:n_y);
        col_idx = (j-1)*n_z + (1:n_z);
        par_idx = (i+j-2)*n_z + (1:n_z);
        G_hankel(row_idx,col_idx) = g_ols(:,par_idx);
    end
end

% Initial OLS estimate of the polynomial coefficients defining the
% null-space relation.
G_hankel_plus = G_hankel(1:Hankel_row-1,:);
G_hankel_minus = G_hankel(Hankel_row,:);
alpha_ols = -G_hankel_minus/G_hankel_plus;

% Step 3: Refine the null-space estimate by WLS.
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

%% Weighted null-space fitting for B and K
% Step 1: Initial OLS estimates of B and K in companion form.
Obs_ols = zeros(n_HOARX,n_x);
for i = 0:n_HOARX-1
    Obs_ols(i+1,:) = C_CF*(A_K_CF^i);
end

g_ols_B_K = [g_ols(1:2:end).'; g_ols(2:2:end).'];
data_B = kron(eye(2),Obs_ols);

B_K_CF_ols = data_B\g_ols_B_K;
B_ols = B_K_CF_ols(1:n_x);
K_ols = B_K_CF_ols(n_x+1:end);
B_K_ols = [B_ols, K_ols];

% Step 2: Construct the WLS weighting matrix. The correction term Xi accounts
% for the uncertainty in the estimated A-polynomial coefficients.
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

% Permutation matrix that maps interleaved ARX coefficients
% [u1 y1 u2 y2 ...] to grouped coefficients [u1 u2 ... y1 y2 ...].
pi_idx = [1:2:2*n_HOARX, 2:2:2*n_HOARX];
P = eye(2*n_HOARX);
P = P(pi_idx,:);

Weighting = P + Xi*Weighting_A;
W_B = (Weighting*Cov_g_ols*Weighting.')\eye(size(Weighting,1));

B_K_CF_wls = (data_B.'*W_B*data_B)\(data_B.'*W_B*g_ols_B_K);
B_wls = B_K_CF_wls(1:n_x);
K_wls = B_K_CF_wls(n_x+1:end);

%% Output state-space matrices
C_wnsf = C_CF;
K_wnsf = K_wls;
B_wnsf = B_wls;
A_wnsf = A_K_CF + K_wnsf*C_wnsf;
D_wnsf = zeros(n_y,n_u);

end

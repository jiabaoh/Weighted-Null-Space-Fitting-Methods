function [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_MIMO242(u,y,n_x,n_HOARX,nu)
%FUNC_WNSF_MIMO242 Weighted null-space fitting for a 2-input, 4-state, 2-output model.
%
% [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_MIMO242(u,y,n_x,n_HOARX,nu)
%
% Estimate a discrete-time MIMO innovation-form state-space model using
% weighted null-space fitting (WNSF). This implementation is specialized
% for the 2-4-2 case: two inputs, four states, and two outputs.
%
% A high-order ARX model is first estimated by ordinary least squares (OLS).
% The matrix A - K*C is then estimated using a structured observability
% parameterization and weighted least squares (WLS). Finally, the input
% matrix B and innovation gain K are estimated and the model is converted to
% the innovation-form state-space realization.
%
% The estimated state-space model has the form
%
%   x(t+1) = A_wnsf*x(t) + B_wnsf*u(t) + K_wnsf*e(t)
%   y(t)   = C_wnsf*x(t) + D_wnsf*u(t) + e(t)
%
% Inputs:
% u        - input data, N-by-2
% y        - output data, N-by-2
% n_x      - system order / number of states; must be 4 in this routine
% n_HOARX  - order of the high-order ARX model
% nu       - observability multi-index; optional, default [1 3]
%
% Outputs:
% A_wnsf   - estimated state matrix, 4-by-4
% B_wnsf   - estimated input matrix, 4-by-2
% C_wnsf   - estimated output matrix, 2-by-4
% D_wnsf   - estimated feedthrough matrix, fixed to zeros(2,2)
% K_wnsf   - estimated innovation gain, 4-by-2
%
% Notes:
% This routine is not a fully general MIMO implementation. It is written for
% the 2-input, 4-state, 2-output case. The current WLS construction uses the
% observability multi-index nu = [1 3], corresponding to
%
%   C = [1 0 0 0;
%        0 1 0 0]
%
% and the structured form
%
%   A - K*C = [a_1;
%              0 0 1 0;
%              0 0 0 1;
%              a_2].
%
% Other multi-index parameterizations require corresponding changes in the
% structured WLS weighting matrices.
%
% Reference implementation standardized for public GitHub release.
%
% Jiabao He, 2026
%

%% Input checks and data orientation
if nargin < 4
    error('Func_WNSF_MIMO242:NotEnoughInputs', ...
        'At least four inputs are required: u, y, n_x, and n_HOARX.');
end

if nargin < 5 || isempty(nu)
    nu = [1 3];
end

if ~isnumeric(u) || ~isnumeric(y)
    error('Func_WNSF_MIMO242:InvalidDataType', ...
        'Inputs u and y must be numeric arrays.');
end

% Orient the data as N-by-channel when a transposed 2-by-N record is supplied.
if size(u,2) ~= 2 && size(u,1) == 2
    u = u.';
end
if size(y,2) ~= 2 && size(y,1) == 2
    y = y.';
end

[N,n_u] = size(u);
[N_y,n_y] = size(y);

if N_y ~= N
    error('Func_WNSF_MIMO242:DataLengthMismatch', ...
        'Input and output data must have the same number of samples.');
end

if n_u ~= 2 || n_y ~= 2
    error('Func_WNSF_MIMO242:InvalidDimensions', ...
        'This function is specialized for two inputs and two outputs.');
end

if n_x ~= 4
    error('Func_WNSF_MIMO242:InvalidStateOrder', ...
        'This specialized implementation requires n_x = 4.');
end

if ~isequal(nu(:).',[1 3])
    error('Func_WNSF_MIMO242:UnsupportedMultiIndex', ...
        'The current WLS implementation supports only the multi-index nu = [1 3].');
end

if n_HOARX <= n_x
    error('Func_WNSF_MIMO242:InvalidHOARXOrder', ...
        'The high-order ARX order must satisfy n_HOARX > n_x.');
end

if N <= n_HOARX
    error('Func_WNSF_MIMO242:InsufficientData', ...
        'The number of samples N must be larger than n_HOARX.');
end

%% Dimensions and high-order ARX regression data
n_z = n_u + n_y;
n = n_HOARX;
n_data = N - n;

% For the 2-4-2 parameterization, f = n_x - 1 and p = n - f are used to
% form the block Hankel matrix of high-order ARX coefficients.
f = n_x - 1;
p = n - f;

z = [u y].';

sum_phiphi = zeros(n*n_z*n_y,n*n_z*n_y);
sum_phiy = zeros(n*n_z*n_y,1);

% Estimate the high-order ARX coefficient matrix G by OLS. The regressor is
% ordered by lag as
% [u_1(t-1); u_2(t-1); y_1(t-1); y_2(t-1); ... ;
%  u_1(t-n); u_2(t-n); y_1(t-n); y_2(t-n)].
for i = 1:n_data
    z_lag = zeros(n*n_z,1);
    for j = 1:n
        rows = (j-1)*n_z + (1:n_z);
        z_lag(rows) = z(:,i+n-j);
    end

    phi_t = kron(z_lag,eye(n_y));
    y_t = y(i+n,:).';

    sum_phiphi = sum_phiphi + phi_t*phi_t.';
    sum_phiy = sum_phiy + phi_t*y_t;
end

% A small diagonal loading improves numerical conditioning for finite data
% records without changing the intended OLS/WLS structure.
lambda = 1e-8;
sum_phiphi_reg = sum_phiphi + lambda*eye(size(sum_phiphi));
vec_g = sum_phiphi_reg\sum_phiy;

g_ols = reshape(vec_g,n_y,n*n_z);

%% Step 1: Build the block Hankel matrix of high-order ARX coefficients
G_hankel = zeros(n_y*(f+1),n_z*p);
for i = 1:f+1
    for j = 1:p
        row_idx = (i-1)*n_y + (1:n_y);
        col_idx = (j-1)*n_z + (1:n_z);
        par_idx = (i+j-2)*n_z + (1:n_z);
        G_hankel(row_idx,col_idx) = g_ols(:,par_idx);
    end
end

%% Step 2: OLS estimate of A - K*C using the selected multi-index
% For nu = [1 3], the selected rows correspond to the observability basis
% [C_1; C_2; C_2*A; C_2*A^2].
idx_plus = [1, n_y, 2*n_y, 3*n_y];
idx_minus = [1+n_y, 2*n_y, 3*n_y, 4*n_y];

G_hankel_plus = G_hankel(idx_plus,:);
G_hankel_minus = G_hankel(idx_minus,:);

a_1_ols = G_hankel_minus(1,:)/G_hankel_plus;
a_2_ols = G_hankel_minus(4,:)/G_hankel_plus;

A_K_ols = [a_1_ols; ...
           0 0 1 0; ...
           0 0 0 1; ...
           a_2_ols];
C_ols = [1 0 0 0; ...
         0 1 0 0];

%% Step 3: WLS refinement of A - K*C
% Covariance of vec(G) in column-stacked order.
R_gamma = sum_phiphi_reg/n_data;
Cov_g_ols = R_gamma\eye(size(R_gamma));

% Convert the covariance to the row-stacked ordering used by the structured
% Toeplitz weighting matrices.
K_comm = local_commutation_matrix(n_y,n*n_z);
Cov_g_ols_row = K_comm.'*Cov_g_ols*K_comm;

% Toeplitz matrices for the two free rows a_1 and a_2 of A - K*C.
a_1_ols_aug = [a_1_ols(1), -1, 0, 0, a_1_ols(2:end), 0];
a_2_ols_aug = [a_2_ols(1),  0, 0, 0, a_2_ols(2:end), -1];

toep_a_1_sub_A = local_subspace_toeplitz(a_1_ols_aug,n_x,n_y,n,p,n_z);
toep_a_2_sub_A = local_subspace_toeplitz(a_2_ols_aug,n_x,n_y,n,p,n_z);

weighting_a_1 = (toep_a_1_sub_A.'*Cov_g_ols_row*toep_a_1_sub_A)\ ...
    eye(size(toep_a_1_sub_A,2));
weighting_a_2 = (toep_a_2_sub_A.'*Cov_g_ols_row*toep_a_2_sub_A)\ ...
    eye(size(toep_a_2_sub_A,2));

M_n1 = (G_hankel_plus*weighting_a_1*G_hankel_plus.')\eye(n_x);
M_n2 = (G_hankel_plus*weighting_a_2*G_hankel_plus.')\eye(n_x);

a_1_wls = G_hankel_minus(1,:)*weighting_a_1*G_hankel_plus.'*M_n1;
a_2_wls = G_hankel_minus(4,:)*weighting_a_2*G_hankel_plus.'*M_n2;

A_K_wls = [a_1_wls; ...
           0 0 1 0; ...
           0 0 0 1; ...
           a_2_wls];
C_wls = C_ols;

%% Step 4: Initial OLS estimate of B and K
% Build the extended observability matrix for the estimated A - K*C.
Obs_wls = zeros(n_y*n,n_x);
for i = 0:n-1
    rows = i*n_y + (1:n_y);
    Obs_wls(rows,:) = C_wls*(A_K_wls^i);
end

% Reorder the estimated high-order coefficients into the form
% Markov = Obs_wls * [B K], where [B K] is n_x-by-(n_u+n_y).
g_ols_trans = reshape(g_ols.',n_z,[]).';
g_ols_trans_1 = g_ols_trans(1:n,:);
g_ols_trans_2 = g_ols_trans(n+1:2*n,:);
g_ols_trans_p = reshape(permute(cat(3,g_ols_trans_1,g_ols_trans_2),[3 1 2]),[],n_z);

Vec_g_ols_trans_p = g_ols_trans_p(:);
Phi_n = kron(eye(n_z),Obs_wls);

B_K_ols_vec = Phi_n\Vec_g_ols_trans_p;
B_K_ols = reshape(B_K_ols_vec,n_x,n_z);

%% Step 5: WLS estimate of B and K
% Construct the correction term caused by uncertainty in the WLS estimate of
% A - K*C. The matrices below are specialized for the 2-4-2, nu = [1 3]
% observability parameterization.
barP = eye(n_y*n_z);
barI = [eye(n_x), zeros(n_x,(n_y-1)*n_x); ...
        zeros(n_x,(n_y-1)*n_x), eye(n_x)];

S_n = zeros(n_y*n_x,n_y*n_x*n);
for k = 1:n-1
    S_k = zeros(n_x*n_x,n_y*n_x);
    for i = 0:k-1
        S_k = S_k + kron((C_wls*(A_K_wls^i)).',A_K_wls^(k-i-1));
    end

    cols = k*(n_y*n_x) + (1:n_y*n_x);
    S_n(:,cols) = barP*barI*S_k;
end

Xi = S_n*kron(eye(n_y*n),B_K_ols);

% Update the structured Toeplitz matrices using the WLS estimates a_1 and a_2.
a_1_wls_aug = [a_1_wls(1), -1, 0, 0, a_1_wls(2:end), 0];
a_2_wls_aug = [a_2_wls(1),  0, 0, 0, a_2_wls(2:end), -1];

toep_a_1_sub_A = local_subspace_toeplitz(a_1_wls_aug,n_x,n_y,n,p,n_z);
toep_a_2_sub_A = local_subspace_toeplitz(a_2_wls_aug,n_x,n_y,n,p,n_z);

Weighting_A1 = toep_a_1_sub_A*weighting_a_1*G_hankel_plus.'*M_n1;
Weighting_A2 = toep_a_2_sub_A*weighting_a_2*G_hankel_plus.'*M_n2;
Weighting_A = [Weighting_A1, Weighting_A2];
Weighting_tmp = Weighting_A*Xi;

% Permutation matrix that maps the row-stacked coefficient order used in
% Cov_g_ols_row to the output-interleaved order used in the B/K regression.
row1 = 1:n_z*n;
row2 = n_z*n+1:n_z*n_y*n;
pi_idx = reshape([reshape(row1,n_z,[]); reshape(row2,n_z,[])],[],1);

P = eye(n_z*n_y*n);
P = P(:,pi_idx);

Weighting = P - Weighting_tmp;
W_B = (Weighting.'*Cov_g_ols_row*Weighting)\eye(size(Weighting,2));

B_K_wls_vec = (Phi_n.'*W_B*Phi_n)\(Phi_n.'*W_B*Vec_g_ols_trans_p);
B_K_wls = reshape(B_K_wls_vec,n_x,n_z);

%% Output state-space matrices
B_wnsf = B_K_wls(:,1:n_u);
K_wnsf = B_K_wls(:,n_u+1:n_u+n_y);
C_wnsf = C_wls;
A_wnsf = A_K_wls + K_wnsf*C_wnsf;
D_wnsf = zeros(n_y,n_u);

end

function K = local_commutation_matrix(m,n)
%LOCAL_COMMUTATION_MATRIX Return K_{m,n} such that vec(X.') = K*vec(X).
K = zeros(m*n,m*n);
for i = 1:m
    for j = 1:n
        K((j-1)*m+i,(i-1)*n+j) = 1;
    end
end
end

function T = local_subspace_toeplitz(a_aug,n_x,n_y,n,p,n_z)
%LOCAL_SUBSPACE_TOEPLITZ Build the structured Toeplitz matrix used for WLS.
%
% a_aug contains one n_x-length coefficient block per output. The resulting
% matrix maps the row-stacked high-order ARX coefficients to the selected
% structured null-space relation.
T = [];
for output_idx = 1:n_y
    block_idx = (output_idx-1)*n_x + (1:n_x);
    coeff_col = [a_aug(block_idx).'; zeros(n-n_x,1)];

    coeff_col_aug = zeros(n_x*numel(coeff_col),1);
    coeff_col_aug(1:n_x:end) = coeff_col;

    coeff_row_aug = [coeff_col(1), zeros(1,p*n_z-1)];
    T = [T; toeplitz(coeff_col_aug,coeff_row_aug)]; %#ok<AGROW>
end
end

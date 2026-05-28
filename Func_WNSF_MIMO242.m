function [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_MIMO242(u,y,n_x,n_HOARX,nu)
%FUNC_WNSF_MIMO242 WNSF for a specialized 2-input, 4-state, 2-output MIMO model.
%
% [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_MIMO242(u,y,n_x,n_HOARX,nu)
%
% Estimate a discrete-time MIMO innovation-form state-space model using
% weighted null-space fitting (WNSF). This routine is specialized for the
% 2-4-2 case, where the system has two inputs, four states, and two outputs.
%
% The method first estimates a high-order MIMO ARX model by ordinary least
% squares (OLS). It then estimates the structured matrix A - K*C using a
% selected multi-index parameterization and WLS. Finally, B and K are
% estimated and the innovation-form state-space model is returned.
%
% The estimated model has the form
%
%   x(t+1) = A_wnsf*x(t) + B_wnsf*u(t) + K_wnsf*e(t)
%   y(t)   = C_wnsf*x(t) + D_wnsf*u(t) + e(t)
%
% Inputs:
% u        - input data, N-by-2
% y        - output data, N-by-2
% n_x      - system order / number of states; must be 4
% n_HOARX  - order of the high-order ARX model
% nu       - multi-index parameterization flag; currently only the first
%            parameterization used in this code is supported
%
% Outputs:
% A_wnsf   - estimated state matrix, 4-by-4
% B_wnsf   - estimated input matrix, 4-by-2
% C_wnsf   - estimated output matrix, 2-by-4
% D_wnsf   - estimated feedthrough matrix, fixed to zeros(2,2)
% K_wnsf   - estimated innovation gain, 4-by-2
%
% Notes:
% This is a performance-preserving standardization of the working MIMO242
% implementation. The numerical recipe is intentionally kept the same as the
% original code, including the explicit inverse-based WLS formulas, the
% selected observability parameterization, the hard-coded row-vectorization
% matrices used in the B/K weighting step, and the final row-vectorized WLS
% solution for B and K.
%
% This routine is not a fully general MIMO implementation. It is intended for
% the 2-input, 4-state, 2-output case only. General MIMO cases require a
% generalized multi-index parameterization and generalized weighting
% matrices.
%
% Reference implementation standardized for public GitHub release.
%
% Jiabao He, 2026
%

%% Input checks
if nargin < 4
    error('Func_WNSF_MIMO242:NotEnoughInputs', ...
        'At least four inputs are required: u, y, n_x, and n_HOARX.');
end

if nargin < 5
    nu = [];
end

if ~isnumeric(u) || ~isnumeric(y)
    error('Func_WNSF_MIMO242:InvalidDataType', ...
        'Inputs u and y must be numeric arrays.');
end

[N,n_u] = size(u);
n_y = size(y,2);

if size(y,1) ~= N
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

if n_HOARX <= n_x
    error('Func_WNSF_MIMO242:InvalidHOARXOrder', ...
        'The high-order ARX order must satisfy n_HOARX > n_x.');
end

if N <= n_HOARX
    error('Func_WNSF_MIMO242:InsufficientData', ...
        'The number of samples N must be larger than n_HOARX.');
end

% The current code keeps the first manually selected multi-index
% parameterization from the original implementation. The input nu is kept in
% the signature for compatibility with existing scripts.
if ~isempty(nu)
    % No action is taken here. This preserves the behavior of the original
    % implementation, where nu was an input but the first parameterization
    % was selected manually in the code.
end

%% Dimensions and high-order ARX coefficient estimation
z = [u y]';
n_z = n_y + n_u;
n = n_HOARX;

% For the 2-4-2 realization used here, f = n_x - 1 and p = n - f.
f = n_x - 1;
p = n - f;

sum_phiphi = zeros(n*n_z*n_y,n*n_z*n_y);
sum_phiy = zeros(n*n_z*n_y,1);

% Estimate the high-order MIMO ARX coefficients by OLS.
for i = 1:(N-n)
    Z_hat = zeros(n*n_z,1);

    for j = 1:n
        rows = (j-1)*n_z + (1:n_z);
        Z_hat(rows,:) = z(:,i+n-j);
    end

    phi_t = (kron(Z_hat',eye(n_y)))';
    y_hat = y';
    Y_hat = y_hat(:,i+n);

    sum_phiphi = sum_phiphi + phi_t*phi_t';
    sum_phiy = sum_phiy + phi_t*Y_hat;
end

vec_g = sum_phiphi\sum_phiy;
g_ols = reshape(vec_g,n_y,n*n_z);

%% Build the block Hankel matrix of high-order ARX coefficients
G_hankel = zeros(n_y*(f+1),n_z*p);

for i = 1:(f+1)
    for j = 1:p
        row_idx = (n_y*(i-1)+1):n_y*i;
        col_idx = (n_u+n_y)*(j-1)+1:(n_u+n_y)*j;
        par_idx = (n_u+n_y)*(i+j-2)+1:(n_u+n_y)*(i+j-1);

        G_hankel(row_idx,col_idx) = g_ols(:,par_idx);
    end
end

%% OLS estimate of A - K*C using the selected parameterization
% This is the first manually selected multi-index parameterization in the
% original code. It gave the desired performance for the 2-4-2 case.
G_hankel_plus = [G_hankel(1,:); ...
                 G_hankel(n_y,:); ...
                 G_hankel(2*n_y,:); ...
                 G_hankel(3*n_y,:)];

G_hankel_minus = [G_hankel(1+n_y,:); ...
                  G_hankel(2*n_y,:); ...
                  G_hankel(3*n_y,:); ...
                  G_hankel(4*n_y,:)];

a_1_ols = G_hankel_minus(1,:)/G_hankel_plus;
a_2_ols = G_hankel_minus(4,:)/G_hankel_plus;

A_K_ols = [a_1_ols; ...
           0 0 1 0; ...
           0 0 0 1; ...
           a_2_ols];
C_ols = [1 0 0 0; ...
         0 1 0 0];

%% WLS refinement of A - K*C
% Estimate the covariance of vec(g_ols).
R_Gama = sum_phiphi/(N-n);
Cov_g_ols = inv(R_Gama);

% Convert the covariance matrix to the row-by-row vectorization used below.
[n_y,nzn] = size(g_ols);
n_g = numel(g_ols);
K = zeros(n_g,n_g);

for i = 1:n_y
    for j = 1:nzn
        K((j-1)*n_y+i,(i-1)*nzn+j) = 1;
    end
end

Cov_g_ols_row = K'*Cov_g_ols*K;

% Build structured Toeplitz matrices for the two free rows of A - K*C.
a_1_ols_aug = [a_1_ols(1,1) -1 0 0 a_1_ols(1,2:end) 0];

column_vector_a_1_1 = [a_1_ols_aug(1,1:n_x)'; zeros(n-n_x,1)];
column_vector_aug_a_1_1 = zeros(n_x*numel(column_vector_a_1_1),1);
column_vector_aug_a_1_1(1:n_x:end) = column_vector_a_1_1;
row_vector_aug_a_1_1 = [column_vector_a_1_1(1,1) zeros(1,p*n_z-1)];
toep_a_1_1_sub_A = toeplitz(column_vector_aug_a_1_1,row_vector_aug_a_1_1);

column_vector_a_1_2 = [a_1_ols_aug(1,1+n_x:end)'; zeros(n-n_x,1)];
column_vector_aug_a_1_2 = zeros(n_x*numel(column_vector_a_1_2),1);
column_vector_aug_a_1_2(1:n_x:end) = column_vector_a_1_2;
row_vector_aug_a_1_2 = [column_vector_a_1_2(1,1) zeros(1,p*n_z-1)];
toep_a_1_2_sub_A = toeplitz(column_vector_aug_a_1_2,row_vector_aug_a_1_2);

toep_a_1_sub_A = [toep_a_1_1_sub_A; toep_a_1_2_sub_A];

% Second free row of A - K*C.
a_2_ols_aug = [a_2_ols(1,1) 0 0 0 a_2_ols(1,2:end) -1];

column_vector_a_2_1 = [a_2_ols_aug(1,1:n_x)'; zeros(n-n_x,1)];
column_vector_aug_a_2_1 = zeros(n_x*numel(column_vector_a_2_1),1);
column_vector_aug_a_2_1(1:n_x:end) = column_vector_a_2_1;
row_vector_aug_a_2_1 = [column_vector_a_2_1(1,1) zeros(1,p*n_z-1)];
toep_a_2_1_sub_A = toeplitz(column_vector_aug_a_2_1,row_vector_aug_a_2_1);

column_vector_a_2_2 = [a_2_ols_aug(1,1+n_x:end)'; zeros(n-n_x,1)];
column_vector_aug_a_2_2 = zeros(n_x*numel(column_vector_a_2_2),1);
column_vector_aug_a_2_2(1:n_x:end) = column_vector_a_2_2;
row_vector_aug_a_2_2 = [column_vector_a_2_2(1,1) zeros(1,p*n_z-1)];
toep_a_2_2_sub_A = toeplitz(column_vector_aug_a_2_2,row_vector_aug_a_2_2);

toep_a_2_sub_A = [toep_a_2_1_sub_A; toep_a_2_2_sub_A];

% WLS estimates of the two free rows. The explicit inverse formulas are kept
% to preserve the original numerical behavior.
weighting_a_1 = inv(toep_a_1_sub_A'*Cov_g_ols_row*toep_a_1_sub_A);
weighting_a_2 = inv(toep_a_2_sub_A'*Cov_g_ols_row*toep_a_2_sub_A);

M_n1 = inv(G_hankel_plus*weighting_a_1*G_hankel_plus');
M_n2 = inv(G_hankel_plus*weighting_a_2*G_hankel_plus');

a_1_wls = G_hankel_minus(1,:)*weighting_a_1*G_hankel_plus'*M_n1;
a_2_wls = G_hankel_minus(4,:)*weighting_a_2*G_hankel_plus'*M_n2;

A_K_wls = [a_1_wls; ...
           0 0 1 0; ...
           0 0 0 1; ...
           a_2_wls];
C_wls = C_ols;

%% OLS estimate of B and K
% Construct the extended observability matrix.
Obs_wls = C_wls;
for i = 1:(n_HOARX-1)
    Obs_wls = [Obs_wls; C_wls*(A_K_wls^i)]; %#ok<AGROW>
end

% Reorder the Markov parameters according to the original implementation.
g_ols_trans = reshape(g_ols',n_z,[])';
g_ols_trans_1 = g_ols_trans(1:n,:);
g_ols_trans_2 = g_ols_trans(1+n:end,:);
g_ols_trans_p = reshape(permute(cat(3,g_ols_trans_1,g_ols_trans_2),[3 1 2]),[],n_z);

Vec_g_ols_trans_p = reshape(g_ols_trans_p,[],1);
Phi_n = kron(eye(n_z),Obs_wls);
B_K_ols_Vec = Phi_n\Vec_g_ols_trans_p;
B_K_ols = reshape(B_K_ols_Vec,n_x,[]);

% Row-vectorized version used in the final WLS solution.
Vec_g_ols_trans_p_row = reshape(g_ols_trans_p',1,[]);
Phi_n_row = kron(Obs_wls',eye(n_z));
B_K_ols_Vec_row = Vec_g_ols_trans_p_row/Phi_n_row;
B_K_ols_row = reshape(B_K_ols_Vec_row,[n_x,n_z])'; %#ok<NASGU>

%% WLS estimate of B and K
% This section intentionally keeps the original hard-coded row-vectorization
% matrices for the 2-4-2 case. A previous generalized version caused a
% dimension mismatch here; the original concatenation form below preserves
% the working dimensions and numerical result.
barP = eye(n_y*n_z);
barI = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; ...
        0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0; ...
        0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0; ...
        0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0; ...
        0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0; ...
        0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0; ...
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; ...
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];

S_n = zeros(n_y*n_x,n_y*n_x);
for k = 1:n_HOARX-1
    S_k = zeros(n_x*n_x,n_y*n_x);

    for i = 0:k-1
        S_k = S_k + kron((C_wls*(A_K_wls^i))',(A_K_wls^(k-i-1)));
    end

    S_n = [S_n, barP*barI*S_k]; %#ok<AGROW>
end

Xi = S_n*(kron(eye(n_y*n),B_K_ols));

% Rebuild the Toeplitz matrices using the WLS estimates of a_1 and a_2.
a_1_wls_aug = [a_1_wls(1,1) -1 0 0 a_1_wls(1,2:end) 0];

column_vector_a_1_1 = [a_1_wls_aug(1,1:n_x)'; zeros(n-n_x,1)];
column_vector_aug_a_1_1 = zeros(n_x*numel(column_vector_a_1_1),1);
column_vector_aug_a_1_1(1:n_x:end) = column_vector_a_1_1;
row_vector_aug_a_1_1 = [column_vector_a_1_1(1,1) zeros(1,p*n_z-1)];
toep_a_1_1_sub_A = toeplitz(column_vector_aug_a_1_1,row_vector_aug_a_1_1);

column_vector_a_1_2 = [a_1_wls_aug(1,1+n_x:end)'; zeros(n-n_x,1)];
column_vector_aug_a_1_2 = zeros(n_x*numel(column_vector_a_1_2),1);
column_vector_aug_a_1_2(1:n_x:end) = column_vector_a_1_2;
row_vector_aug_a_1_2 = [column_vector_a_1_2(1,1) zeros(1,p*n_z-1)];
toep_a_1_2_sub_A = toeplitz(column_vector_aug_a_1_2,row_vector_aug_a_1_2);

toep_a_1_sub_A = [toep_a_1_1_sub_A; toep_a_1_2_sub_A];

% Second free row of A - K*C.
a_2_wls_aug = [a_2_wls(1,1) 0 0 0 a_2_wls(1,2:end) -1];

column_vector_a_2_1 = [a_2_wls_aug(1,1:n_x)'; zeros(n-n_x,1)];
column_vector_aug_a_2_1 = zeros(n_x*numel(column_vector_a_2_1),1);
column_vector_aug_a_2_1(1:n_x:end) = column_vector_a_2_1;
row_vector_aug_a_2_1 = [column_vector_a_2_1(1,1) zeros(1,p*n_z-1)];
toep_a_2_1_sub_A = toeplitz(column_vector_aug_a_2_1,row_vector_aug_a_2_1);

column_vector_a_2_2 = [a_2_wls_aug(1,1+n_x:end)'; zeros(n-n_x,1)];
column_vector_aug_a_2_2 = zeros(n_x*numel(column_vector_a_2_2),1);
column_vector_aug_a_2_2(1:n_x:end) = column_vector_a_2_2;
row_vector_aug_a_2_2 = [column_vector_a_2_2(1,1) zeros(1,p*n_z-1)];
toep_a_2_2_sub_A = toeplitz(column_vector_aug_a_2_2,row_vector_aug_a_2_2);

toep_a_2_sub_A = [toep_a_2_1_sub_A; toep_a_2_2_sub_A];

Weighting_A1 = toep_a_1_sub_A*weighting_a_1*G_hankel_plus'*M_n1;
Weighting_A2 = toep_a_2_sub_A*weighting_a_2*G_hankel_plus'*M_n2;
Weighting_A = [Weighting_A1 Weighting_A2];
Weighting_tmp = Weighting_A*Xi;

% Permutation matrix from the original row-vectorized implementation.
row1 = 1:n_z*n;
row2 = n_z*n+1:n_z*n_y*n;
pi = reshape([reshape(row1,n_z,[]); reshape(row2,n_z,[])],[],1);

P = eye(n_z*n_y*n);
P = P(:,pi);

Weighting = P - Weighting_tmp;
W_B = inv(Weighting'*Cov_g_ols_row*Weighting);

% WLS solution. Both column-vectorized and row-vectorized versions are kept
% as in the original implementation; the row-vectorized result is used in
% the final output.
B_K_wls_Vec = inv(Phi_n'*W_B*Phi_n)*Phi_n'*W_B*Vec_g_ols_trans_p;
B_K_wls = reshape(B_K_wls_Vec,n_x,[]); %#ok<NASGU>

B_K_wls_Vec_row = Vec_g_ols_trans_p_row*W_B*Phi_n_row'*inv(Phi_n_row*W_B*Phi_n_row');
B_K_wls_row = reshape(B_K_wls_Vec_row,[n_x,n_z])';

%% Output state-space matrices
C_wnsf = C_ols;
K_wnsf = B_K_wls_row(:,1+n_y:end);
B_wnsf = B_K_wls_row(:,1:n_y);
A_wnsf = A_K_wls + K_wnsf*C_wnsf;
D_wnsf = zeros(n_y,n_u);

end

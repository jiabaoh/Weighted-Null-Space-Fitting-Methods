function [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_NSF_SISO(u,y,n_x,n_HOARX)
%FUNC_NSF_SISO Unweighted null-space fitting for SISO state-space models.
%
% [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_NSF_SISO(u,y,n_x,n_HOARX)
%
% Estimate a discrete-time SISO innovation-form state-space model using
% null-space fitting (NSF). This function is the unweighted counterpart of
% the WNSF implementation: a high-order ARX model is first estimated by
% ordinary least squares (OLS), and the reduced-order state-space matrices
% are then obtained using OLS-based null-space fitting only. No weighted
% least-squares (WLS) refinement is performed.
%
% The estimated state-space model has the form
%
%   x(t+1) = A_wnsf*x(t) + B_wnsf*u(t) + K_wnsf*e(t)
%   y(t)   = C_wnsf*x(t) + D_wnsf*u(t) + e(t)
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
% The output variable names are kept consistent with the WNSF functions for
% easier comparison, although this routine performs NSF/OLS only.
%
% Reference implementation standardized for public GitHub release.
%
% Jiabao He, 2026
%

%% Input checks and data orientation
if nargin < 4
    error('Func_NSF_SISO:NotEnoughInputs', ...
        'Four inputs are required: u, y, n_x, and n_HOARX.');
end

if ~isvector(u) || ~isvector(y)
    error('Func_NSF_SISO:SISOOnly', ...
        'This function supports SISO data only. Inputs u and y must be vectors.');
end

u = u(:);
y = y(:);

N = size(u,1);
if size(y,1) ~= N
    error('Func_NSF_SISO:DataLengthMismatch', ...
        'Input and output data must have the same number of samples.');
end

if n_x < 1 || n_HOARX <= n_x
    error('Func_NSF_SISO:InvalidOrders', ...
        'The orders must satisfy n_x >= 1 and n_HOARX > n_x.');
end

if N <= n_HOARX
    error('Func_NSF_SISO:InsufficientData', ...
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

%% Step 1: Estimate high-order ARX coefficients by OLS
g_ols = Y_hankel/Z_hankel;

%% Step 2: OLS null-space fitting for A - K*C
% Construct the block Hankel matrix of high-order ARX coefficients. The
% null-space relation provides the companion-form coefficients of A - K*C.
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

% Convert the OLS polynomial coefficients to companion canonical form.
I_n_x = eye(n_x);
A_K_CF = [-flip(alpha_ols).', I_n_x(:,1:n_x-1)];
C_CF = I_n_x(1,:);

%% Step 3: OLS estimation of B and K
% Build the observability matrix associated with the estimated A - K*C and C.
Obs_ols = zeros(n_HOARX,n_x);
for i = 0:n_HOARX-1
    Obs_ols(i+1,:) = C_CF*(A_K_CF^i);
end

% Separate the interleaved high-order ARX coefficients into input-related
% and output-related parts. This odd/even separation is valid only for SISO
% data ordered as z = [u y].
g_ols_B_K = [g_ols(1:2:end).'; g_ols(2:2:end).'];
data_B = kron(eye(2),Obs_ols);

B_K_CF_ols = data_B\g_ols_B_K;
B_ols = B_K_CF_ols(1:n_x);
K_ols = B_K_CF_ols(n_x+1:end);

%% Output state-space matrices
C_wnsf = C_CF;
K_wnsf = K_ols;
B_wnsf = B_ols;
A_wnsf = A_K_CF + K_wnsf*C_wnsf;
D_wnsf = zeros(n_y,n_u);

end

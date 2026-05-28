function [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_ARMAX(u,y,n_x,n_HOARX)
%FUNC_WNSF_ARMAX Weighted null-space fitting for a SISO ARMAX model.
%
% [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_ARMAX(u,y,n_x,n_HOARX)
%
% Estimate a SISO ARMAX-type polynomial model using a high-order ARX
% approximation and weighted null-space fitting (WNSF), then convert the
% estimated polynomial parameters into an innovation-form state-space model.
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
% n_HOARX  - order of the high-order ARX approximation
%
% Outputs:
% A_wnsf   - estimated state matrix
% B_wnsf   - estimated input matrix
% C_wnsf   - estimated output matrix
% D_wnsf   - estimated feedthrough matrix, fixed to zero in this version
% K_wnsf   - estimated innovation gain
%
% Notes:
% This routine is intended for SISO ARMAX models. It first estimates a
% high-order ARX approximation using the regressor
%
%   [-y(t-1); ... ; -y(t-n); u(t-1); ... ; u(t-n)],
%
% where n = n_HOARX. The reduced-order ARMAX parameters are then obtained
% by WNSF and mapped to a companion-form state-space realization.
%
% This version is standardized for readability while preserving the original
% numerical recipe. In particular, the covariance/weighting definitions and
% explicit inverse-based formulas are kept unchanged from the working
% implementation to avoid changing the empirical performance.
%
% Reference implementation standardized for public GitHub release.
%
% Jiabao He, 2026
%

%% Input checks and data orientation
if nargin < 4
    error('Func_WNSF_ARMAX:NotEnoughInputs', ...
        'Four inputs are required: u, y, n_x, and n_HOARX.');
end

if ~isnumeric(u) || ~isnumeric(y)
    error('Func_WNSF_ARMAX:InvalidDataType', ...
        'Inputs u and y must be numeric arrays.');
end

if ~isvector(u) || ~isvector(y)
    error('Func_WNSF_ARMAX:SISOOnly', ...
        'This function supports SISO data only. Inputs u and y must be vectors.');
end

u = u(:);
y = y(:);

[N,n_u] = size(u);
n_y = size(y,2);
n_z = n_y + n_u;
n = n_HOARX;

if size(y,1) ~= N
    error('Func_WNSF_ARMAX:DataLengthMismatch', ...
        'Input and output data must have the same number of samples.');
end

if n_x < 1 || n <= n_x
    error('Func_WNSF_ARMAX:InvalidOrders', ...
        'The orders must satisfy n_x >= 1 and n_HOARX > n_x.');
end

if N <= n
    error('Func_WNSF_ARMAX:InsufficientData', ...
        'The number of samples N must be larger than n_HOARX.');
end

%% Step 1: Construct the high-order ARMAX/ARX regression data
% The regressor is ordered as
% [-y(t-1); ... ; -y(t-n); u(t-1); ... ; u(t-n)].
z = [u y]';
Z_hankel = zeros(n_z*n,N-n);

for i = 1:(N-n)
    U_hat = zeros(n_u*n,1);
    Y_hat = zeros(n_u*n,1);

    for j = 1:n
        row_idx = (j-1)*n_u + (1:n_u);
        sample_idx = i + n - j;

        U_hat(row_idx,:) =  z(1,sample_idx);
        Y_hat(row_idx,:) = -z(2,sample_idx);
    end

    Z_hankel(:,i) = [Y_hat; U_hat];
end

Y_hankel = y(n+1:end,:)';

%% Step 2: Estimate the high-order ARMAX coefficients by OLS
g_ols = Y_hankel/Z_hankel;

% Keep the original covariance-like matrix used in the working version.
% Note that this is intentionally not inverted here.
Cov_g_ols = Z_hankel*Z_hankel';

%% Step 3: Build the reduced-order null-space equations
bar_I = [eye(n_x); zeros(n-n_x,n_x)];

% AR polynomial part.
alpha_row = [1, zeros(1,n_x-1)];
alpha_column = [1; g_ols(1,1:n-1)'];
Toep_alpha = toeplitz(alpha_column,alpha_row);

% Input polynomial part.
beta_row = [0, zeros(1,n_x-1)];
beta_column = [0; g_ols(1,n+1:end-1)'];
Toep_beta = toeplitz(beta_column,beta_row);

% Null-space equation for theta = [a; b; c].
Q_hankel = [bar_I,        zeros(n,n_x), -Toep_alpha; ...
            zeros(n,n_x), bar_I,        -Toep_beta];

%% Step 4: Initial OLS estimate of the reduced-order parameters
% This explicit inverse form is preserved from the working implementation.
theta_ols = inv(Q_hankel'*Q_hankel)*(Q_hankel')*(g_ols');

%% Step 5: Weighted least-squares refinement
% Construct the weighting matrix from the estimated C-polynomial.
theta_C_row = [1, zeros(1,n-1)];
theta_C_column = [1; theta_ols(2*n_x+1:end,:); zeros(n-n_x-1,1)];
Toep_C = toeplitz(theta_C_column,theta_C_row);

K_weighting = [Toep_C,    zeros(n,n); ...
               zeros(n,n), Toep_C];

% Keep the original inverse-based weighting construction.
K_weighting_inv = inv(K_weighting);
W_theta = (K_weighting_inv)'*Cov_g_ols*K_weighting_inv;

% WLS estimate of theta = [a; b; c].
theta_wls = inv(Q_hankel'*W_theta*Q_hankel)*(Q_hankel')*W_theta*(g_ols');

%% Step 6: Convert the SISO ARMAX model to state-space form
% The polynomial parameters are mapped to a companion-form realization.
% Here c is the estimated noise polynomial parameter and k = c - a.
a = theta_wls(1:n_x,:);
b = theta_wls(n_x+1:2*n_x,:);
c = theta_wls(2*n_x+1:end,:);
k = c - a;

I_n_x = eye(n_x);
A_wnsf = [-a, I_n_x(:,1:n_x-1)];
C_wnsf = I_n_x(1,:);
K_wnsf = k;
B_wnsf = b;
D_wnsf = zeros(n_y,n_u);

end

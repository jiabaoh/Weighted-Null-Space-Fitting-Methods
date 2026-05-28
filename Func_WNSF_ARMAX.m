function [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_ARMAX(u,y,n_x,n_HOARX)
%FUNC_WNSF_ARMAX Estimate a SISO ARMAX model and convert it to state space.
%
% [A_wnsf,B_wnsf,C_wnsf,D_wnsf,K_wnsf] = Func_WNSF_ARMAX(u,y,n_x,n_HOARX)
%
% Estimate a high-order SISO ARMAX-type polynomial model using least
% squares and weighted null-space fitting (WNSF), then convert the estimated
% polynomial model into a companion-form innovation state-space model.
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
% n_HOARX  - order of the high-order ARMAX/ARX approximation
%
% Outputs:
% A_wnsf   - estimated state matrix
% B_wnsf   - estimated input matrix
% C_wnsf   - estimated output matrix
% D_wnsf   - estimated feedthrough matrix, fixed to zero in this version
% K_wnsf   - estimated innovation gain
%
% Notes:
% This routine is intended for SISO ARMAX models. The implementation assumes
% one input and one output. The algorithm first estimates high-order ARMAX
% coefficients, refines a reduced-order polynomial description with WNSF,
% and finally maps the polynomial parameters to a companion-form state-space
% realization.
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

if ~isvector(u) || ~isvector(y)
    error('Func_WNSF_ARMAX:SISOOnly', ...
        'This function supports SISO data only. Inputs u and y must be vectors.');
end

u = u(:);
y = y(:);

N = size(u,1);
if size(y,1) ~= N
    error('Func_WNSF_ARMAX:DataLengthMismatch', ...
        'Input and output data must have the same number of samples.');
end

if n_x < 1 || n_HOARX <= n_x
    error('Func_WNSF_ARMAX:InvalidOrders', ...
        'The orders must satisfy n_x >= 1 and n_HOARX > n_x.');
end

if N <= n_HOARX
    error('Func_WNSF_ARMAX:InsufficientData', ...
        'The number of samples N must be larger than n_HOARX.');
end

%% Dimensions and high-order regression matrix
n_y = 1;
n_u = 1;
n_z = n_y + n_u;
n = n_HOARX;
n_data = N - n;

% The high-order regression vector is ordered as
% [-y(t-1); ... ; -y(t-n); u(t-1); ... ; u(t-n)].
% This corresponds to a SISO ARMAX/ARX polynomial approximation.
Z_hankel = zeros(n_z*n,n_data);
for i = 1:n_data
    y_lag = zeros(n,1);
    u_lag = zeros(n,1);

    for j = 1:n
        sample_idx = i + n - j;
        y_lag(j) = -y(sample_idx);
        u_lag(j) =  u(sample_idx);
    end

    Z_hankel(:,i) = [y_lag; u_lag];
end

Y_hankel = y(n+1:end).';

%% Step 1: Estimate high-order ARMAX coefficients by OLS
g_ols = Y_hankel/Z_hankel;

% Covariance-like weighting matrix for the OLS coefficient estimate. A small
% diagonal loading improves numerical conditioning when the regression matrix
% is close to singular.
lambda = 1e-8;
Cov_g_ols = Z_hankel*Z_hankel.' + lambda*eye(size(Z_hankel,1));

%% Step 2: Build the null-space equations
% The reduced-order parameter vector is
% theta = [a_1 ... a_nx, b_1 ... b_nx, c_1 ... c_nx]'.
% The matrices below encode the polynomial relations between the high-order
% ARMAX estimate and the reduced-order model.
bar_I = [eye(n_x); zeros(n-n_x,n_x)];

alpha_row = [1, zeros(1,n_x-1)];
alpha_column = [1; g_ols(1:n-1).'];
Toep_alpha = toeplitz(alpha_column,alpha_row);

beta_row = [0, zeros(1,n_x-1)];
beta_column = [0; g_ols(n+1:end-1).'];
Toep_beta = toeplitz(beta_column,beta_row);

Q_hankel = [bar_I,       zeros(n,n_x), -Toep_alpha; ...
            zeros(n,n_x), bar_I,       -Toep_beta];

% Initial least-squares estimate of the reduced-order polynomial parameters.
theta_ols = Q_hankel\g_ols.';

%% Step 3: Weighted least-squares refinement
% Construct the weighting matrix from the estimated C-polynomial. The same
% polynomial weighting is used for the output and input coefficient blocks.
theta_C_row = [1, zeros(1,n-1)];
theta_C_column = [1; theta_ols(2*n_x+1:end); zeros(n-n_x-1,1)];
Toep_C = toeplitz(theta_C_column,theta_C_row);

K_weighting = [Toep_C,    zeros(n,n); ...
               zeros(n,n), Toep_C];

K_weighting_inv = K_weighting\eye(size(K_weighting));
W_theta = K_weighting_inv.'*Cov_g_ols*K_weighting_inv;

theta_wls = (Q_hankel.'*W_theta*Q_hankel)\ ...
    (Q_hankel.'*W_theta*g_ols.');

%% Step 4: Convert the SISO ARMAX model to companion-form state space
% The companion realization uses
% A - K*C = [-a I],  C = [1 0 ... 0],  and  K = c - a.
a = theta_wls(1:n_x);
b = theta_wls(n_x+1:2*n_x);
c = theta_wls(2*n_x+1:end);
k = c - a;

I_n_x = eye(n_x);
A_minus_KC = [-a, I_n_x(:,1:n_x-1)];
C_wnsf = I_n_x(1,:);
K_wnsf = k;
B_wnsf = b;
A_wnsf = A_minus_KC + K_wnsf*C_wnsf;
D_wnsf = zeros(n_y,n_u);

end

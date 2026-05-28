function [A_est1_ols,B_est1_ols,C_est1_ols,D_est1_ols,K_est1_ols] = Func_PARSIME(y,u,n_x,f,p,Dflag)
%FUNC_PARSIM_E Robust PARSIM-E implementation for SISO/MISO/MIMO data.
%
% Same interface as the original function:
%   [A,B,C,D,K] = Func_PARSIM_E(y,u,n_x,f,p,Dflag)
%
% Inputs:
%   y      : sample x n_y output matrix
%   u      : sample x n_u input matrix
%   n_x    : model order
%   f      : future block size
%   p      : past block size
%   Dflag  : true/false, estimate direct feedthrough D or force D = 0
%
% Outputs:
%   A_est1_ols : n_x x n_x
%   B_est1_ols : n_x x n_u
%   C_est1_ols : n_y x n_x
%   D_est1_ols : n_y x n_u
%   K_est1_ols : n_x x n_y

% ------------------------- input checks -------------------------
if nargin < 6
    error('Func_PARSIM_E requires exactly 6 inputs: y,u,n_x,f,p,Dflag.');
end
if ~ismatrix(y) || ~ismatrix(u) || size(y,1) ~= size(u,1)
    error('y and u must be 2-D matrices with the same number of rows/samples.');
end
if any(~isfinite(y(:))) || any(~isfinite(u(:)))
    error('y and u must contain only finite values.');
end
if n_x < 1 || f < 2 || p < 1
    error('Require n_x >= 1, f >= 2, and p >= 1.');
end

n_u    = size(u,2);
n_y    = size(y,2);
sample = size(y,1);
N      = sample - (p + f - 1);

if N <= 0
    error('Not enough samples. Need sample >= p + f.');
end
% Build block-Hankel data using the usual PARSIM convention.
[Y_f,Y_p,U_f,U_p] = local_stackdata(y,u,f,p);
Z_p = [Y_p; U_p];
n_zp = size(Z_p,1);       % p*(n_y+n_u)
if n_x > min(f*n_y, n_zp)
    error('n_x must be <= min(f*n_y, p*(n_y+n_u)). Increase f/p or reduce n_x.');
end

% ---------------- estimate extended observability-times-L --------
Obs_L_est_ols = zeros(f*n_y, n_zp);

% First future block.
Y_fi = Y_f(1:n_y,:);
Theta_i = right_lstsq(Y_fi, Z_p);
Obs_L_est_ols(1:n_y,:) = Theta_i(:,1:n_zp);
Innovation_est = Y_fi - Theta_i*Z_p;

% Remaining future blocks.
for i = 2:f
    row_y = (i-1)*n_y + (1:n_y);
    Y_fi = Y_f(row_y,:);

    U_i_minus_1 = U_f(1:(i-1)*n_u,:);
    Reg_i = [Z_p; U_i_minus_1; Innovation_est];

    Theta_i = right_lstsq(Y_fi, Reg_i);
    Obs_L_est_ols(row_y,:) = Theta_i(:,1:n_zp);

    Innovation_est = [Innovation_est; Y_fi - Theta_i*Reg_i]; %#ok<AGROW>
end

% ---------------- weighted SVD and A/C extraction ----------------
% Projection onto the orthogonal complement of future inputs.
UUt = U_f*U_f.';
P_perp = eye(N) - U_f.' * robust_solve(UUt, U_f);
P_perp = real((P_perp + P_perp.')/2);

W2_arg = Z_p * P_perp * Z_p.';
W2 = sqrtm_psd(W2_arg);

[U_ols,S_ols,~] = svd(Obs_L_est_ols * W2, 'econ');
U_ols_1 = U_ols(:,1:n_x);
S_ols_1 = S_ols(1:n_x,1:n_x);

Obs_hat1_ols = U_ols_1 * sqrtm_psd(S_ols_1);
C_est1_ols = Obs_hat1_ols(1:n_y,:);

Obs_down = Obs_hat1_ols(1:(f-1)*n_y,:);
Obs_up   = Obs_hat1_ols(n_y+1:f*n_y,:);
A_est1_ols = left_lstsq(Obs_down, Obs_up);

% ---------------- estimate K from the innovation Toeplitz matrix --------
Y_f_hat = Y_f - Obs_L_est_ols*Z_p;

G_f_ols = zeros(f*n_y, f*n_y);
G_f_ols(1:n_y,1:n_y) = eye(n_y);

R_e = Innovation_est(1:n_y,:) * Innovation_est(1:n_y,:).' / N;
F_est_ols = chol_psd(R_e);
F_est_ols_inv = robust_solve_general(F_est_ols, eye(n_y));

for i = 2:f
    row_y = (i-1)*n_y + (1:n_y);
    Y_fi_hat = Y_f_hat(row_y,:);

    % Direct/current and past input terms have i*n_u columns.
    % Innovation terms have (i-1)*n_y columns.
    U_i = U_f(1:i*n_u,:);
    E_i_minus_1 = Innovation_est(1:(i-1)*n_y,:);
    Reg_i = [U_i; E_i_minus_1];

    Theta_i = right_lstsq(Y_fi_hat, Reg_i);

    % BUGFIX: skip i*n_u input columns, not i*n_y columns.
    G_noise_i = Theta_i(:, i*n_u+1:end);
    G_f_ols(row_y, 1:i*n_y) = [G_noise_i, eye(n_y)];
end

% Average equal block diagonals to enforce lower block-Toeplitz structure.
G_f_ols = enforce_lower_toeplitz(G_f_ols, n_y, f);

G_f_ols_last_blk = G_f_ols(n_y+1:end, 1:n_y);
Obs_hat1_ols_first_blk = Obs_hat1_ols(1:(f-1)*n_y, :);
K_est1_ols = left_lstsq(Obs_hat1_ols_first_blk, G_f_ols_last_blk);

A_est1_ols = real(A_est1_ols);
K_est1_ols = real(K_est1_ols);
C_est1_ols = real(C_est1_ols);

% ---------------- estimate B and, optionally, D ------------------
AK_est1_ols = A_est1_ols - K_est1_ols*C_est1_ols;

best_order = min(3*(f+p), sample-1);
sample_BD = sample - best_order;
if sample_BD <= 0
    error('Not enough samples to estimate B/D. Increase sample size or reduce f/p.');
end

if Dflag
    n_theta = n_x*n_u + n_y*n_u;
else
    n_theta = n_x*n_u;
end

sum_PhiPhi = zeros(n_theta,n_theta);
sum_PhiY   = zeros(n_theta,1);
I_y = eye(n_y);

for i = 1:sample_BD
    k = i + best_order;

    TmpB_K = zeros(n_y, n_u*n_x);
    Tmpy_K = zeros(n_y, 1);

    AKpow = eye(n_x);
    for j = 1:best_order
        k_prev = k - j;
        Cbar = F_est_ols_inv * C_est1_ols * AKpow;

        % Compatible with column-major vec(B): vec(B) = [B(:,1); B(:,2); ...].
        TmpB_K = TmpB_K + kron(u(k_prev,:), Cbar);
        Tmpy_K = Tmpy_K + Cbar * K_est1_ols * y(k_prev,:).';

        AKpow = AKpow * AK_est1_ols;
    end

    Phi_Y = F_est_ols_inv*y(k,:).' - Tmpy_K;

    if Dflag
        Phi_D_0 = kron(u(k,:), I_y);
        Phi = [TmpB_K, Phi_D_0];
    else
        Phi = TmpB_K;
    end

    sum_PhiPhi = sum_PhiPhi + Phi.'*Phi;
    sum_PhiY   = sum_PhiY   + Phi.'*Phi_Y;
end

Vec_theta = robust_solve(sum_PhiPhi, sum_PhiY);
Vec_BK = Vec_theta(1:n_x*n_u,:);
B_minus_KD = reshape(Vec_BK, n_x, n_u);

if Dflag
    Vec_D = Vec_theta(n_x*n_u+1:end,:);
    D_est1_ols = F_est_ols * reshape(Vec_D, n_y, n_u);
    B_est1_ols = B_minus_KD + K_est1_ols*D_est1_ols;
else
    D_est1_ols = zeros(n_y,n_u);
    B_est1_ols = B_minus_KD;
end

B_est1_ols = real(B_est1_ols);
D_est1_ols = real(D_est1_ols);
end

% ======================================================================
% Local helper functions
% ======================================================================
function [Y_f,Y_p,U_f,U_p] = local_stackdata(y,u,f,p)
%LOCAL_STACKDATA Block-Hankel stacking.
% Column k contains:
%   past   : samples k, ..., k+p-1
%   future : samples k+p, ..., k+p+f-1
sample = size(y,1);
n_y = size(y,2);
n_u = size(u,2);
N = sample - (p + f - 1);

Y_p = zeros(p*n_y, N);
U_p = zeros(p*n_u, N);
Y_f = zeros(f*n_y, N);
U_f = zeros(f*n_u, N);

for k = 1:N
    for i = 1:p
        Y_p((i-1)*n_y+(1:n_y), k) = y(k+i-1,:).';
        U_p((i-1)*n_u+(1:n_u), k) = u(k+i-1,:).';
    end
    for i = 1:f
        Y_f((i-1)*n_y+(1:n_y), k) = y(k+p+i-1,:).';
        U_f((i-1)*n_u+(1:n_u), k) = u(k+p+i-1,:).';
    end
end
end

function Theta = right_lstsq(Y, X)
%RIGHT_LSTSQ Solve Y ~= Theta*X robustly without pinv(X).
G = X*X.';
RHS = X*Y.';
Theta = robust_solve(G, RHS).';
end

function Theta = left_lstsq(X, Y)
%LEFT_LSTSQ Solve X*Theta ~= Y robustly.
G = X.'*X;
RHS = X.'*Y;
Theta = robust_solve(G, RHS);
end

function X = robust_solve(A, B)
%ROBUST_SOLVE Solve A*X=B. Adds tiny ridge only when A is ill-conditioned.
A = real((A + A.')/2);
n = size(A,1);
if n == 0
    X = zeros(0,size(B,2));
    return;
end

if all(isfinite(A(:))) && rcond(A) > 1e-12
    X = A\B;
else
    scale = trace(A)/max(n,1);
    if ~isfinite(scale) || scale <= 0
        scale = norm(A,'fro')/max(n,1);
    end
    if ~isfinite(scale) || scale <= 0
        scale = 1;
    end
    lambda = sqrt(eps(class(A))) * scale;
    X = (A + lambda*eye(n,class(A)))\B;
end
end


function X = robust_solve_general(A, B)
%ROBUST_SOLVE_GENERAL Solve A*X=B for a possibly nonsymmetric square A.
n = size(A,1);
if n == 0
    X = zeros(0,size(B,2));
    return;
end
if all(isfinite(A(:))) && rcond(A) > 1e-12
    X = A\B;
else
    scale = norm(A,'fro')/max(n,1);
    if ~isfinite(scale) || scale <= 0
        scale = 1;
    end
    lambda = sqrt(eps(class(A))) * scale;
    X = (A + lambda*eye(n,class(A)))\B;
end
end

function Sroot = sqrtm_psd(S)
%SQRTM_PSD Symmetric positive-semidefinite matrix square root.
S = real((S + S.')/2);
[V,D] = eig(S);
d = real(diag(D));
d(d < 0 & abs(d) < 1e-10*max(1,max(abs(d)))) = 0;
d = max(d,0);
Sroot = real(V*diag(sqrt(d))*V.');
Sroot = real((Sroot + Sroot.')/2);
end

function L = chol_psd(S)
%CHOL_PSD Lower factor L with L*L' approximately equal to S.
S = real((S + S.')/2);
n = size(S,1);
[L,pflag] = chol(S,'lower');
if pflag == 0
    return;
end

scale = norm(S,'fro')/max(n,1);
if ~isfinite(scale) || scale <= 0
    scale = 1;
end
jitter = sqrt(eps(class(S))) * scale;
for kk = 1:12
    [L,pflag] = chol(S + jitter*eye(n,class(S)),'lower');
    if pflag == 0
        return;
    end
    jitter = 10*jitter;
end

% Last-resort PSD factor; not triangular, but square and invertible after floor.
[V,D] = eig(S);
d = max(real(diag(D)), jitter);
L = real(V*diag(sqrt(d)));
end

function G_toep = enforce_lower_toeplitz(G, n_y, f)
%ENFORCE_LOWER_TOEPLITZ Average equal block diagonals of an f-by-f block matrix.
G_toep = zeros(size(G));
G_blocks = zeros(n_y,n_y,f);

for lag = 0:f-1
    acc = zeros(n_y,n_y);
    count = 0;
    for row_blk = lag+1:f
        col_blk = row_blk - lag;
        row_idx = (row_blk-1)*n_y + (1:n_y);
        col_idx = (col_blk-1)*n_y + (1:n_y);
        acc = acc + G(row_idx,col_idx);
        count = count + 1;
    end
    G_blocks(:,:,lag+1) = acc / count;
end

% The leading innovation term should be identity.
G_blocks(:,:,1) = eye(n_y);

for row_blk = 1:f
    row_idx = (row_blk-1)*n_y + (1:n_y);
    for col_blk = 1:row_blk
        col_idx = (col_blk-1)*n_y + (1:n_y);
        lag = row_blk - col_blk;
        G_toep(row_idx,col_idx) = G_blocks(:,:,lag+1);
    end
end
end

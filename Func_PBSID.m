function [Ad,Bd,Cd,Dd,K,Lambda] = Func_PBSID(y,u,ksf,ksp,n,dflag);
%
%   Predictor-Based Subspace Identification
%   ---------------------------------------
% for the implemetation of this method, we refer to the PBSID-Toolbox
% https://github.com/jwvanwingerden/PBSID-Toolbox

%
%   The algorithm identifies a state-space innovation model
%   from input-output data:
%
%           x_{k+1} = Ad x_k + Bd u_k + K e_k
%               y_k = Cd x_k + Dd u_k + e_k
%            cov(e) = Lambda
%
%   In this implementation, the direct feedthrough term is
%   constrained to zero:
%
%           Dd = 0
%
%
%   Usage:
%
%           [Ad,Bd,Cd,Dd,K,Lambda] = ...
%               Func_PBSID(y,u,ksf,ksp,n,dflag);
%
%
%   Inputs:
%
%           y:      matrix of measured outputs
%           u:      matrix of measured inputs
%           ksf:    number of future block rows
%           ksp:    number of past block rows
%           n:      model order
%           dflag:  direct-term estimation flag
%
%                   In this implementation Dd is always set to zero.
%                   The argument is retained for interface compatibility.
%
%
%   Outputs:
%
%           Ad:       state transition matrix
%           Bd:       input matrix
%           Cd:       output matrix
%           Dd:       direct feedthrough matrix
%           K:        innovation gain
%           Lambda:   innovation covariance matrix
%
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         CHECK THE ARGUMENTS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Check the number of input arguments
if (nargin < 2)
    error('Func_PBSID needs at least two arguments');
end


% Default value of dflag
if (nargin < 6) || isempty(dflag)
    dflag = 0;      % Do not estimate the direct term by default
end


% Default model-order range
if (nargin < 5) || isempty(n)
    n = 1:10;
end


% Estimate the past horizon
if (nargin < 4) || isempty(ksp)
    ksp = estimate_past_with_AiC(y,u);
end


% Set the future horizon
if (nargin < 3) || isempty(ksf)
    ksf = ksp;
else
    ksf = min(ksf,ksp);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                           CHECK THE DATA
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Dimensions
[p,Ny] = size(y);
[m,Nu] = size(u);

pp = p + m;


% Check consistency between input and output data
if (Nu ~= Ny)
    error('u and y must have the same number of columns');
end


% Number of effective data points
N = Ny - (ksf + ksp) + 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                           BEGIN ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% **************************************
%               STEP 1
%     High-order ARX estimation
% **************************************


% Combined input-output data
z = [y;u];


% Construct regression matrices
YYarx = y(:,ksp+1:ksp+1+N+ksf-2) / sqrt(N+ksf-1);
ZZarx = Hankel(z,ksp,N+ksf-1);


% Estimate high-order ARX parameters
THETA = YYarx / ZZarx;


% **************************************
%               STEP 2
%    Construct predictor matrices
% **************************************


% Initialize past predictor matrices
Past  = zeros(ksf*p-p,(ksp+1)*pp);
Past1 = zeros(ksf*p-p,(ksp+1)*pp);


% Construct past predictor matrices
for i = 1:ksf-1
    for j = i:ksp

        Past((i-1)*p+1:i*p,(j-1)*pp+1:j*pp) = ...
            THETA(:,(j-i)*pp+1:(j-i+1)*pp);

        Past1((i-1)*p+1:i*p,j*pp+1:j*pp+pp) = ...
            THETA(:,(j-i)*pp+1:(j-i+1)*pp);

    end
end


% Initialize future predictor matrix
Future = zeros(ksf*p-p,(ksf-1)*pp);


% Construct future predictor matrix
for i = 2:ksf-1
    for j = 1:i-1

        Future((i-1)*p+1:i*p,(j-1)*pp+1:j*pp) = ...
            THETA(:,(ksp-i+j)*pp+1:(ksp-i+j+1)*pp);

    end
end


% **************************************
%               STEP 3
%      Construct Hankel matrices
% **************************************


% Current output data
Y = y(:,ksp+1:ksp+N) / sqrt(N);


% Output Hankel matrix
Yc = Hankel(y,ksp+ksf-1,N);
Yc = Yc(ksp*p+1:(ksp+ksf-1)*p,:);


% Combined input-output Hankel matrix
Zc = Hankel(z,ksp+ksf-1,N);


% Past data
Zpp = Zc(1:(ksp+1)*pp,:);


% Future data
Zc = Zc(ksp*pp+1:(ksp+ksf-1)*pp,:);


% Remove the future contribution
Yc = Yc - Future*Zc;


% **************************************
%               STEP 4
%        Compute weighting matrix
% **************************************


% Output covariance matrix
W = Yc*Yc';


% Regularized square-root weighting matrix
W = sqrtm(W + 0.01*eye(size(W)));


% Inverse weighting matrix
Wp = pinv(W);


% Clear unused large matrices
clear Yc Zc;


% **************************************
%               STEP 5
%       Compute predicted outputs
% **************************************


% Predicted future outputs
Yh = Past*Zpp;


% One-step-shifted predicted future outputs
Yh1 = Past1*Zpp;


% **************************************
%               STEP 6
%               SVD
% **************************************


% Compute the weighted SVD
[Us,Ss,~] = svd(Wp*Yh*Yh'*Wp');


% **************************************
%               STEP 7
%       Determine the model order
% **************************************


if min(n) < max(n)

    % Singular values
    testo = diag(Ss);

    % Default model order
    ndef = max(max(find(testo > (max(testo)+min(testo))/2)));

    % Display singular values
    figure;
    bar(testo);

    title('Select model order in command window.');

    disp('Press any key to continue');
    disp(['default=',num2str(ndef)]);

    n = input('Select model order: (''Return'' gives default) ');

    if isempty(n)
        n = ndef;
        disp(['Order chosen to ',int2str(n)]);
    end

    close;


elseif isempty(n)

    testo = diag(Ss);

    n = max(max(find(testo > ...
        (max(testo)+min(testo))/2)));

end


% **************************************
%               STEP 8
%       Estimate the state sequence
% **************************************


% Extended observability subspace
Gammak = Us(:,1:n);


% Weighted left inverse
pGammak = Gammak'*Wp;


% Input data
U = u(:,ksp+1:ksp+N) / sqrt(N);


% Estimated state sequences
X  = pGammak*Yh;
X1 = pGammak*Yh1;


% **************************************
%               STEP 9
%      Estimate innovation sequence
% **************************************


% Initial output regression
Theta2 = Y / X;


% Estimated innovations
Eh = Y - Theta2*X;


% **************************************
%               STEP 10
%      Estimate Ad, Bd, and K
% **************************************


% State transition regression
Theta1 = X1 / [X;U;Eh];


% Extract system matrices
Ad = Theta1(1:n,1:n);

Bd = Theta1(1:n,n+1:n+m);

K = Theta1(1:n,n+m+1:n+m+p);


% **************************************
%               STEP 11
%          Estimate Cd and Dd
% **************************************


% Output matrix regression
Theta2 = Yh / X;


% Extract output matrix
Cd = Theta2(1:p,1:n);


% Enforce zero direct feedthrough
Dd = zeros(p,m);


% **************************************
%               STEP 12
%     Estimate innovation covariance
% **************************************


Lambda = Eh*Eh';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                            END ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end
    

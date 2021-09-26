function [output] = GPC_binary(X,Y,COVF,InitialPara,loglik,Dr1_loglik,Dr2_loglik,Xstar)
% Training data X(d*N)
%

% Number and dimension of samples
NumTrain = length(Y);
NumTest = length(Xstar(1,:));
Dim = length(X(:,1));

% Number of iteration
I = 10;

% initial hyperparameters
theta = InitialPara;

% initial latent function
f = zeros(NumTrain,1);

% % covariance matrix of training data
% for i = 1:NumTrain
%     for j = 1:NumTrain
%         K(i,j) = COVF(X(:,i),X(:,j),sf,l);
%     end
% end

temp = 100;
for count = 1:I
    
% find f_hat by minimizing the negative posterior -p(f|X,y)
Neg_Log_Post = @(f) 0.5*f'*inv(K(theta,X,COVF))*f + 0.5*log(det(K(theta,X,COVF))) + 0.5*NumTrain*log(2*pi) - Lik(f,loglik,Y);
Dr1_Neg_Log_Post = @(f) inv(K(theta,X,COVF))*f - Dr1Lik(f,Dr1_loglik,Y);
Dr2_Neg_Log_Post = @(f) inv(K(theta,X,COVF)) - Dr2Lik(f,Dr2_loglik,Y);

[f_hat, neg_posterior_f] = Fun_mini_Newton(Neg_Log_Post,Dr1_Neg_Log_Post,Dr2_Neg_Log_Post,f,1e3,1e-6,0.1);

% optimize hyperparameters by minimizing the negative marginal likelihood -q(y|X,theta)
W = -Dr2Lik(f_hat,Dr2_loglik,Y);
B = @(theta) eye(NumTrain) + sqrt(W)*K(theta,X,COVF)*sqrt(W);
Neg_Log_Marg = @(theta) 0.5*f_hat'*inv(K(theta,X,COVF))*f_hat - Lik(f_hat,loglik,Y) + 0.5*log(det(eye(NumTrain)+sqrt(W)*K(theta,X,COVF)*sqrt(W)));
Dr1_Neg_Log_Marg = @(theta) Fun_Dr1_Neg_Log_Marg(theta,X,f_hat,W,B);
Dr2_Neg_Log_Marg = @(theta) Fun_Dr2_Neg_Log_Marg(theta,X,f_hat,W,B);

[theta_hat, neg_marginal] = Fun_mini_gredd(Neg_Log_Marg,Dr1_Neg_Log_Marg,theta,1e3,1e-6,0.001)

%[theta_hat, marginal] = Fun_mini_Newton(Neg_Log_Marg,Dr1_Neg_Log_Marg,Dr2_Neg_Log_Marg,InitialPara,1e3,1e-6,0.001);

if temp-neg_marginal<0.1
    break;
end

f = f_hat;
theta = theta_hat;

temp = neg_marginal;
end

% Predict the mean and covariance of test latent variables
W = -Dr2Lik(f,Dr2_loglik,Y);
Bmatrix = B(theta);
L = chol(Bmatrix);
fstar_mean = Kstar(theta,X,Xstar,COVF)'*Dr1Lik(f,Dr1_loglik,Y);
v = L\(sqrt(W)*Kstar(theta,X,Xstar,COVF));
fstar_cov = Kstar2(theta,Xstar,COVF) - v'*v;

% Compute predictive probabilites
%MAP prediction
output = 1./(ones(NumTest,1)+exp(-fstar_mean));

end

function output = Lik(f,loglik,Y)
output = 0;
for fi = 1:length(f)
    output = output + loglik(Y(fi),f(fi));
end
end

function output = Dr1Lik(f,Dr1_loglik,Y)
for fi = 1:length(f)
    output(fi,1) = Dr1_loglik(Y(fi),f(fi));
end
end

function output = Dr2Lik(f,Dr2_loglik,Y)
for fi = 1:length(f)
    output(fi,1) = Dr2_loglik(Y(fi),f(fi));
end
output = diag(output);
end

function output = K(theta,X,COVF)
for fi = 1:length(X(1,:))
    for fj = 1:length(X(1,:))
        output(fi,fj) = COVF(X(:,fi),X(:,fj),theta(1),exp(theta(2)));
    end
end
end

function output = Kstar(theta,X,Xstar,COVF)
for fi = 1:length(X(1,:))
    for fj = 1:length(Xstar(1,:))
        output(fi,fj) = COVF(X(:,fi),Xstar(:,fj),theta(1),exp(theta(2)));
    end
end
end

function output = Kstar2(theta,Xstar,COVF)
for fi = 1:length(Xstar(1,:))
    for fj = 1:length(Xstar(1,:))
        output(fi,fj) = COVF(Xstar(:,fi),Xstar(:,fj),theta(1),exp(theta(2)));
    end
end
end

function output = Fun_Dr1_Neg_Log_Marg(theta,X,f_hat,W,B)
sf = theta(1);
l = exp(theta(2));
for fi = 1:length(X(1,:))
    for fj = 1:length(X(1,:))
        K(fi,fj) = sf^2*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2);
        Dr1K_sf(fi,fj) = 2*sf*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2);
        Dr1K_logl(fi,fj) = sf^2*norm(X(:,fi)-X(:,fj))^2/l^2*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2);
    end
end
output(1,1) = -0.5*f_hat'*(inv(K)*Dr1K_sf*inv(K))*f_hat + 0.5*trace(inv(B(theta))*W*Dr1K_sf);
output(2,1) = -0.5*f_hat'*(inv(K)*Dr1K_logl*inv(K))*f_hat + 0.5*trace(inv(B(theta))*W*Dr1K_logl);
end

function output = Fun_Dr2_Neg_Log_Marg(theta,X,f_hat,W,B)
sf = theta(1);
l = exp(theta(2));
for fi = 1:length(X(1,:))
    for fj = 1:length(X(1,:))
        K(fi,fj) = sf^2*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2);
        Dr1K_sf(fi,fj) = 2*sf*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2);
        Dr1K_logl(fi,fj) = sf^2*norm(X(:,fi)-X(:,fj))^2/l^2*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2);
        Dr2K_sf(fi,fj) = 2*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2);
        Dr2K_logl(fi,fj) = sf^2*norm(X(:,fi)-X(:,fj))^2*(-2*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2)/l^2+norm(X(:,fi)-X(:,fj))^2*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2)/l^4);
        Dr2K_sf2l(fi,fj) = 2*sf*norm(X(:,fi)-X(:,fj))^2/l^2*exp(-0.5/l^2*norm(X(:,fi)-X(:,fj))^2);
        Dr2K_l2sf(fi,fj) = Dr2K_sf2l(fi,fj);
    end
end
output(1,1) = 0.5*trace(-W*inv(B(theta))*W*Dr1K_sf*inv(B(theta))*Dr1K_sf+W*inv(B(theta))*Dr2K_sf) -...
    -0.5*f_hat'*inv(K)*(-2*Dr1K_sf*inv(K)*Dr1K_sf+Dr2K_sf)*inv(K)*f_hat;
output(1,2) = 0.5*trace(-W*inv(B(theta))*W*Dr1K_logl*inv(B(theta))*Dr1K_sf+W*inv(B(theta))*Dr2K_sf2l) -...
    -0.5*f_hat'*inv(K)*(-Dr1K_logl*inv(K)*Dr1K_sf+Dr2K_sf2l-Dr1K_sf*inv(K)*Dr1K_logl)*inv(K)*f_hat;
output(2,1) = 0.5*trace(-W*inv(B(theta))*W*Dr1K_sf*inv(B(theta))*Dr1K_logl+W*inv(B(theta))*Dr2K_l2sf) -...
    -0.5*f_hat'*inv(K)*(-Dr1K_sf*inv(K)*Dr1K_logl+Dr2K_l2sf-Dr1K_logl*inv(K)*Dr1K_sf)*inv(K)*f_hat;
output(2,2) = 0.5*trace(-W*inv(B(theta))*W*Dr1K_logl*inv(B(theta))*Dr1K_logl+W*inv(B(theta))*Dr2K_logl) -...
    -0.5*f_hat'*inv(K)*(-2*Dr1K_logl*inv(K)*Dr1K_logl+Dr2K_logl)*inv(K)*f_hat;

end


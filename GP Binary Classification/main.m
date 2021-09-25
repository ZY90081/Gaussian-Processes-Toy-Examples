% This code is a toy example of GP classification
% The inputs are vector and outputs are binary labels
% Liu Yang @ Stony Brook
% 09-29-2020.

close all;
clear all;
clc;

%% Generate training set and testing set
Num_train = 50;
Num_test = 10;
Mu_c1 = [2;5];   % mean of class 1
Mu_c2 = [5;-1];  % mean of class 2
Cov_c1 = diag([6 10]);   % covariance of class 1
Cov_c2 = [10 5;5 10];    % covariance of class 2
X_c1 = chol(Cov_c1)'*randn(2,Num_train/2)+repmat(Mu_c1,1,Num_train/2);
X_c2 = chol(Cov_c2)'*randn(2,Num_train/2)+repmat(Mu_c2,1,Num_train/2);
TrainX = [X_c1 X_c2];
TrainY = [repmat(-1,1,Num_train/2) repmat(1,1,Num_train/2)]';
Xstar_c1 = chol(Cov_c1)'*randn(2,Num_test/2)+repmat(Mu_c1,1,Num_test/2);
Xstar_c2 = chol(Cov_c2)'*randn(2,Num_test/2)+repmat(Mu_c2,1,Num_test/2);
TestX = [Xstar_c1 Xstar_c2];
TestY = [repmat(-1,1,Num_test/2) repmat(1,1,Num_test/2)]';

%plotting
figure(); hold on;
plot(X_c1(1,:),X_c1(2,:),'or','LineWidth',2);
plot(X_c2(1,:),X_c2(2,:),'+b','LineWidth',2);
plot(TestX(1,:),TestX(2,:),'p','LineWidth',2,'MarkerSize',6);

figure();
x=linspace(-5, 15);
y=linspace(-10, 15);
[X,Y]=meshgrid(x,y);
Z = [X(:) Y(:)];
z1 = mvnpdf(Z,Mu_c1',Cov_c1);
z2 = mvnpdf(Z,Mu_c2',Cov_c2);
z = z1+z2;
z =  reshape(z,length(y),length(x));
surf(X,Y,z,'FaceAlpha',0.9);
shading flat
axis tight
figure()
contour(X,Y,z);

clear x y z X Y Z z1 z2 

%% Specify the covariance function

% squared exponential
K = @(x1,x2,sf,l) sf^2*exp(-0.5/l^2*norm(x1-x2)^2);

%% Specify the likelihood function

% logistic regression
Logistic = @(y,f) 1./(1+exp(-y*f));
Log_logistic = @(y,f) -log(1+exp(-y*f));
De1_log_logistic = @(y,f) (y+1)/2 - 1./(1+exp(-f));
De2_log_logistic = @(y,f) -1./(1+exp(-f))*(1-1./(1+exp(-f)));

% probit regression
CumGauss = @(y,f) normcdf(y*f);
Log_cumGauss = @(y,f) log(normcdf(y*f));
De1_log_cumGauss = @(y,f) y*normpdf(f)/normcdf(y*f);
De2_log_cumGauss = @(y,f) -normpdf(f)^2/normcdf(y*f)^2 - y*f*normpdf(f)/normcdf(y*f);


%% 


hyperp = [1;0]; % initial hyperparameters

outputprob = GPC_binary(TrainX,TrainY,K,hyperp,Log_logistic,De1_log_logistic,De2_log_logistic,TestX);

figure(1)
for i = 1:Num_test
    text(TestX(1,i)+0.1,TestX(2,i),sprintf('%0.4f',outputprob(i)));
end


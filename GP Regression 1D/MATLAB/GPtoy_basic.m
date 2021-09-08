% This code is a GP toy to improve my understanding.
% Both inputs and outputs are scalers.
% Liu Yang @ Stony Brook
% 07-20-2018.
% Update 08-09-2019


close all;
clear all;
clc;

%% 

xo = -4:0.1:4;
yo = sin(3*xo)-2*sin(xo+pi/2); % Real function.
en = 0.5; % standard derivation of error.
x = [-3.5 -3.1 -3 -2.5 -1.5 -1.1 1 1.1 1.3 1.9 2.1 2.9 3.3 3.8];
y = sin(3*x)-2*sin(x+pi/2) + randn(1,length(x))*en; % Training data.
N = length(x); % number of training data.

figure(1);  %==========================Ground truth and training data
hold on; grid on;
plot(xo,yo,'-');
plot(x,y,'o');
legend('Ground Truth','Training Data(with noise)');

% Optimize hyperparameter by computing marginal likelihood
theta = MarginalLikelihood([x;y],en);
sigmaf = theta(1);
l = 10^(theta(2));

% Optimize hyperparameter by cross-validation
theta = LOO_CV([x;y],en);
sigmaf = theta(1);
l = 10^(theta(2));

% meanfunc = [];
% covfunc = @covSEiso;
% likfunc = @likGauss;
% hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
% hyp2 = minimize(hyp, @gp, -1000, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
% sigmaf = exp(hyp2.cov(2));
% l = exp(hyp2.cov(1));


k = @(x1,x2) sigmaf^2*exp(-1/(2*l^2)*(x1-x2)^2); % covariance function.
itv = -10:0.1:10;
n = length(itv);
Pcov = zeros(n,n);
for i = 1:n
    for j = 1:n
        Pcov(i,j) = k(itv(i),itv(j));
    end
end

figure(2);  %==========================Prior covariance of f*
[X,Y] = meshgrid(itv);
pcolor(X,Y,Pcov);
axis ij
shading interp;
colorbar;

Pvar = k(0,0); % prior variance of f*
Pstd = sqrt(Pvar); % prior standard deviation
Pavg = zeros(1,n); % prior mean.
Pgp = mvnrnd(Pavg,Pcov,3); % draw three random function from prior gp.

figure(3);  %=========================Prior function samples
hold on; grid on;
plot(itv,Pavg,'b--','LineWidth',2);
%plot(itv,Pavg+2*Pstd,'k-');
%plot(itv,Pavg-2*Pstd,'k-');
patch([itv fliplr(itv)],[Pavg+2*Pstd fliplr(Pavg-2*Pstd)],[0.93,0.84,0.84],'EdgeColor','none');
axis([-10,10,-4,4]);
alpha(0.8);
plot(itv,Pgp(1,:),'r');
plot(itv,Pgp(2,:),'g');
plot(itv,Pgp(3,:),'m');
legend('mean','95% confidence region','sample1','sample2','sample3');

ktrte = zeros(N,n);
ktrtr = zeros(N,N);
for i = 1:n
    for j = 1:N
        ktrte(j,i) = k(itv(i),x(j)); % cov of training data and testing data
    end
end
for i = 1:N
    for j = 1:N
        ktrtr(i,j) = k(x(i),x(j)); % cov of training data
    end
end

%$$$$$$Taking inverse 
% L = ktrte'*inv(ktrtr+en^2*eye(N));
% Qavg = L*y'; % posterior mean.
% Qcov = Pcov - L*ktrte;
%$$$$$$$$$$$$$$$$$$$

%$$$$$$$Algorithm 2.1    
% L = chol(ktrtr,'lower');
% Qavg = ktrte'*(L'\(L\y'));
% v = L\ktrte;
% Qcov = Pcov - v'*v;

Qavg = ktrte'*((ktrtr+en^2*eye(N))\y');
Qcov = Pcov - ktrte'*((ktrtr+en^2*eye(N))\ktrte);
%$$$$$$$$$$$$$$$$$$$$


figure(4);  %==========================Posterior covariance of f*
[X,Y] = meshgrid(itv);
pcolor(X,Y,Qcov);
axis ij
shading interp;
colorbar;

Qvar = diag(Qcov); % Posterior variances of each testing points.
Qstd = sqrt(Qvar);
Qgp = mvnrnd(Qavg,Qcov,3); % draw three random function from posterior gp.


figure(5);  %==========================Posterior function samples
hold on; grid on;
plot(itv,Qavg,'b--','LineWidth',2);
patch([itv fliplr(itv)],[Qavg'+2*Qstd' fliplr(Qavg'-2*Qstd')],[0.93,0.84,0.84],'EdgeColor','none');
axis([-10,10,-4,4]);
alpha(0.8);
plot(x,y,'ro');
plot(itv,Qgp(1,:),'r');
plot(itv,Qgp(2,:),'g');
plot(itv,Qgp(3,:),'m');
legend('mean','95% confidence region','training data (with noise)','sample1','sample2','sample3');

figure(6);    %========================Samples of Post. cov 
hold on; grid on;
x_ = [-3 1 7]; % three input samples.
mark = ['r','g','m'];
m = length(x_);
for i = 1:m
    idx = find(itv==x_(i));
    plot(itv,Qcov(idx,:),mark(i));
end
legend('x*=-3','x*=1','x*=7');

ycov = Qcov + en^2*eye(n); %posterior cov of output y*
yvar = diag(ycov);
ystd = sqrt(yvar);
figure(7)   %========================Posterior output samples
hold on; grid on;
plot(itv,Qavg,'b--','LineWidth',2);
patch([itv fliplr(itv)],[Qavg'+2*ystd' fliplr(Qavg'-2*ystd')],[0.93,0.84,0.84],'EdgeColor','none');
axis([-10,10,-4,4]);
alpha(0.8);
plot(x,y,'ro');
legend('mean of y*','95% confidence region','training data (with noise)');


figure(8)   %==========================Compare the real function and the prediction.
hold on; grid on;
plot(xo,yo,'r-','LineWidth',2);
plot(itv,Qavg,'b--');
%plot(itv,Qavg1,'k--');
patch([itv fliplr(itv)],[Qavg'+2*ystd' fliplr(Qavg'-2*ystd')],[0.95,0.87,0.73],'EdgeColor','none');
patch([itv fliplr(itv)],[Qavg'+2*Qstd' fliplr(Qavg'-2*Qstd')],[0.75,0.86,0.77],'EdgeColor','none');
alpha(0.7);
axis([-4,4,-4,4]);
legend('ground truth','mean of f* and y*','95% confidence region of y*','95% confidence region of f*');


%% Compare weight function, equivalent kernel

xo = 0:0.01:1;
yo = sin(3*xo)-2*sin(0.7*xo+pi/2); % Real function.
en = sqrt(0.1); % standard derivation of error.
N = 50; % number of training data.
x = rand(1,N); % generate training data randomly.
y = sin(3*x)-2*sin(0.7*x+pi/2) + randn(1,N)*en; % Training data.

sigmaf = 1; % hyperparameters.
l = 0.004;
k = @(x1,x2) sigmaf^2*exp(-1/(2*l)*(x1-x2)^2); % covariance function.

xstar1 = 0.5;   % test data
xstar2 = 0.06;

for i = 1:N
    for j = 1:N
        K(i,j) = k(x(i),x(j));
    end
    kstar1(i) = k(x(i),xstar1);
    kstar2(i) = k(x(i),xstar2);
end

h1 = inv(K+en^2*eye(N))*kstar1';
h2 = inv(K+en^2*eye(N))*kstar2';

figure(9);    %==========================weight function
subplot(2,2,1)
hold on; grid on;
plot(x,h1,'.','MarkerSize',10);
subplot(2,2,2)
hold on; grid on;
plot(x,h2,'.','MarkerSize',10);

enpri = sqrt(10); % change the error
h3 = inv(K+enpri^2*eye(N))*kstar1';
subplot(2,2,3)
hold on; grid on;
plot(x,h3,'.','MarkerSize',10);


NN = 50; % number of points in a dense grid
xx = 1/NN:1/NN:1; % points in a dense grid
sigma = (en*NN/N)^2;
for i = 1:NN
    for j = 1:NN
        KK(i,j) = k(xx(i),xx(j));
    end
end
S = KK*inv(KK+sigma*eye(NN));  % Smoothing Matrix
h1 = S(find(xx==xstar1),:);   % find the corresponding column of test data
h2 = S(find(xx==xstar2),:);
sigma = (enpri*NN/N)^2;
S = KK*inv(KK+sigma*eye(NN)); 
h3 = S(find(xx==xstar1),:);

figure(9);    %==========================equivalent kernel
subplot(2,2,1)
plot(xx,h1,'-','LineWidth',2);
subplot(2,2,2)
plot(xx,h2,'-','LineWidth',2);
subplot(2,2,3)
plot(xx,h3,'-','LineWidth',2);

figure(9);    %==========================SE kernel
subplot(2,2,1)
plot(xx,KK(:,find(xx==xstar1))*(max(h1)/max(KK(:,find(xx==xstar1)))),'--','LineWidth',2);
subplot(2,2,2)
plot(xx,KK(:,find(xx==xstar2))*(max(h2)/max(KK(:,find(xx==xstar2)))),'--','LineWidth',2);
subplot(2,2,3)
plot(xx,KK(:,find(xx==xstar1))*(max(h3)/max(KK(:,find(xx==xstar1)))),'--','LineWidth',2);


figure(9) 
subplot(2,2,4)
plot(xx,h1,'-','LineWidth',2);
hold on;

NN2 = 200;
xx = 1/NN2:1/NN2:1; % points in a dense grid
sigma = (en)^2;
for i = 1:NN2
    for j = 1:NN2
        KK(i,j) = k(xx(i),xx(j));
    end
end
S = KK*inv(KK+sigma*eye(NN2));  % Smoothing Matrix
h2 = S(find(xx==xstar1),:);


plot(xx,h2,'-','LineWidth',2);

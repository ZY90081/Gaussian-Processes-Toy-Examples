% This code is a GP toy to improve my understanding on model selection
% Liu Yang @ Stony Brook

close all;
clear all;
clc;

%% Squared exponential covariance function with different hyperparameters

D = 2; % dimension
l1 = 1;  % for M1
l2 = [1;5]; % for M2
A = [1;-1]; l3 = [6;6]; % for M3

M1 = l1^(-2)*eye(D);
M2 = diag(l2.^(-2));
M3 = A*A' + diag(l3.^(-2));

sigmaf = 1;
sigman = 0.01;

k = @(x1,x2,M) sigmaf^2*exp(-0.5*(x1-x2)'*M*(x1-x2))+sigman;
range = -2:0.2:2;
n = length(range);
Mavg = zeros(1,n^2);

x = [];
for xi = -2:0.2:2
    for xii = -2:0.2:2
        x = [x [xi;xii]];
    end
end
for i = 1:n^2
    for j = 1:n^2
        Mcov1(i,j) = k(x(:,i),x(:,j),M1);
        Mcov2(i,j) = k(x(:,i),x(:,j),M2);
        Mcov3(i,j) = k(x(:,i),x(:,j),M3);
    end
end
Mgp1 = mvnrnd(Mavg,Mcov1,1);
Mgp2 = mvnrnd(Mavg,Mcov2,1);
Mgp3 = mvnrnd(Mavg,Mcov3,1);
Mgp1 = reshape(Mgp1,[n,n]);
Mgp2 = reshape(Mgp2,[n,n]);
Mgp3 = reshape(Mgp3,[n,n]);

figure(1)
subplot(2,2,1)
[X,Y] = meshgrid(range,range);
surf(X,Y,Mgp1,'EdgeColor','interp');
title('M1');

subplot(2,2,3)
surf(X,Y,Mgp2,'EdgeColor','interp');
title('M2');

subplot(2,2,4)
surf(X,Y,Mgp3,'EdgeColor','interp');
title('M3');


%% Marginal likelihood
xo = -4:0.1:4;
yo = sin(3*xo)-2*sin(xo+pi/2); % Real function.
en = 0.5; % standard derivation of error.
x = [-3.5 -2 -1.1 1.9 2.3 3.8];
y = sin(3*x)-2*sin(x+pi/2) + randn(1,6)*en; % Training data.
N = length(x); % number of training data.

n1 = 6; n2 = 30; n3 = 50;  % number of training data.
x1 = x;
x2 = rand(1,n2)*8 - 4; 
x3 = rand(1,n3)*8 - 4;     % training data in range [-4,4]
y1 = y;
y2 = sin(3*x2)-2*sin(x2+pi/2) + randn(1,n2)*en;
y3 = sin(3*x3)-2*sin(x3+pi/2) + randn(1,n3)*en;

figure(2);  %==========================Ground truth and training data
hold on; grid on;
plot(xo,yo,'-');
plot(x,y,'o');
legend('Ground Truth','Training Data(with noise)');
h1=plot(x2,y2,'*');
h2=plot(x3,y3,'d');
pause(1);
delete(h1); delete(h2);

% - Sigma_f
l = 1;
int = 1:1:10;
N_exp = length(int); % number of examples
for n = 1:N_exp
    sigmaf = int(n);
    k = @(x1,x2) sigmaf^2*exp(-1/(2*l^2)*(x1-x2)^2); % covariance function.
    
    for i = 1:n1
        for j = 1:n1
            K1(i,j) = k(x1(i),x1(j));
        end
    end
    for i = 1:n2
        for j = 1:n2
            K2(i,j) = k(x2(i),x2(j));
        end
    end
    for i = 1:n3
        for j = 1:n3
            K3(i,j) = k(x3(i),x3(j));
        end
    end
    Ky1 = K1 + en^2*eye(n1);
    Ky2 = K2 + en^2*eye(n2);
    Ky3 = K3 + en^2*eye(n3);
    
    log_like(n) = -0.5*y3*inv(Ky3)*y3'-0.5*log(det(Ky3))-n3/2*log(2*pi);
    data_fit(n) = -0.5*y3*inv(Ky3)*y3';
    com_penalty(n) = -0.5*log(det(Ky3));
    
    like1(n) = -0.5*y1*inv(Ky1)*y1'-0.5*log(det(Ky1))-n1/2*log(2*pi);
    like2(n) = -0.5*y2*inv(Ky2)*y2'-0.5*log(det(Ky2))-n2/2*log(2*pi);    
end

figure(3)
subplot(1,2,1)
plot(int,log_like,'-','LineWidth',2);
hold on;
plot(int,data_fit,'-.','LineWidth',2);
plot(int,com_penalty,'--','LineWidth',2);
xlabel('amplitude');
ylabel('log-likelihood');
legend('marginal likelihood','data fit','negtive complexity penalty');

subplot(1,2,2)
plot(int,log_like,'-','LineWidth',2);
hold on;
plot(int,like1,'-.','LineWidth',2);
plot(int,like2,'--','LineWidth',2);
xlabel('amplitude');
ylabel('log-likelihood');
legend('n = 50','n = 6','n = 30');

% - lengthscale
int = -1:0.02:1;
N_exp = length(int); % number of examples
sigmaf = 1; % set amplitude
for n = 1:N_exp
    l = 10^(int(n));
    k = @(x1,x2) sigmaf^2*exp(-1/(2*l^2)*(x1-x2)^2); % covariance function.
    
    for i = 1:n1
        for j = 1:n1
            K1(i,j) = k(x1(i),x1(j));
        end
    end
    for i = 1:n2
        for j = 1:n2
            K2(i,j) = k(x2(i),x2(j));
        end
    end
    for i = 1:n3
        for j = 1:n3
            K3(i,j) = k(x3(i),x3(j));
        end
    end
    Ky1 = K1 + en^2*eye(n1);
    Ky2 = K2 + en^2*eye(n2);
    Ky3 = K3 + en^2*eye(n3);
    
    log_like(n) = -0.5*y3*inv(Ky3)*y3'-0.5*log(det(Ky3))-n3/2*log(2*pi);
    data_fit(n) = -0.5*y3*inv(Ky3)*y3';
    com_penalty(n) = -0.5*log(det(Ky3));
    
    like1(n) = -0.5*y1*inv(Ky1)*y1'-0.5*log(det(Ky1))-n1/2*log(2*pi);
    like2(n) = -0.5*y2*inv(Ky2)*y2'-0.5*log(det(Ky2))-n2/2*log(2*pi);    
end

figure(4)
subplot(1,2,1)
plot(int,log_like,'-','LineWidth',2);
hold on;
plot(int,data_fit,'-.','LineWidth',2);
plot(int,com_penalty,'--','LineWidth',2);
xlabel('lengthscale');
ylabel('log-likelihood');
legend('marginal likelihood','data fit','negtive complexity penalty');

subplot(1,2,2)
plot(int,log_like,'-','LineWidth',2);
hold on;
plot(int,like1,'-.','LineWidth',2);
plot(int,like2,'--','LineWidth',2);
xlabel('lengthscale');
ylabel('log-likelihood');
legend('n = 50','n = 8','n = 21');

% - both
intl = -2:0.02:2;
intf = 0.5:0.5:10;
Nl = length(intl);
Nf = length(intf);
for n = 1:Nl
    for m = 1:Nf
        l = 10^(intl(n));
        sigmaf = intf(m);
        k = @(x1,x2) sigmaf^2*exp(-1/(2*l^2)*(x1-x2)^2); % covariance function.
        for i = 1:n3
            for j = 1:n3
                K3(i,j) = k(x3(i),x3(j));
            end
        end
        Ky3 = K3 + en^2*eye(n3);
        like3(n,m) = -0.5*y3*inv(Ky3)*y3'-0.5*log(det(Ky3))-n3/2*log(2*pi);  
    end
end
figure(5)
[X,Y] = meshgrid(intl,intf);
contour(X',Y',like3);
xlabel('index of lengthscale');
ylabel('amplitude');


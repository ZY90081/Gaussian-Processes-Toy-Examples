function output = neg_LOO(D,theta,sigma)

x = D(1,:);   % training input
y = D(2,:);   % training target
N = length(y);
sigmaf = theta(1);
l = 10^(theta(2));
k = @(x1,x2) sigmaf^2*exp(-1/(2*l^2)*(x1-x2)^2);


for i = 1:N
    for j = 1:N
        K(i,j) = k(x(i),x(j));
    end
end
Ky = K + sigma^2*eye(N);

beta = inv(Ky);
alpha = beta*y';

L = 0; % initial L
for m = 1:N
    mu = y(m) - alpha(m)/beta(m,m);
    sig2 = 1/beta(m,m);
    temp = 0.5*log(sig2)+0.5*(y(m)-mu)^2/sig2+0.5*log(2*pi);
    L = L + temp;
end

output = L;

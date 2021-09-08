function output = neg_partiallog_marginal_likelihood(D,theta,sigma)

x = D(1,:);   % training input
y = D(2,:);   % training target
N = length(y);
sigmaf = theta(1);
l = 10^(theta(2));

k = @(x1,x2) sigmaf^2*exp(-1/(2*l^2)*(x1-x2)^2);
pk1 = @(x1,x2) 2*sigmaf*exp(-1/(2*l^2)*(x1-x2)^2);
pk2 = @(x1,x2) sigmaf^2*(x1-x2)^2/l^3*exp(-1/(2*l^2)*(x1-x2)^2);

for i = 1:N
    for j = 1:N
        K(i,j) = k(x(i),x(j));
        PK1(i,j) = pk1(x(i),x(j));
        PK2(i,j) = pk2(x(i),x(j));
    end
end
Ky = K + sigma^2*eye(N);
PKy1 = PK1 + sigma^2*eye(N);
PKy2 = PK2 + sigma^2*eye(N);

alpha = inv(Ky)*y';

output(1) = -0.5*trace((alpha*alpha'-inv(Ky))*PKy1);
output(2) = -0.5*trace((alpha*alpha'-inv(Ky))*PKy2);


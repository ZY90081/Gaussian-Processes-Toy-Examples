function output = neglog_marginal_likelihood(D,theta,sigma)

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
output = 0.5*y*inv(Ky)*y' + 0.5*log(det(Ky)) + 0.5*N*log(2*pi);

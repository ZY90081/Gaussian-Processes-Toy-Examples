function output = neg_partial_LOO(D,theta,sigma)

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

beta = inv(Ky);
alpha = beta*y';
Z1 = beta*PK1;
Z2 = beta*PK2;

output = zeros(1,2);
for m = 1:N
    
    S = Z1*beta;
    h = Z1*alpha;
    temp1 = (0.5*(1+alpha(m)^2/beta(m,m))*S(m,m) - alpha(m)*h(m))/beta(m,m);
    S = Z2*beta;
    h = Z2*alpha;    
    temp2 = (0.5*(1+alpha(m)^2/beta(m,m))*S(m,m) - alpha(m)*h(m))/beta(m,m);
    
    output(1) = output(1) + temp1;
    output(2) = output(2) + temp2;
end
end


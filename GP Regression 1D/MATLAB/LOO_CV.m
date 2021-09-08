function x = LOO_CV(D,sigma)

% Input: D - training data
% Output: x - hyperparameters

GAMMA = 0.001;    % step size (learning rate)
MAX_ITER = 1000;  % maximum number of iterations
FUNC_TOL = 0.1;   % termination tolerance for F(x)


fvals = [];       % store F(x) values across iterations
progress = @(iter,x) fprintf('iter = %3d: x = %-10s, neg_log_likelihood = %f\n', ...
    iter, mat2str(x,6), fvals(iter));

% Iterate
iter = 1;         % iterations counter
x = [1,0];    % initial guess
fvals(iter) = neg_LOO(D,x,sigma);
fprintf('iter = %3d: x = %-10s, neg_log_likelihood = %f\n', ...
    iter, mat2str(x,6), fvals(iter));
while iter < MAX_ITER && fvals(end) > FUNC_TOL
    iter = iter + 1;
    x = x - GAMMA * neg_partial_LOO(D,x,sigma);  % gradient descent
    fvals(iter) = neg_LOO(D,x,sigma);     % evaluate objective function
   fprintf('iter = %3d: x = %-10s, neg_log_likelihood = %f\n', ...
    iter, mat2str(x,6), fvals(iter));      % show progress
end


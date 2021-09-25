function [x, FFval]= Fun_mini_gredd(Fun,J,Initial,MaxIteration,Termination,Step)

% minimize a differentiable multivariate function by Newton’s method with Levenberg-Marquardt modification
% fixed step size
% fixed modification number
% inputs:   Fun - target function
%           J, H - first and second derivates of function
%           Initial - start point
%           MaxIteration - maximum number of iterations
%           Termination - termination tolerance for function

GAMMA = Step;    % step size 
%MU = 0.001;  % modification

Dim = length(Initial); % dimension of states.

fvals = [];       % store F(x) values across iterations
%progress = @(iter,x) fprintf('iter = %3d: x = %-10s, fun = %f\n', ...
%    iter, mat2str(x,6), fvals(iter));

% Iterate
iter = 1;         % iterations counter
x = Initial;      % initial guess
fvals(iter) = Fun(x);
diff(:,iter) = ones(Dim,1);
%fprintf('iter = %3d: x = %-10s, fun = %f\n', ...
%    iter, mat2str(x,6), fvals(iter));

while iter <= MaxIteration && norm(diff(:,end)) > Termination
     iter = iter + 1;
%     e = eig(H(x)); 
%     if any(e<0)
%         MU = -min(e)+0.001;
%     else
%         MU = 0;
%     end
     diff(:,iter) = GAMMA*J(x);
    x = x - diff(:,iter);
    fvals(iter) = Fun(x);     % evaluate objective function
   %fprintf('iter = %3d: x = %-10s, neg_log_likelihood = %f\n', ...
    %iter, mat2str(x,6), fvals(iter));      % show progress
end

FFval = fvals(iter);
end

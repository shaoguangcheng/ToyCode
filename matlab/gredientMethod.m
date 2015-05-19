% This code can be use to compute optimal value of objective function with multi-variables.
% This code is constructed by using gradient method.

% N : total number of variables
% F : objective function you want to calculate optimal value.
%     note : the variables you can use in F must be like the form
%     x1,x2,x3...For example, F = 'x1*x2+x3^2+x4^4'

% xInit : the initial value of x1,x2,...
% sigma : termination condition of iteration process

% xResult : the optimal value of variable
% object  : the optimal value of objective function

% note : you just need to modify N, F,X and sigma to use this code.

% If you have any problem with this code, please contact me directly.
% E-mail : chengshaoguang@mail.nwpu.edu.cn
% My ID  : 2013200094
% Tel : 15934859714
% Write by ChengShaoguang at Northwest Polythechnical University. 

% You just need to modify N, F, xInit, sigma to solve your own problems.
N = 2;
x = sym('x',[N,1]);
F = 'x1^2-2*x1*x2+4*x2^2+x1-3*x2';
xInit = [1,1]';
sigma  = 1e-1;

% You do not need to modify the following part in this code.
if length(xInit) ~= N
    disp 'error : the size of initial value must be equal to the number of variable\n';
end

gradF = jacobian(F,x); % gradient of F
hessenF = jacobian(gradF,x); % hessen matrix of F
alpha = simplify((gradF * gradF')/(gradF * hessenF * gradF')); % compute step size

xPrevious = xInit;
gradFNum = subs(gradF,x(1:end),xPrevious);

iter = 0;
object = 0;
fprintf('starting computing : \n');
while norm(gradFNum,2) >= sigma   % termination condition
    alphaNum  = subs(alpha,x(1:end),xPrevious);
    xPrevious = xPrevious - alphaNum * gradFNum'; % iteration
    gradFNum  = subs(gradF,x(1:end),xPrevious);norm(gradFNum,2);
    
    object = subs(F,x(1:end),xPrevious); % function value of each iteration
    iter = iter + 1;
    fprintf('\niter = %d, F = %f, ', iter,object);
    fprintf('x = [');fprintf('%f ',xPrevious);fprintf(']');
end

if isnan(object) || isinf(object)
    fprintf('\nthe optimal point does not exist \n');
else
    xResult = xPrevious; % optimal x
    fprintf('\n\ntotal iter = %d, optimal F = %f, ',iter,object);
    fprintf('optimal x = [');fprintf('%f ',xPrevious);fprintf(']\n');
end

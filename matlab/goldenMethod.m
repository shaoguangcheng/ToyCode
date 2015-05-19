% This code can be use to compute optimal value of objective function with single variable in specified
% range that only has single extremum.
% This code is constructed by using Golden Section method.

% F : objective function you want to calculate optimal value.

% initRange : the initial range  that only has single extremum.
% sigma : termination condition of iteration process

% xResult : the optimal value of variable
% object  : the optimal value of objective function

% note : you just need to modify F,X and sigma to use this code.

% If you have any problem with this code, please contact me directly.
% E-mail : chengshaoguang@mail.nwpu.edu.cn
% My ID  : 2013200094
% Tel : 15934859714
% Write by ChengShaoguang at Northwest Polytechnical University. 

% You just need to modify F, initRange, sigma to solve your own problems.
F = 'x^3-3*x+1';
initRange = [0,2]; %the intial range  that only has single extremum.
sigma = 1e-3;

% You do not need to modify the following part in this code.
goldenRatio = 0.618;
fun = sym(F);
lower = initRange(1); % lower boundary
upper = initRange(2); % upper boundary

iter = 0;
object = 0;
while abs(upper - lower) >= sigma
    lamda = upper - goldenRatio*(upper - lower); % lower split point
    miu   = lower + goldenRatio*(upper - lower); % upper split point
    
    f1 = subs(F,'x',lamda);
    f2 = subs(F,'x',miu);
    object = subs(F,'x',(lamda+miu)/2); % function value of each iteration
    
    if f1 >= f2  % update range length
        lower = lamda;
    else
        upper = miu;
    end
    
    iter = iter + 1;
    fprintf('iter = %d, lower = %f, upper = %f, F = %f \n',iter, lower,lamda,object);
end

if isnan(object) || isinf(object)
    fprintf('\nthe optimal point does not exist \n');
else
    xResult = (lamda+miu)/2;
    fprintf('\ntotal iter = %d, optimal x = %f, optimal F = %f \n',iter, (lamda+miu)/2,object);
end
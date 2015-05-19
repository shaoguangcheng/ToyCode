% This code can be use to compute optimal value of objective function with single variable.
% You should specify the two initial points to iterate.

% This code is constructed by using Three Times Fit method.

% F : objective function you want to calculate optimal value
% initX : the two initial points that you must set in advance in order to iterate.
% sigma : termination condition of iteration process

% xResult : the optimal value of variable
% object  : the optimal value of objective function

% note : you just need to modify F,initX and sigma to use this code.

% If you have any problem with this code, please contact me directly.
% E-mail : chengshaoguang@mail.nwpu.edu.cn
% My ID  : 2013200094
% Tel : 15934859714
% Write by ChengShaoguang at Northwest Polytechnical University. 

% You just need to modify F, initX, sigma to solve your own problems.
F = 'x^3-3*x+1'; % object function
initX = [0,2]; % the two initial points to iterate. 
sigma = 1e-1;  % termination condition

% You do not need to modify the following code when you using this code.
fun = sym(F);
dF = diff(fun,'x');
x1 = initX(1);
x2 = initX(2);

iter = 0;
object = 0;
while abs(x2-x1) >= sigma
    dF_1 = subs(dF,'x',x1);
    dF_2 = subs(dF,'x',x2);
    F_1  = subs(F,'x',x1);
    F_2  = subs(F,'x',x2);
    
    U1 = dF_1 + dF_2 - 3*(F_1 - F_2)/(x1-x2);
    U2 = sqrt(U1^2 - dF_1*dF_2);
    
    p = dF_1 - dF_2+2*U2;
    if p == 0
        p = 0.1;
    end
    x3 = x2-(x2-x1)*(dF_1 + U2 -U1)/p;
    x1 = x2;
    x2 =x3;
    object = subs(F, 'x', x3);
    
    iter = iter +1;
    fprintf('iter = %d, x = %f, F = %f \n',iter,x3,object);
end

if isnan(object) || isinf(object)
    fprintf('\nthe optimal point does not exist \n');
else
    xResult = middle;
    fprintf('\ntotal iter = %d, optimal x = %f, optimal F = %f \n',iter,x3,object);
end
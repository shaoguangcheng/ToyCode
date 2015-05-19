% This code can be use to compute optimal value of objective function with single variable.
% You should sprecify the three initial points to iterate.

% This code is constructed by using Two Times Fit method.

% F : objective function you want to calculate optimal value
% initX : the three intial points that you must set in advance in order to iterate.
%        note : initX(1) < initX(2) < initX(3); 
% sigma : termination condition of iteration process

% xResult : the optimal value of varaible
% object  : the optimal value of objective function

% note : you just need to modify F,X and sigma to use this code.

% If you have any problem with this code, please contact me directly.
% E-mail : chengshaoguang@mail.nwpu.edu.cn
% My ID  : 2013200094
% Tel : 15934859714
% Write by ChengShaoguang at Northwest Ploytechnical University. 

% You just need to modify F, initX, sigma to solve your own problems.
F = 'x^3';
initX = [0,1,15]; % the three intial point to iterate. 
                 % note : initX(1) < initX(2) < initX(3); 
sigma = 1e-3;

% You do not need to modify the following part in this code.
lower = initX(1);
middle = initX(2);
upper = initX(3);

FNum = 1e10;
FMiddle = 0;
iter = 0;
fprintf('starting computing : \n');
while abs(FNum-FMiddle) >= sigma  % termination condition 
    FLower = subs(F,'x',lower);
    FMiddle = subs(F,'x',middle);
    FUpper  = subs(F,'x',upper);
    
    xNew = 0.5*((middle^2-upper^2)*FLower + (upper^2 - lower^2)*FMiddle + (lower^2-middle^2)*FUpper)/...
           ((middle-upper)*FLower + (upper - lower)*FMiddle + (lower-middle)*FUpper); % new point to insert (iteration process)
    FNum = subs(F,'x',xNew);
        
    if xNew < middle % update the three points
        if FNum < FMiddle
            middle = xNew;
            upper = middle;
        else
            lower = middle;
            middle = xNew;
        end
    else 
        if FNum < FMiddle
            lower = middle;
            middle = xNew;
        else
            middle = xNew;
            upper = middle;
        end
    end
    
    iter = iter+1;
    fprintf('iter = %d, x = %f, F = %f \n',iter,middle,FNum);
end

if isnan(FNum) || isinf(FNum)
    fprintf('\nthe optimal point does not exist \n');
else
    xResult = middle;
    fprintf('\ntotal iter = %d, optimal x = %f, optimal F = %f \n',iter,middle,FNum);
end
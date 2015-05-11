function [x] = conjugateGradient(func, x0)
eplison = 1e-5;
maxIter = 100;
k = 0;
x = x0;

A = [1,2,3;
    2,6,6;
    3,6,8];

while k < maxIter 
    x
    gNew = func(x);
    if norm(gNew) < eplison
        break
    end
    
    if k == 0
       dNew = -gNew;
    else
       beta = gNew'*gNew/(gOld'*gOld);
       dNew = -gNew + beta*dOld;
    end
    
    alpha = -1*(gNew'*dNew)/(dNew'*A'*dNew);
    x = x + alpha*dNew;
    
    dOld = dNew;
    gOld = gNew;
    
    k = k + 1;
end




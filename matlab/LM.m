function [x, residual] = LM(rFunc, JFunc, x0)
eplison = 1e-9;
kmax = 1000;
k = 0;

rOld = rFunc(x0);
lambda = norm(rOld);
%lambda = 2;

n = length(x0);
while k < kmax
    J = JFunc(x0);
    H = J'*J;
    G = J'*rOld;
    
    if norm(G) < eplison
        break;
    end

    x = x0 - (H + lambda*eye(n))\G;

    rNew = rFunc(x);
    lambda = norm(rNew);
    x0 = x;
    rOld = rNew;
    
    k = k+1;
end

    x = x0;
    residual = norm(rOld);
end


function [ y ] = expMAP(lambda, x)
    size = length(x);
    tmp = -lambda*sum(x);
    y = lambda^size*exp(tmp);
end


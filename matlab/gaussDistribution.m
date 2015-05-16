function [ y ] = gaussDistribution(x, miu, delta)
c = 1/(sqrt(2*pi)*delta);
y = c * exp(-(x-miu)^2/(2*delta^2));
end


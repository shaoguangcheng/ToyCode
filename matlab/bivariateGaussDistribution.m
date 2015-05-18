function [y] = bivariateGaussDistribution(x, miu, delta)
% D-dimension gauss dist
% N(x|miu, delta) = 1/(2*pi)^(D/2)*1/sqrt(det(delta)) *exp(-0.5*(x-u)'*inv(delta)*(x-u))
    detDelta = delta(1)*delta(4)-delta(2)*delta(3);
    c = 1/(2*pi*sqrt(detDelta));
    y = c*exp(-0.5*(x-miu)'/delta*(x-miu));
end


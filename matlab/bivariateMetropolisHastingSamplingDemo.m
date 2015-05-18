% A demo to show bivariate Metropolis Hasting samplings
% For simplicity, I just use uniform distribution as proposal distribution
% In this demo, I sample p(x, y) ~ N(miu, delta), miu = (1, 1), delta = [1,
% 0.7;0.7, 1];

N = 10000;
k = 1;
x = zeros(2, N);

upperBound = 4;
lowerBound = -2; 

x(1, k) = rand()*(upperBound - lowerBound) + lowerBound; % init x(1, :). For two dimisions
x(2, k) = rand()*(upperBound - lowerBound) + lowerBound;

miu = [1;1];
sigma = [1, 0.7;
        0.7, 1];

while k < N
    k = k + 1;
    theta1 = rand()*(upperBound - lowerBound) + lowerBound; % sample the first variable from proposal dist q(x1|x1(k-1))
    tmp = [theta1; x(2, k-1)]; 
    alpha  = min(1, bivariateGaussDistribution(tmp, miu, sigma)/bivariateGaussDistribution(x(:, k-1), miu, sigma)); % cal accept ratio. alpha = min(1, p(x1, x2(k-1))/(p(x1(k-1), x2(k-1)))* q(x1(k-1)|x1)/q(x1|x1(k-1)))
    u = rand();
    
    if(u < alpha)
        x(1, k) = theta1;
    else
        x(1, k) = x(1, k-1);
    end
    
    theta2 = rand()*(upperBound - lowerBound) + lowerBound; % sample the second variable from proposal dist dist q(x2|x2(k-1))
    tmp = [x(1, k); theta2];
    tmp2 = [x(1, k); x(1, k-1)];
    alpha  = min(1, bivariateGaussDistribution(tmp, miu, sigma)/bivariateGaussDistribution(tmp2, miu, sigma)); % cal accept ratio alpha = min(1, p(x1, x2)/(p(x1, x2(k-1)))* q(x2(k-1)|x2)/q(x2|x2(k-1)))
    u = rand();
    
    if u < alpha
        x(2, k) = theta2;
    else
        x(2, k) = x(2, k-1);
    end
end

disp 'mean error:'; (miu   - mean(x, 2)).^2
disp 'cov error :'; (sigma - cov(x')).^2



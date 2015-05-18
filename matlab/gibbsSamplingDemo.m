% A demo to show gibbs samplings
% In this demo, I sample p(x, y) ~ N(miu, delta), miu = (1, 1), delta = [1,
% rou;rou, 1];

% p(x|y) = N(miu(1)+rou*(y - miu(2)), sqrt(1-rou^2))
% we need to sample from p(x|y) and p(y|x)

% This demo shows that Gibbs sampling is much better than MH sampling (just in this situation)
% Because I choose uniform dist as proposal dist for simplicity

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
    
rou = 0.7;
sigma_ = sqrt(1-rou^2);
while k < N
   k = k + 1;
   
   miu1 = miu(1) + rou*(x(2, k-1) - miu(2)); 
   x(1, k) = randn()*sigma_ + miu1; % sample the first variate
   
   miu2 = miu(2) + rou*(x(1, k) - miu(1));
   x(2, k) = randn()*sigma_ + miu2; % sample the second variate
end

disp 'mean error:'; (miu   - mean(x, 2)).^2
disp 'cov error :'; (sigma - cov(x')).^2


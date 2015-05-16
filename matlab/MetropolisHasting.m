% A demo to show Metropolis Hasting samplings
% For simplicity, I just use uniform distribution as proposal distribution
% In this demo, I sample p(x) ~ N(1,1)

N = 10000;
k = 1;
x = zeros(1, N);

upperBound = 4;
lowerBound = -2;

x(1) = rand() * (upperBound - lowerBound) + lowerBound; % init starting state

while k < N
   k = k +1;
   y = rand() * (upperBound - lowerBound) + lowerBound; % sample a canidate state from proposal dist q(x|x(k-1)). Here I use uniform dist q(x) = 1/(upperBound - lowerBound)
   alpha = min(1, gaussDistribution(y, 1, 1)/gaussDistribution(x(k-1), 1, 1)); % cal accept ratio. alpha = min(1, p(x)/(p(x(k-1)))* q(x(k-1)|x)/q(x|x(k-1)))
   u = rand(); 
   if u < alpha % accept the transfer if u is less than accept ratio 
       x(k) = y;
   else
       x(k) = x(k-1);
   end
end

fprintf('mean error : %f \n', (1-mean(x))^2);
fprintf('var  error : %f \n', (1-var(x))^2);
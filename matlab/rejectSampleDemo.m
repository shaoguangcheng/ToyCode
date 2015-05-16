% A demo to show rejection sample
% For simplicity, I just use uniform distribution as proposal distribution
% In this demo, I sample p(x) ~ N(1,1)

n = 10000; % number points to sample
x = zeros(1, n); % sampled points
k = 1;
c = 1/sqrt(2*pi)*6; % choose c that satisfies c*q(x) >= p(x). Here I choose q(x) = 1/6 that x in [-2, 4]. (1-3*1, 1+3*1)

while k <= n
   theta = rand()*6-2; % sample from q(x)
   u = rand()*c*1/6;   % sample from [0, c*q(x)]
   pTheta = 1/6*c*exp(-(theta-1)^2/2); % cal p(x)
   
   if(u <= pTheta)     % accept sample point
       x(k) = theta;
       k = k+1;
   end
end

fprintf('mean error : %f \n', (1-mean(x))^2);
fprintf('var  error : %f \n', (1-var(x))^2);


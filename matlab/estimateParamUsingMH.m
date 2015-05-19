% A demo to show hoe to use sampling methods to estimate the probability 
% distrbutions' params.

% In this demo, some sample are generated from p(x|lambda) = lambda*exp(-lambda*x)
% My goal is to estimate lambda by using these samples (This problem is easy to
% solve with Maximum Likelyhood, but I just want to show another method)

% Useful when the sample size is small

lambda = 2;
e = 1/lambda;

N = 10;
x = exprnd(e, [1, N]); % generate some samples from exp dist
sumX = sum(x);

upperBound = 5;
lowerBound = 0;

t = 1;
T = 10000;
estimatedLambda = zeros(1, T);
estimatedLambda(t) = 1;

while t < T
   t = t + 1;
   theta = rand()*(upperBound - lowerBound) + lowerBound;
   alpha = min(1, expMAP(theta, x)/expMAP(estimatedLambda(t-1), x));
   u = rand();
   if u <= alpha
       estimatedLambda(t) = theta;
   else
       estimatedLambda(t) = estimatedLambda(t-1);
   end
end

% because of the burn-in process, we just use last 100 estimated lambda

fprintf('estimation error: %f\n', (lambda-mean(estimatedLambda(5000:end)))^2);
fprintf('ML estimation error: %f\n', (lambda-N/sumX)^2);
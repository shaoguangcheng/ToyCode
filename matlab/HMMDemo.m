% A simple demo to show discrete HMM

N = 2; % number of state
M = 2; % number of observation
state = [1, 2]; % state space
ob    = [1, 2];

a = 0.1;
b = 0.8;
A = [1-a, a;
    a, 1-a];
B = [b, 1-b;
    1-b, b];

K = 100;
k = 1;

x = zeros(1, K); % state sequence
y = zeros(1, K); % ob sequence

u = rand(); % init state
if u > 0.5
    x(k) = 1;
else
    x(k) = 2; 
end

u = rand();
if x(k) == 1
    if u <= b
        y(k) = 1;
    else
        y(k) = 2;
    end
else
    if u <= 1-b
        y(k) = 1;
    else
        y(k) = 2;
    end
end

while k < K
   k = k + 1;
   u = rand();
   
   if x(k-1) == 1 % cal the state the time k
       if u <= 1-a
           x(k) = 1;
       else
           x(k) = 2;
       end
   else
       if u <= a
           x(k) = 1;
       else
           x(k) = 2;
       end
   end
   
   u = rand();
    if x(k) == 1
        if u <= b
            y(k) = 1;
        else
            y(k) = 2;
        end
    else
        if u <= 1-b
            y(k) = 1;
        else
            y(k) = 2;
        end
    end   
end

figure(1);
plot(1:K, x);
axis([1, K, 0, 3]);
title('real state');
figure(2);
plot(1:K, y, 'ro');
axis([1, K, 0, 3]);
title('observation');

[P, S] = HMMViterbi([0.5, 0.5], A, B, y);

figure(3);
plot(1:K, S);
axis([1, K, 0, 3]);
title('estimated state');

correctRatio = length(find(S-x == 0))/K;
fprintf('Correct ratio : %f\n', correctRatio);
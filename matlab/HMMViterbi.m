function [ P, S ] = HMMViterbi(p0, A, B, O)
% Viterbi algorithm to compute the most likely latent state sequence that
% generates obeservations

% p0 : the prior probability distribution for state
% A  : state transfer matrix
% B  : obeservation matrix
% O  : obeservation sequence

N = size(A, 1); % number of state
T = length(O);  % length of observation sequence
t = 1;

% Here can be optimized. (using less memory)
X = zeros(T, N); % record the maximum prob of state i at time t
Y = zeros(T, N); % record the previous state

S = zeros(1, T);

for i = 1 : N
   X(t, i) = p0(i)*B(i, O(t));
end

while t < T
    t = t + 1;
    
    for j = 1 : N
       index = 0;
       for i = 1 : N
          tmp = X(t-1, i) * A(i, j);
          if X(t, j) < tmp
             X(t, j) = tmp;
             index = i;
          end
       end
       
       X(t, j) = X(t, j) * B(j, O(t));
       Y(t, j) = index;       
    end
end

[P, index] = max(X(T, :));
S(T) = index;
t = T;

while t > 1
   t = t - 1;
   S(t) = Y(t+1, S(t+1));
end

end


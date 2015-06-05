% A demo to show svd decomposition and PCA using svd

mu = [0, 2];
sigma = [1, 0.8;
        0.8, 1];  
N = 100;
    
x = mvnrnd(mu, sigma, N);

x(:, 1) = x(:, 1)/var(x(:,1)); 
x(:, 2) = (x(:,2)-mean(x(:,2)))/var(x(:,2));

figure(1);
plot(x(:,1), x(:,2), '.');

X = x'*x/N;
[v, d] = eig(X);

hold on;

y = x*v;
plot(y(:,1), y(:,2), 'r.');

hold on;

z = x*v(:,2);
plot(z, zeros(1,N), 'g.');

[U, S, V] = svd(x);




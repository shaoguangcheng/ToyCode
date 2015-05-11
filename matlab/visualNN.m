% to show NN can approximate any functions(single input)
W = [100, 100; -0.5, 0.5];
b = [-40,  40; 0, 0];

x = -1:0.01:1;
y = zeros(size(x));

N = size(x,2);

for i = 1:N
   y1 = sigmoid(x(i)*W(1,1)+b(1,1));
   y2 = sigmoid(x(i)*W(1,2)+b(1,2));
   y(i) = y1*W(2,1)+y2*W(2,2)+b(2,1);
%   y(i)  = sigmoid(y1*W(2,1)+y2*W(2,2)+b(2,1));
end

plot(x,y);
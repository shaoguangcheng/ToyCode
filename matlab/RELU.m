function [ y ] = RELU( x )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    y = zeros(size(x));
    index = x > 0;
    y(index) = x(index);
end


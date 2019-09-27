function [out] = dsigmoid(in)
%DSIGMOID Summary of this function goes here
%   Detailed explanation goes here
out = sigmoid(in).*(1-sigmoid(in));
end


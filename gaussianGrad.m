function g = gaussianGrad(z)
% Compute gaussian function
g = -2 .* z .* exp(-z.*z);
end
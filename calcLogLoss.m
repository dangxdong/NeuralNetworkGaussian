function [cost] = calcLogLoss(pred, y)
m = size(y, 1);
prednew = max(min(pred, (1 - 10^(-15))), 10^(-15));
cost = -y .* log(prednew) - (1-y) .* log(1 - prednew) ;
cost = 1 / m * sum(cost);
cost = sum(cost);
end
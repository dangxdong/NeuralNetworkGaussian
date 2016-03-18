function W = randInitGradGaussian(l1, l2)
% randomly initializes the weights of a layer with l1 incoming
% and l2 outgoing. 
epsilon = sqrt(6)/sqrt(l2+l1); 
W1 = rand(round(l2/2), 2 + l1) * 2 * epsilon - epsilon - 1;
W2 = rand(l2-round(l2/2), l1 + 2) * 2 * epsilon - epsilon + 1;
W = [W1;W2];
end

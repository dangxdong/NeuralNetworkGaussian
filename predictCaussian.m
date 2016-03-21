function [p pn]= predictCaussian(Theta1, Theta2, X)

% Returns the predicted count of y given the
% trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

% input to hidden layer, logistic regression
z = [ones(m, 1) X ones(m, 1)] * Theta1';
h1 = gaussian(z);
h1 = [ones(m, 1) h1 ones(m, 1)];
% hidden layer to output: Poisson regression
z2 = Theta2 * h1';
h2 = gaussian(z2);
% just return the h2 values. 
p = h2';
pn = round(p);

end

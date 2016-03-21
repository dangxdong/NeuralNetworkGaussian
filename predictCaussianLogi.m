function [p pn]= predictCaussianLogi(Theta1, Theta2, X)

% Returns the predicted count of y given the
% trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);

% input to hidden layer, gaussian regression
x1 = [ones(m, 1) X ones(m, 1)];
z =  Theta1 * x1';
h1 = gaussian(z);
h1 = [ones(1, m); h1;ones(1, m)];
% hidden layer to output: logistic regression
z2 = Theta2 * h1;
h2 = sigmoid(z2);
% just return the h2 values. 
p = h2';
pn = round(p);

end

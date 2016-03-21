# Neural Network Gaussian
Here is a prototype of ANN using Gaussian instead of Logistic as the activation function.

This is one of the possible implementations of the method of RBF (Radial basis function) network.

The cost function, nnCostFunctionGaussian.m, is the kernel of this model, which takes the training data and relevant parameters as input and returns the cost (logarithmic loss with L2 regularization) and gradient (calculated by backpropagation) as output.

A two-hidden-layer cost function, nnCostFuncGaussian2hl.m, is also provided here, but may need further review.

Different from usual ANN models, two (rather than one) "intercept units" are used with the input and the hidden layers. Also the generation of initial model coefficients (the Theta values, or say, the weights) is also modified to generated values beside -1 and 1, rather than around 0. These are taken as a strategy to keep the symmetry of the Gaussian distribution for the next layer.

Use nnGaussian.m as the entry point of modeling, by passing the trainig predictor matrix, training label vector (or matrix) and a hidden layer size to it. It will return two coefficient matrices, theta1 and theta2. You can call them w1 and w2 or anything you are used to.

Then use predictCaussian.m to predict, by passing theta1, theta2 and the new data set containing the predictors. It will return the predicted values in the form of decimals between 0 and 1.

Then the function calcLogLoss.m can be used, to measure the logarithmic loss with the predicted values and true label values.

So on top of nnGaussian.m, predictCaussian.m and calcLogLoss.m, you can investigate and test different parameters like the hidden layer size and the L2 regularization coefficient (lambda).

Beside the model with Gaussian function both between the input and hidden layers and between the hidden and output layers, a hybrid model is also provided, with Gaussian regression between the input and the hidden layers, and with logistic regression between the hidden and the output layers. The files nnGaussianLogistic.m, predictCaussianLogi.m and nnCostFuncGaussianLogistic.m are provided for this purpose.

Special note: you will need to have fmincg.m (Carl Edward Rasmussen) in your workspace directory, as the optimization function for the model. The fminunc.m or other optimization functions available to Octave/Matlab may also be used.


# NeuralNetworkGaussian
Here is a prototype of ANN using Gaussian instead of Logistic as the activation function.

This is to implement the method of RBF (Radial basis function) network.

The cost functions are provided here to be included in neural network models.

The single-hidden-layer cost function is ready to used in combination with the other functions here.

The two-hidden-layer cost function will be modified later.

Two (rather than one) "intercept" units are used in the input and the hidden layer, to keep the symmetry of the Gaussian distribution for the next layer.

Use nnGaussian.m for modeling, by passing the trainig predictor matrix, training label vector (or matrix) and a hidden layer size to it. It will return two coefficient matrices, theta1 and theta2. You can call them w1 and w2 or anything you like.

Then use predictCaussian.m to predict, by passing theta1, theta2 and the new data set containing the predictors. It will return predicted values in the form of decimals between 0 and 1.

Then the function calcLogLoss.m can be used, to measure the log loss with the predicted values and true label values.

So on top of nnGaussian.m and predictCaussian.m, you can investigate and test different parameters like the L2 regularization coefficient, here refered to by lambda.


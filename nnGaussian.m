function [Theta1, Theta2, minicost] = nnGaussian(X, y, ...
                           hidden_layer_size = 0, lambda = 0, maxIter = 1000)

input_layer_size  = size(X,2);
num_labels = size(y,2); 

if (hidden_layer_size == 0) 
    hidden_layer_size = round((input_layer_size+num_labels)/2);
end

initial_Theta1 = randInitGradGaussian(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitGradGaussian(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  Train the model.
options = optimset('MaxIter', maxIter);
costFunction = @(p) nnCostFunctionGaussian(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 2)), ...
                 hidden_layer_size, (input_layer_size + 2));

Theta2 = reshape(nn_params((1 + hidden_layer_size * (input_layer_size + 2)):end), ...
                 num_labels, (hidden_layer_size + 2));

end

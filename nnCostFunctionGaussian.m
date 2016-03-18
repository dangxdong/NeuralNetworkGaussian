function [J2 grad] = nnCostFunctionGaussian(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Adapted for Gaussian activation. 

% First, reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 2)), ...
                 hidden_layer_size, (input_layer_size + 2));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 2))):end), ...
                 num_labels, (hidden_layer_size + 2));

% Setup some useful variables
m = size(X, 1);
         
% need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% begin to calculate J and gradients:

X1 = [ones(m,1) X ones(m,1)]; % so X1 is m * (n+2)

ZZ1 = Theta1 * X1'; % so ZZ1 is n(L2)*m dimensional

% Gaussian function:
AA1 = gaussian(ZZ1); % so AA1 is the predicted middle layer unit values

AA1 = [ones(1, m); AA1; ones(1, m)]; % Now AA1 is(n(L2)+2)*m

% As Theta2 is n(L3) * (n(L2)+1)
ZZ2 = Theta2 * AA1; % ZZ2 is n(L3)*m = k*m
% Gaussian function:
AA2 = gaussian(ZZ2);

hh= AA2'; % by transposing, we get predicted hh as m * n(L3)=m*k

% Here the y is already m*k, so just use y to do the calculation
% And we know that hh is the predicted yy, so all the cost is based on hh and yy.

% because both y and hh are m*k, use dot() rather than the matrix multiplication,
term1 = -1 * dot(y, log(hh));
term2 = dot((y - 1), log(1-hh));  % Both term1 and term2 are 1*k dimensional

J = 1 / m * (term1 + term2);   % now J is also 1*k dimensional, and not yet regularized

J1 = sum(J);  % now J is a scalar, not yet regularized

Temp1 = Theta1;
Temp1(:, 1) = 0;
Temp1(:, end) = 0;
Temp2 = Theta2;
Temp2(:, 1) = 0;
Temp2(:, end) = 0;
% So Temp1 and Temp2 are modified Theta1 and Theta2, with first columns all zeros
% The regularization term can thus be written as:
termR = 0.5 * lambda / m * (sum(dot(Temp1,Temp1))+sum(dot(Temp2,Temp2)));

J2 = J1 + termR; % Now J is ready.

%
% Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 


% To make it easier to see, y is transposed so each column is a sample record, 
% so we use AA2 instead of hh (remember hh=AA2'), and y' instead of y, both k*m

delta3 = AA2 - y';  
% Why don't we do sigterm3 = sigmoidGradient(ZZ2); 
% and then delta3 = delta3 .* sigterm3;  ??????  This is a mathematical question.

% This line is special for Gaussian function:
delta3 = 2 * delta3 .* ZZ2 ./ (AA2 - 1);

% use AA2 and AA1 instead of the variable names A3 and A2
% so delta3 is k*m

% Theta2 is k * (n(L2)+1), in this case is 10*26

% !!== need to change the codes below to vectorize

delta2 = Theta2' * delta3;  
% delta3 is changed from k*1 to k*m. The above delta2 is thus 
% changed from (n(L2)+1)*1 to (n(L2)+1)*m.

delta2 = delta2(2:end-1,:);      % we only need delta2 n(L2)*m to clculate Theta1

% use ZZ1 to calculate the sigmoidgradient term; ZZ1 is n(L2)*m dimensional
% instead of only the t-th column of ZZ1
% Gaussian gradient!!!
sigterm2 = gaussianGrad(ZZ1);
    
% so delta2 and sigterm2 are elementwisely multiplicable:
delta2 = delta2 .* sigterm2;   % so we get the delta2 we want    

% Remember that:
% Theta1_grad = zeros(size(Theta1));   n(L2) * n+1
% Theta2_grad = zeros(size(Theta2));   k * (n(L2)+1)

% Use the AA1 values in this t-th column, which is (n(L2)+1) * 1

% as above, delta3 is k*m, AA1' is m *(n(L2)+1), delta3 * AA1' is k*(n(L2)+1);
% !!! Note here delta3 * AA1' == delta3(:,1)*AA1'(1,:)+delta3(:,2)*AA1'(2,:)+...
Theta2_grad = Theta2_grad + delta3 * AA1';
% likewise, delta2 n(L2)*m, X1 is m*n+1, they can be multiplied to get Theta1_grad
Theta1_grad = Theta1_grad + delta2 * X1;

% Remember the Temp1 and Temp2 are modified Theta1 and Theta2, with first columns all zeros
% so we can just use them for regularization
Theta1_grad = 1 / m .* (Theta1_grad + lambda .* Temp1);
Theta2_grad = 1 / m .* (Theta2_grad + lambda .* Temp2);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

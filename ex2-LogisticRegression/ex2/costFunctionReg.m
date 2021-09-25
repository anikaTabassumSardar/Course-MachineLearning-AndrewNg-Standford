function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

  %DIMENSIONS: 
  %   theta = (n+1) x 1
  %   X     = m x (n+1)
  %   y     = m x 1
  %   grad  = (n+1) x 1
  %   J     = Scalar
  
 z = X * theta;
 h_x = sigmoid(z); % m x 1

 %should not be regulaizing theta_0 hence the below
reg_term = (lambda/(2*m)) * sum((theta(2:end)) .^2);

[J_withoutReg, grad_withoutReg] = costFunction(theta, X, y);
J = J_withoutReg + reg_term;

grad(1) = (1/m) * (X(:,1)' * (h_x - y));
grad(2:end)= (1/m) * (X(:,2:end)' * (h_x - y)) + (lambda/m) * theta(2:end); %nx1 


% =============================================================

end

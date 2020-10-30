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

h=sigmoid(X*theta);%100*1
J=( -rot90(y)*log(h)-rot90(1.-y)*log(1.-h) )/m;%1*100 x 100*1
J=J+( sum(theta.*theta)-theta(1)*theta(1) )*lambda/(m*2);
%一开始-theta(1)*theta(1)没写

grad=sum( (h-y).*X )/m + rot90(theta*lambda/m);
grad(1)=grad(1)-theta(1)*lambda/m;



% =============================================================

end

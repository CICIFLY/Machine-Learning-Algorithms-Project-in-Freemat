% formula :  y_hat = theta0 + thata1 * sin( 2*pai*f*x(i) ) +  thata2 / 3 * sin( 6*pai*f*x(i) ) + thata3/5 * sin( 10*pai*f*x(i) )
%                    + thata4/7 * sin( 14*pai*f*x(i) ) + thata5/9 * sin( 18*pai*f*x(i) ) + thata6/11 * sin( 22*pai*f*x(i) )

% gradient descent 
% the key is to solve 7 parameters:  initiate parameters = [0;0;0;0;0;0;0]
% the formula to solve parameters:   get h first,   h = (x * parameters - y)'
%  parameters(i) = parameters(i) - learningRate * 1/m * h * x(:,i); 
%  costHistory(i) = (x * parameters - y)'  * (x * parameters - y) / (2*m);
%  error_function = 1/2m * sum((y_hat-y).^2)




clc;
clear;

% load the data
data = load('sqrwave.dat');

% set up all the parameters
x1 = data(:,1);  % first column is x
y = data(:,2);
m = size(data,1); % size of rows
n = size(data,2);

% create augumented x matrix
x = [ones(m,1) sin(2*3.1415926.*x1) sin(6*3.1415926.*x1)./3 sin(10*3.1415926.*x1)./5 sin(14*3.1415926.*x1)./7 ...
sin(18*3.1415926.*x1)./9 sin(22*3.1415926.*x1)./11];   % add more columns
% size(x)


% parameters we want to solve for 
parameters = [0;0;0;0;0;0;0]; % semi colon in the brackets means it will add more rows

% hyper parameters
learningRate = 0.1;
repetition = 50;

% Storage
costHistory = zeros(repetition,1);

% iterate across epochs 
for i =1:repetition
    h = (x * parameters - y)';  % transpose of function inputs    
    for j = 1: n
        parameters(j) = parameters(j) - learningRate * 1/m * h * x(:,j);  % update the initiated parameter , 1/m * h * x(:,1) is the gradient 
        
    end 
    % cost function
    costHistory(i) = (x * parameters - y)'  * (x * parameters - y) / (2*m);    
    costHistory(i)
end
% parameters(1:3)
parameters


% pseudo inverse
x_star = inv(x' * x)*(x' * y);  % x_star creates parameters 

% do not need this 
% theta(1) = x_star(1);
% theta(2) = x_star(2);
% theta(3) = x_star(3);

% plot all them together in one line code
plot(x1, y, 'r', x1,  x * parameters,'k',x1, x*x_star, 'b'); % plot should be lines    x*x_star = y_hat
% original data --- gradient descent --- pseudo inverse 

% plot the cost
figure;
plot(costHistory, 1:repetition ,'b');



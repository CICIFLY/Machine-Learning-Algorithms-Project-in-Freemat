% formula :  y_hat = theta0 + thata1 * cos(x1) +  3 * thata2 * sin(x2 /3)

% the key is to solve the parameters:  initiate parameters = [0;0]
% the formula to solve parameters:   get h first,   h = (x * parameters - y)'
%  parameters(i) = parameters(i) - learningRate * 1/m * h * x(:,i); 
%  costHistory(i) = (x * parameters - y)'  * (x * parameters - y) / (2*m);
%  error_function = 1/2m * sum((y_hat-y).^2)


% gradient descent 


clc;
clear;

% load the data
data = load('sincos.dat');

% set up all the parameters
x1 = data(:,1);  % first column is x
x2 = data(:,2);  % second column is y
y = data(:,3);
m = size(data,1); % size of rows
n = size(data,2);

% create augumented x matrix
x = [ones(m,1)  cos(x1)  3.*sin(x2./3)];   % add more columns
% size(x)


% parameters we want to solve for 
parameters = [0;0;0]; % semi colon in the brackets means it will add more rows

% hyper parameters
learningRate = 0.1;
repetition = 10;

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
% parameters


% pseudo inverse
x_star = inv(x' * x)*(x' * y);  % x_star creates parameters 

% do not need this 
% theta(1) = x_star(1);
% theta(2) = x_star(2);
% theta(3) = x_star(3);

% plot all them together in one line code
plot3(x1, x2, y, 'r+', x1, x2, x * parameters,'k.',x1, x2, x*x_star, 'b.'); % plot3    x*x_star = theta(1)+ theta(2) * x(:,2) + theta(3)*x(:,3)


% plot the cost
figure;
plot(costHistory, 1:repetition ,'b');



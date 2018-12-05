% pseudo inverse: 
% the key is to solve the parameters:  initiate parameters = [0;0]
% formula X_star = inv(A' * A) * ( A' * B) 

% gradient descent: 
% the key is to solve the parameters:  initiate parameters = [0;0]
% the formula to solve parameters:   get h first,   h = (x * parameters - y)'
%  parameters(i) = parameters(i) - learningRate * 1/m * h * x(:,i); 
%  costHistory(i) = (x * parameters - y)'  * (x * parameters - y) / (2*m);

% cost_function/error_function = 1/2m * sum((y_hat-y).^2)

clc;
clear;


% load the data
data = load('test_parabola_2.dat');  % size 62 by 2

% set up all the parameters
x1 = data(:,1);  % first column is x
x2 = x1.^2;  % second column is y

y = data(:,2);
m = size(data,1); % size of rows
m

% create augumented x matrix
x = [ones(m,1) x1 x2 ];   % add more columns
n = size(x,2);
n

figure;
plot(x(:,2),x(:,3),'rx');  % plot second column and y
hold on

% parameters we want to solve for 
parameters = [0;0;0]; % semi colon in the brackets means it will add more rows

% hyper parameters
learningRate = 0.00001; %eta
repetition = 1000;  %epoch

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

% gradient descent : plot the linear fit 
plot(x(:,2), parameters(1) + parameters(2) * x(:,2) + parameters(3) * x(:,3),'ko'); % can not be 'k' coz it will be a plane, here should  be dots, look more resonable 
hold on;
% parameters

% pseudo inverse
x_star = inv(x' * x)*(x' * y);  % x_star creates parameters 

theta(1) = x_star(1);
theta(2) = x_star(2);
theta(3) = x_star(3);

plot(x(:,2),theta(1)+theta(2)*x(:,2)+theta(3)*x(:,3) , 'go'); 
hold on;

% plot the cost
figure;
plot(costHistory, 1:repetition ,'r');



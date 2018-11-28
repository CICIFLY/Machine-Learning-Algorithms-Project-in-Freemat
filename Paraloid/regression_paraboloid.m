% formula :  y_hat = theta0 + thata1 * x1^2 + 2 * thata2 * x1 * x2 + theta3 * x2^2

% pseudo inverse: 
% the key is to solve the parameters:  initiate parameters = [0;0]
% formula X_star = inv(A' * A) * ( A' * B) 

% gradient descent: 
% the key is to solve the parameters:  initiate parameters = [0;0]
% the formula to solve parameters:   get h first,   h = (x * parameters - y)'
%  parameters(i) = parameters(i) - learningRate * 1/m * h * x(:,i); 
%  costHistory(i) = (x * parameters - y)'  * (x * parameters - y) / (2*m);

%  cost_function / error_function = 1/2m * sum((y_hat-y).^2)

clc;
clear;


% load the data
data = load('test_paraboloid_2.dat'); 
data
% set up all the parameters
x1 = data(:,1).^2;    % first column is x
x2 = 2.*data(:,1).* data(:,2); 
x3 = data(:,2).^2;

y = data(:,3);
m = size(data,1); % size of rows
m

% create augumented x matrix
x = [ones(m,1) x1 x2 x3];   % add more columns
n = size(x,2);
n

figure;
plot3(data(:,1),data(:,2),data(:,3),'rx');  % actual values
hold on

% parameters we want to solve for 
parameters = [0;0;0;0]; % semi colon in the brackets means it will add more rows

% hyper parameters
learningRate = 0.1; 
% how to decide learning rate: if error goes down just slowly, then increase the learning rate to make it faster
% if the error goes up, means it is diverging, which is bad. should decrease the leaning rate , try 4 or 5 zeros. 


repetition = 300;  %epoch

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
%  here x has been changed so weuse data(:,1)
plot3(data(:,1),data(:,2), parameters(1) + parameters(2) * x(:,2) + 2.*parameters(3) * x(:,3) + parameters(4) * x(:,4),'ko');  
hold on;
% parameters

% pseudo inverse
x_star = inv(x' * x)*(x' * y);  % x_star creates parameters 

theta(1) = x_star(1);
theta(2) = x_star(2);
theta(3) = x_star(3);
theta(4) = x_star(4);

plot3(data(:,1),data(:,2),theta(1)+ theta(2)*x(:,2)+ 2.*theta(3)*x(:,3)+theta(4)*x(:,4) 'go'); 
hold on;

% plot the cost
figure;
plot(costHistory, 1:repetition ,'r');



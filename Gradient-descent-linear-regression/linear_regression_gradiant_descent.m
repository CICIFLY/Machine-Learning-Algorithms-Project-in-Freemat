% the key is to solve the parameters:  initiate parameters = [0;0]
% the formula to solve parameters:   get h first,   h = (x * parameters - y)'
%  parameters(i) = parameters(i) - learningRate * 1/m * h * x(:,i); 
%  costHistory(i) = (x * parameters - y)'  * (x * parameters - y) / (2*m);


clc;
clear;

% load the data
data = load('test.dat');


% set up all the parameters
X = data(:,1);  % first column is x
y = data(:,2);  % second column is y
m = size(X,1); % size of rows
% create augumented x matrix
x = [ones(m,1) X ];   % add more columns

plot(x(:,2),y,'rx');  % plot second column and y
hold on

% parameters we want to solve for 
parameters = [0;0]; % semi colon in the brackets means it will add more rows

% hyper parameters
learningRate = 0.01;
repetition = 30;

% Storage
costHistory = zeros(repetition,1);

% iterate across epochs 
for i =1:repetition
    h = (x * parameters - y)';  % transpose of function inputs
    parameters(1) = parameters(1) - learningRate * 1/m * h * x(:,1);  % update the initiated parameter
    parameters(2) = parameters(2) - learningRate * 1/m * h * x(:,2);
 
    % cost function
    costHistory(i) = (x * parameters - y)'  * (x * parameters - y) / (2*m);    
    costHistory(i)
end

% plot the linear fit 
plot(x(:,2), parameters(1) + parameters(2) * x(:,2) ,'k');
% parameters

% plot the cost
figure;
plot(costHistory, 1:repetition ,'b');


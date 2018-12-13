% the key is to solve the parameters:  initiate parameters = [0;0]
% the formula to solve parameters:   get h first,   h = (x * parameters - y)'
%  parameters(i) = parameters(i) - learningRate * 1/m * h * x(:,i); 
%  costHistory(i) = (x * parameters - y)'  * (x * parameters - y) / (2*m);
%  error_function = 1/2m * sum((y_hat-y).^2)
%  gradient descent 

clc;
clear;

% linlear regression through gradiant descent

% load the data
cancer = csvread('Breast_Cancer.dat');

% set up all the parameters
x1 = cancer(:,1);  % first column is x1
x2 = cancer(:,2);  % second column is x2
x3 = cancer(:,3);
x4 = cancer(:,4);
x5 = cancer(:,5);
x6 = cancer(:,6);
x7 = cancer(:,7);
x8 = cancer(:,8);
x9 = cancer(:,9);
x10 = cancer(:,10);  
y = cancer(:,11);

% 1) create augumented x matrix
x = [ones(size(cancer,1),1) x2  x3  x4  x5  x6  x7  x9];  % not necessary to include all 10 features but here more features increase the accuracy 


% 2) Normalize using standard deviation
m = size(x,1); % size of rows
n = size(x,2);
means_x = mean(x);   % get the means of each column
stds_x = std(x); % get the std of each column

for i = 1:m 
    for j = 1: n
        if x(i,j) == 0
            x(i,j)=means_x(j);  % replace the missing values with mean of the column
        end
    end 
    x(i,2:end) = (x(i,2:end) - means_x(2:end)) / stds_x(2:end);           %first column are 1s, so start from 2
end


plot3(x(:,2),x(:,3),y,'rx');  
hold on


% 3) Split up the data into test and train
x_train = x(1:630, :);
y_train = cancer(1:630, 11);
%size(x_train)

x_test = x(630:end,:);
y_test = cancer(630:end,11);


% 4) Gradient Descent Algorithm
% parameters we want to solve for 
parameters = [0;0;0;0;0;0;0;0];  % semi colon in the brackets means it will add more rows

% hyper parameters
learningRate = 1e-8;
repetition = 2000;

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
%parameters

% plot the linear fit 
plot3(x(:,2), x(:,3),parameters(1) + parameters(2) * x(:,2) + parameters(3) * x(:,3) + parameters(4) * x(:,4) +  parameters(5) * x(:,5) + parameters(6) * x(:,6) + parameters(7) * x(:,7) + parameters(8) * x(:,8),'ko');
% can not be 'k' coz it will be a plane, here should  be dots, look more resonable 
hold on                                                   

% plot the cost
figure;
plot(costHistory, 1:repetition ,'b');


% 5) Test on the test set : compare predicted y and actual y to see the accuracy
printf('Test Classification:')
testPredictions = x_test*parameters;
for i = 1: size(x_test,1)
    if testPredictions(i) >= 0.5   
        testPredictions(i) = 4;
        %testPredictions(i) = 1;
    else
        testPredictions(i) = 2;
        %testPredictions(i) = 0;
    end
end

[ y_test  real(testPredictions) ] 

count = 0 ;
for i=1:size(x_test,1)
    if y_test(i) == testPredictions(i)
        count = count+1;
    end
end

count / size(x_test,1)


























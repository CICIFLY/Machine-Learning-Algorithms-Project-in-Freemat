clc;
clear;

pima = load('pima.dat');
x1 = pima(:,1);
x2 = pima(:,2);    
x3 = pima(:,3);
x4 = pima(:,4);
x5 = pima(:,5);
x6 = pima(:,6);
x7 = pima(:,7);
x8 = pima(:,8);

plot(x2,x7,'*');   % read the txt info , these two variables are closely related

% 1) Augmented X matrix
x = [ones(size(x1,1),1) x2 x6];

% 2) Normalize using standard deviation
m = size(x,1);   %768
n = size(x,2);   % 3 
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

% 3) Split up the data into test and train
x_train = x(1:650, :);
y_train = pima(1:650, 9);
%size(x_train)

x_test = x(650:end,:);
y_test = pima(650:end,9);

%x_train;

% 4) Gradient Descent Algorithm
parameters = [0;0;0;];
learningRate = 1e-4;   % make it only 3 decimal
reduce =  500;
epochs = 2000;
costHistory = zeros(epochs,1);
m = size(x_train,1);

for i = 1:epochs
    h = (sigmoid(x_train*parameters)- y_train)';   % h should be transposed
    
    for k = 1 : size(parameters,1)
        parameters(k) =  parameters(k) - learningRate*h*x_train(:,k);
    end
    
    % Adaptive 
    if ( i~= 0 && mod(i,reduce)==0)
        learningRate = learningRate/2;
    end
        
    % cost histoty
    costHistory(i) = logCost(x_train,y_train,parameters);
    costHistory(i)  % it should have the i 
    
end
parameters

figure;
plot(1:epochs,costHistory,'b');

% 5) Test on the test set

printf('Test Classification:')
testPredictions = sigmoid(x_test*parameters);
for i = 1: size(x_test,1)
    if testPredictions(i) >= 0.5
        testPredictions(i) = 1;
    else
        testPredictions(i) = 0;
    end
end

[ y_test  real(testPredictions) ] 

count = 0 ;
for i=1:size(x_test,1)
    if y_test(i) == testPredictions(i)
        count = count+1;
    end
end

count / size(x_test,1)   % how well it predicts


% 6) Deterime Accuracy of classification
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
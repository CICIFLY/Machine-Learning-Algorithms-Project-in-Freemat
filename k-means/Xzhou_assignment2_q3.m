% linear regression , pseudo inverse


clc    % clear the screen
clear all % clearing memory buffer

% the key
% y_hat = m1 * x1 + m2 * x2 + m3 * x3 + m4 * x4 + m5 * x5 + m6 * x6 + m7 * x7 + m8 * x8 + b     formula
% we try to find X_star   X_star= A_cross * B_star        X_star = 9*1 vector : m1,m2,m3,m4,m4,m5,m6,m7,m8,b 

Data = load('pima_clean.txt');


% instances                                      
x = [Data(:,1:8)];
y = [Data(:, 9)];


% need to fill null , preprocess the data 
% preprocess the data and normalzie the data 
for i = 1:8
    x(:,i)(x(:,i) == 0) = mean(x(:,i)) ; % fill the null, replace all 0s with mean value
    x(:,i) = x(:,i)/max(x(:,i));  % normalize it , each column / mean of each column 
end 
x

N_x = size(x,1);  % size of 1s , 1 is all rows, way to add extra column to the matrix 
x1 = ones(N_x(1),1);    % 1 stands for 1 column

A = [ x x1] ; % A is all features and extra 1 column
B = y ;     % B is labels


A_tran = A';
A_star = A_tran * A;
A_cross = inv(A_star);

B_star = A_tran * B;

x_star = A_cross * B_star ; % all the parameters are here 
% x_star

m1_star = x_star(1);
m2_star =  x_star(2);
m3_star =  x_star(3);
m4_star =  x_star(4);
m5_star =  x_star(5);
m6_star =  x_star(6);
m7_star =  x_star(7);
m8_star =  x_star(8);
B_star = x_star(9);

   
    
y_hat = m1_star * x(:,1) + m2_star * x(:,2) + m3_star * x(:,3) + m4_star * x(:,4) + m5_star * x(:,5) + m6_star * x(:,6)+ m7_star * x(:,7) + m8_star * x(:,8) + B_star ;
y_error = abs(y-y_hat);

plot(y,'bo', y_hat,'r+' , y_error,'y+')   % do not need x , just y values can be plot coz the x has too much columns, not a single one                       % 


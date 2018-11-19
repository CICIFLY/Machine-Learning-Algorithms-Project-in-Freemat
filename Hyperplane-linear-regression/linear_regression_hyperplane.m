clc    % clear the screen
clear  % clearing memory buffer

Data = load('linear_regression_data.dat')

% instances                                       % way to add extra column to the matrix 
x = Data( : , 1);
N_x = size(x);  % size of 1s
x0 = ones(N_x(1),1);                                      % 1 stands for 1 column

% instances
y = Data( : , 2)
% labels
z = Data( : , 3)

A = [ x y x0] ;
B = z ;

A_tran = A';
A_star = A_tran * A;
A_cross = inv(A_star);

B_star = A_tran * B;

x_star = A_cross * B_star ;
m1_star = x_star(1);
m2_star =  x_star(2);
B_star = x_star(3);

z_star = m1_star * x + m2_star * y + b_star;
error_z = abs(z-z_star);

% plot(x,y,z,'b.' ,x,y,z_star,'r+')                         % raw data plotted on a  xyz space

x_rand = rand(10000,1) * 10 ;                               % generate random numbers for a column 
y_rand = rand(10000,1) * 10 ;

z_rand = m1_star * x_rand + m2_star * y_rand + b_star;
plot3(x_rand, y_rand , z_rand ,'b+' , x ,y,z ,'r+')         % raw data plotted on a  xyz space





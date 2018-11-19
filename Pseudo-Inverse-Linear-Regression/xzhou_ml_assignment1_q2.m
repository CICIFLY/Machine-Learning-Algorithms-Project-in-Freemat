% overdetermined problem : pseudo inverse for hyperplane data 

x = [ -0.1, -0.2, -0.3, -0.4, -0.5,  0.0,  0.1, 0.2, 0.3, 0.4, 0.5]' ;   % instances ,    " ' " transpose it make row be column 
y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]' ;  % labels,    " ' " transpose it make row be column     ???? are x and z instances , y labels in all cases , which one is label ? 
z = [0, 127, 255, 200, 100, 65, 30, 255, 128, 89, 230]' ;  % instances ,    " ' " transpose it make row be column 
size_x = size(x);
x_1 = ones(size_x(1),1);     

 %  z_hat = m1 * x + m2*y + b ;   % predict y value     x , y, z_hat ,  why this one does not work ? undefined variable m1 , why first question work ????

A = [ x,  y / 10.0 ,  x_1]; % this is where hyperplane comes from, multi dimensional array
B =  z / 230 ;   % pay attention b transpose 
% x = ( (A(T) * A ) inverse)  * (A(T) * B)
x_star = inv(A' * A) * (A' * B) ;   % output:  [ 4.1591 , 0.1868 ] , x star will calculate the two parameters' values

m1_star = x_star(1) ;   % first value is parameter m1
m2_star = x_star(2) ;    % second value is parameter m2 
b_star = x_star(3) ;  % third value is parameter b 
z_star = m1_star * x +  m2_star * y + b_star ; 
plot(x, y , z, 'g+' , 'r' , x , y, z_star 'b+')  % must single quote          ???? why my plot has a funny end 




    






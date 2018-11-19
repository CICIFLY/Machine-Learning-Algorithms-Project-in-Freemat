% overdetermined problem : pseudo inverse linear data 
x = [ 0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]' ;  % instances ,    " ' " transpose it make row be column 
y = [0.73,0.85,0.92,1.23,1.73,1.95,2.11,2.71,3.43,4.42,4.85]' ;  % labels,    " ' " transpose it make row be column 
y_hat = m * x + b ;   % predict y value   but here m and b are unknown, why not raise error , undefined variables ??? 
% size_x = size(x);
% x_1 = ones(size_x(1),1);   can I write it in this way ???? 
% A = [ x, x_1] ; 
A = [ 0.0, 1
         0.1, 1
         0.2, 1
        0.3, 1
        0.4, 1
        0.5, 1
        0.6, 1
        0.7, 1
        0.8, 1
        0.9, 1
        1.0, 1];
B = y / 4.85;        % after normalized , do we need to get it back to original data during plotting ?????
%  formula   x = ( (A(T) * A ) inverse)  * (A(T) * B)
x_star = inv(A' * A) * (A' * B) ;   % output:  [ 4.1591 , 0.1868 ] , x star will calculate the two parameters' values
m_star = x_star(1) ;   % first value will be parameter m 
b_star = x_star(2) ;    % second value will be parameter b 
y_star = m_star * x+ b_star ; 
plot(x, y , 'g+' , x , y_hat, 'r' , x , y_star, 'b+') % must single quote     ????  why do we need plot three , what is the one for y-star 




    






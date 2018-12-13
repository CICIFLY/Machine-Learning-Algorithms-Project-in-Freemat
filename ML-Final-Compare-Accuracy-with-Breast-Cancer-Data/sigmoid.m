function out = sigmoid(in)   % what comes out will be between 0 and 1 
    out = 1 / (1+exp(-(in)'));
    out = out' ;
end
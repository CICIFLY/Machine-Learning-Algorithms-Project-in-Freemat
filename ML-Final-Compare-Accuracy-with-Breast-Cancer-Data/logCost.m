function [costfcn] = logCost(x,y,params)
    
    numSamples = size(1,x);
    
    temp = 0.0;
    for m = 1:numSamples   % for loop handles the sum in the formula
        h = sigmoid(x*params);
        if y(m) == 1         % if statment handles the two cases when y = 0 and  y = 1 in the formula
            temp = temp + log(h); 
        else
            temp = temp + log(1-h);
        end
    end
    costfcn = -temp ; 
end 
        
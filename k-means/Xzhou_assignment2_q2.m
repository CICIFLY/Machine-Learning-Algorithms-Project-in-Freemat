clc;
clear;

% Number of centroids
K = 2;


%load pima data , pay attention to the format
load pima_clean.txt ;  


% it is dofferent from the data we had before , not (x, y) stands for a point , now it is 4 feature and 1 label for each flower 
with_diabetes = pima_clean((pima_clean(:,9) == 1),:);       % with diabetes: all rows with column values = 1 
without_diabetes= pima_clean((pima_clean(:,9) == 0),:);


% need to fill null , preprocess the data 
% preprocess the data and normalzie the data 
for i = 1:8

    without_diabetes(:,i)(without_diabetes(:,i) == 0) = mean(without_diabetes(:,i)) ; % fill the null, replace all 0s with mean value
    without_diabetes(:,i) = without_diabetes(:,i)/max(without_diabetes(:,i));  % normalize it , each column / mean of each column 

    with_diabetes(:,i)(with_diabetes(:,i) == 0) = mean(with_diabetes(:,i)) ; % fill the null, replace all 0s with mean value
    with_diabetes(:,i) = with_diabetes(:,i)/max(with_diabetes(:,i));  % normalize it , each column / mean of each column 

end 


% pack up our data
X = [with_diabetes;
     without_diabetes];
     
    
% Let m=rows n=cols
m = size(X,1);
n = size(X,2);

% Randomly init the centroids
centroids = zeros(K,n);
randidx = randperm(m);
centroids = X(randidx(1:K),:);


plot(with_diabetes(:,2),with_diabetes(:,3),'r*', without_diabetes(:,2),without_diabetes(:,3),'b*',centroids(:,1),centroids(:,2),'co');  
figure
% hold on

%%%%%%% START LOOP %%%%%%%
delta = 0.01;
break_err = 999;
while break_err > delta

  % Create the index map bin
  indices = zeros(m,1);

  for i=1:m
    
      % Initial min index guess
      k = 1;
    
      % Euclidean Distance for first K centroid
      min_dist = sqrt(sum((X(i,:) - centroids(1,:)).^2));
    
      for j=2:K
        
          dist = sqrt(sum((X(i,:) - centroids(j,:)).^2));
        
          if(dist < min_dist)
              min_dist = dist;
              k = j;
          end
      end 
      indices(i) = k;
  end

  % How would I store the old centroids?
  old_centroids = centroids;

  
  % How can I update the centroids?
  for i=1:K
  
      % How can I get all the points closest to a centroid?
      temp_bin = X(indices==i,:);
  
      % How can I count how many there are?
      count = size(temp_bin,1);
      
      % What is my update rule?
       centroids(i,:) = sum(temp_bin) * 1/count;

  end    
      
  % How can I easily get an error to break at convergence?
  break_err = sum(sqrt(sum((old_centroids - centroids).^2))./K);
  break_err
  
%%%%%%% END LOOP %%%%%%%
end

plot(with_diabetes(:,2),with_diabetes(:,3),'r*', without_diabetes(:,2),without_diabetes(:,3),'b*',centroids(:,2),centroids(:,3),'ko');  

%test = [X indices]
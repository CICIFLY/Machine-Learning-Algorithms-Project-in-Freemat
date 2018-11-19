clc;
clear;

% Number of centroids
K = 3;

%load iris data , pay attention to the format
load iris.txt ;  


% it is dofferent from the data we had before , not (x, y) stands for a point , now it is 4 feature and 1 label for each flower 
setosa = iris((iris(:,5) == 0),:);       % setosa: all rows with column values = 1 
versicolor = iris((iris(:,5) == 1),:);
virginica = iris((iris(:,5) == 2),:);
                                                                                                                 % 1 means x , 2 means y , for rows and columns

% pack up our data
X = [setosa;
     versicolor;
     virginica];
   
% Let m=rows n=cols
m = size(X,1);
n = size(X,2);

% Randomly init the centroids
centroids = zeros(K,n);
randidx = randperm(m);
centroids = X(randidx(1:K),:);
plot(setosa(:,1),setosa(:,2),'r*',versicolor(:,1),versicolor(:,2),'b*',virginica(:,1),virginica(:,2),'g*',centroids(:,1),centroids(:,2),'co');    
% hold on
figure
% How would I wrap this in a loop
% to converge? Instead of using a
% static iteration value

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

plot(setosa(:,1),setosa(:,2),'r*',versicolor(:,1),versicolor(:,2),'b*',virginica(:,1),virginica(:,2),'g*',centroids(:,1),centroids(:,2),'ko');


%test = [X indices]
    
    
    


























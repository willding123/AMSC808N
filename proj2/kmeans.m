function [D,R, f] = kmeans(A,k)
% Kmeans clustering 
n = length(A); 
R = A(randi([1,n],k,1),:); % initialize with random rows as centers
D = zeros(n,k+2); % cluster indices to which each row belongs
iter = 1;
itermax = 1000; 
f = zeros(itermax,1); 
f(1) = norm(A,'fro'); 

while iter <itermax 
for i=1:n
    for j=1:k
    D(i,j) = norm(A(i,:)-R(j,:))^2; 
    end
    [md, ci] = min(D(i,1:k));
    D(i,k+1) = md;
    D(i,k+2) = ci;
end
f(iter+1) = sum(D(:,k+1));
   
for i=1:k
    R(i,:) = mean(A(D(:,k+2)==i,:));
end

iter = iter+1; 
end
fprintf("Stopped at iteration %d\n", iter)

end


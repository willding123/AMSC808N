function [X, Y, f, stopIter] = AI(M, lambda, k, Omg)
X = ones(size(M,1), k);
Y= ones(size(M,2),k);
m = length(M); 
n = size(M,2);
maxIter = 100;
tol = 1;
iter = 0;
f= zeros(maxIter, 1); 
proj = @(M) M.*Omg;
d = norm(proj(X*Y')-M,'fro');
f(1) = d; 
while iter< maxIter && d >tol
    iter = iter+1; 
    % update X 
for i =1:m
    b = [M(i,Omg(i,:)==1)'; zeros(k,1)];
    A = [Y(Omg(i,:)==1,:); sqrt(lambda)* eye(k)];
    X(i,:)= (A\b)';
    
end
    % update Y  
for j =1:n
   b = [M(Omg(:,j)==1,j); zeros(k,1)];
   A = [X(Omg(:,j)==1,:); sqrt(lambda)* eye(k)];
   Y(j,:) = (A\b)';
    
end
%     e = norm(Xold*Yold'-X*Y', 'fro');
    d= norm(proj(X*Y')-proj(M), 'fro');
    f(iter+1) = d; 
    fprintf("Iter: %d, error: %d\n", iter, d);
end
stopIter = iter; 
fprintf("Stopped at %d, error: %d\n", iter, d);
end

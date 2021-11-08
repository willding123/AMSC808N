function [W,H,f] = projGD(A,k)
% Projected gradient descent
[m,n] = size(A);
W = randi(5,m,k)/2;
H = randi(5,k,n)/2; 
tol = 1e1;
alpha = @(iter) 0.001;
iter =1; 
itermax = 1000; 
f =zeros(100,1);
d = norm(A-W*H,'fro')^2;
f(1) = d; 
R = A-W*H; 
proj = @(M) max(M,0);
fprintf("iteration %d, fval = %d\n", iter, d); 

while d>tol  && iter < itermax
    W(:) = proj(W + alpha(iter)*R*H'); 
    H(:) = proj(H + alpha(iter)*W'*R); 
    R(:) = A-W*H; 
    d= norm(A-W*H,'fro')^2;
    iter = iter+1; 
    f(iter) = d; 
    if mod(iter, 100)==0
        fprintf("iteration %d, fval = %d, rank of W %d, rank of H %d\n", iter, d, rank(W), rank(H)); 
    end

end
fprintf("Stopped at iteration %d, fval = %d\n", iter, d); 
end


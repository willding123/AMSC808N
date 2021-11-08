function [W,H,f]  = LS(A,k)
% Lee-Seung Scheme 
[m,n] = size(A);
W = randi(5,m,k)/2;
H = randi(5,k,n)/2; 
tol = 1e1;
iter =1; 
itermax = 1000; 
f =zeros(100,1);
d = norm(A-W*H,'fro')^2;
f(1) = d; 
proj = @(M) max(M,0);
fprintf("iteration %d, fval = %d\n", iter, d); 

while d>tol  && iter < itermax
    W(:) = proj(W.*(A*H'))./proj(W*H*H'); 
    H(:) = proj(H.*(W'*A))./proj(W'*W*H); 
    d= norm(A-W*H,'fro')^2;
    iter = iter+1; 
    f(iter) = d; 
    if mod(iter, 100)==0
        fprintf("iteration %d, fval = %d, rank of W %d, rank of H %d\n", iter, d, rank(W), rank(H)); 
    end

end
fprintf("Stopped at iteration %d, fval = %d\n", iter, d); 

end


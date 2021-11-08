function [M, f, stopIter] = Nuc(A, lambda, Omg)
M = ones(size(A)).*2.5;
maxIter = 100;
tol = 1;
iter = 0;
f= zeros(maxIter, 1); 
proj = @(M) M.*Omg;
d = norm(proj(M)-A,'fro');
f(1) = d; 
while iter< maxIter && d >tol
    iter = iter+1; 
    [U, S, V] = svd(M+proj(A-M));
    S(1:length(V),:) = diag(max(diag(S)-lambda,0));
    M(:) = U*triu(S)*V';
    d= norm(proj(A-M), 'fro');
    f(iter+1) = d; 
    fprintf("Iter: %d, error: %d\n", iter, d);
end
stopIter = iter; 
fprintf("Stopped at %d, error: %d\n", iter, d);
end

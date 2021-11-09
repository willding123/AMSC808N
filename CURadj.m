function [C,U, R,l1,l2] = CURadj(A,k,t,q)
% CUR decomposition 
eps = 0.1; 
[C, l1] = columnselect(A,k,eps,t,q);
[R, l2] = columnselect(A',k,eps,t,q);
U = pinv(C)*A*pinv(R');

end

function [C, p1] = columnselect(A, k, eps,t,q)
n =size(A,2);
alpha = 0.075; 
[~, ~, V] = svd(A,'econ');
c= t*k*log(k/eps);
p = sum((V(:,1:k)).^2,2)/k;
if size(p) ==size(q)
    p1 = alpha*q+(1-alpha)*p; 
else 
    p1 = p;
end

d = min(c*p1,1);
C = A(:,d>rand(n,1));
end



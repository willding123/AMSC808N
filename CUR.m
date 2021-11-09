function [C,U, R,l1,l2] = CUR(A,k,t)
% CUR decomposition 
eps = 0.1; 
[C, l1] = columnselect(A,k,eps,t);
[R, l2] = columnselect(A',k,eps,t);
U = pinv(C)*A*pinv(R');

end

function [C, p] = columnselect(A, k, eps,t)
n =size(A,2);
[~, ~, V] = svd(A,'econ');
c= t*k*log(k/eps);
p = sum((V(:,1:k)).^2,2)/k;
d = min(c/k*sum((V(:,1:k)).^2,2),1);
C = A(:,d>rand(n,1));
end



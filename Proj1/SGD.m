function [w,f,normgrad] = SGD(fun,gfun,Xtrain,w,bsz,kmax,tol)
[n, ~] = size(Xtrain); 
I = randperm(n,bsz)'; 
f = zeros(kmax+1,1);
normgrad = zeros(kmax+1,1);
g(:) = gfun(I,w);
normgrad(1) = norm(g);
f(1) = fun(I,w);
fprintf("k = %d, f=%d, ||g|| =%d\n ", 0,f(1), normgrad(1));
alpha = @(iter) 0.5/iter;
for k=1:kmax 
    w(:) = w-alpha(k)*g';
    I(:) = randperm(n,bsz)'; 
    g(:)  = gfun(I,w);
    normgrad(k+1) = norm(g); 
    f(k+1) = fun(I,w);    
    if mod(k,100)==0
        fprintf('k = %d, alpha = %d, f = %d, ||g|| = %d\n',k,alpha(k), f(k+1),normgrad(k+1));
    end
   
    if normgrad < tol 
        break;
    end
    
end

fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end
        
        
    
    

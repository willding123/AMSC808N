function [w,f,normgrad] = ASG(fun,gfun,Xtrain,w,bsz,kmax,tol)

[n, ~] = size(Xtrain); 
I = randperm(n,bsz)'; 
f = zeros(kmax+1,1);
normgrad = zeros(kmax+1,1);
alpha = @(iter) 0.1;
mu = @(iter) 1-3/(5+iter);
wprev = zeros(size(w));
y = (1+mu(0))*w - mu(0)*wprev; 
g(:) = gfun(I,y); 
wprev(:) = w; 
w(:) = y - 0.1*g'; 
normgrad(1) = norm(g);
f(1) = fun(I,w);
fprintf("k = %d, f=%d, ||g|| =%d\n", 0,f(1), normgrad(1));

for k=1:kmax 
    y(:) = (1+mu(k))*w-mu(k)*wprev;
    I(:) = randperm(n,bsz)'; 
    g(:)  = gfun(I,y);
    wprev(:) = w;
    w(:) = y - alpha(k)*g'; 
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


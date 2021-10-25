function [w,f,normgrad] = SAdam(fun,gfun,Xtrain,w,bsz,kmax,tol)

[n, ~] = size(Xtrain); 
I = randperm(n,bsz)'; 
f = zeros(kmax+1,1);
normgrad = zeros(kmax+1,1);
alpha = @(iter) 0.01;
beta1 = 0.9;
beta2 = 0.999;
g(:) = gfun(I,w); 
m = zeros(size(g));
mt = zeros(size(g));
v = zeros(size(g));
vt = zeros(size(g));
normgrad(1) = norm(g);
eps = 10^-8;
f(1) = fun(I,w);
fprintf("k = %d, f=%d, ||g|| =%d\n", 0,f(1), normgrad(1));

for k=1:kmax 
    
    m(:) = beta1*m +(1-beta1)*g;
    v(:) = beta2*v + (1-beta2)*(g.^2);
    mt(:) = m/(1-beta1^k);
    vt(:) = v/(1-beta2^k);
    w(:) = w - alpha(k)*mt'./(sqrt(vt)+eps)';
    
    I(:) = randperm(n,bsz)'; 
    g(:)  = gfun(I,w);
   
    normgrad(k+1) = norm(g); 
    f(k+1) = fun(I,w);    
    if mod(k,100)==0
        fprintf('k = %d, alpha = %d, f = %d, ||g|| = %d\n',k+1,alpha(k), f(k+1),normgrad(k+1));
    end
   
    if normgrad < tol 
        break;
    end
    
end

fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));

end


function [w,f,normgrad] = GN(fun,w,kmax,tol)
% Gauss Newton 
% Initialize variables
f = zeros(kmax+1,1);
normgrad = zeros(kmax+1,1);
iter=1; 
[r,J] = fun(w);
g = J'*r; 
nor = norm(g); 
normgrad(1) = nor; 
f(1) = 0.5*norm(r)^2; 
% alpha = @(iter) 0.1; 
fprintf('iter = %d, f = %d, ||g|| = %d\n',iter, f(iter),normgrad(iter));

while iter < kmax && nor > tol 
    p = (J'*J+eye(length(w))*1e-6)\(-J'*r); % compute direction
    [a,j,rtry, Jtry,f(iter+1)] = linesearch(w,p,g,f(iter),fun); % use line search to compute step size
    if j == 10 
        p=-g;
        [a,~,rtry, Jtry,f(iter+1)] =  linesearch(w,p,g,f(iter),fun);
    end
    J(:) = Jtry; r(:) =rtry;
    w = w+a*p;
    g = J'*r; 
    nor = norm(g); 
    normgrad(iter+1) = nor; 
    if mod(iter+1,1)==0
        fprintf('iter = %d, f = %d, ||g|| = %d\n',iter+1, f(iter+1),normgrad(iter+1));
    end
    iter = iter+1; 
end
fprintf('GaussNewton: %d iterations, norm(g) = %d\n',iter,nor);


end
function [a,j,r, J,f1] = linesearch(x,p,g,f0,fun)
    a = 1;
    gam = 0.9;
    jmax = 10;
    eta = 0.5;
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        [r,J] = fun(xtry);
        f1 = r'*r/2;
        if f1 < f0 + a*aux
            break;
        else
            a = a*gam;
        end
    end
end

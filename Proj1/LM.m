function [w,f,normgrad] = LM(fun,w,kmax,tol)
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


% trust region params 
dmax = 101.05; 
dmin = 1; 
d = 0.2*dmax; 
eta = 0.1; 
I =eye(size(J,2), size(J,2));
while iter < kmax && nor > tol 
    
    B = J'*J + (1e-6)*I;
    pstar = -B\g; % unconstrained minimizer
    if norm(pstar) <= d
        p = pstar;
    else % solve constrained minimization problem
        lam = 1; % initial guess for lambda
        while 1 
        B1 = B + lam*I;
        C = chol(B1); % do Cholesky factorization of B 
        p = -C\(C'\g); % solve B1*p = -g
        np = norm(p);
        dd = abs(np - d); % d is the trust region radius 
        
        if dd < 1e-6
            break 
        end
%         q = C'\p; % solve C^\top q = p
        q = B1^(0.5)\p;
        nq = norm(q);
        lamnew = lam + (np/nq)^2*(np - d)/d; 
        if lamnew < 0
            lam = 0.5*lam; 
        else
            lam = lamnew;
        end
        end
    end
    [rp,~] = fun(w+p);
    mp = f(iter)+p'*g+0.5*p'*J'*J*p;
    rho = 0.5*(norm(r)^2-norm(rp)^2)/(0.5*norm(r)^2-mp);
    if rho < 0.25
        d = max(0.25*d,dmin); 
    else
        if rho>0.75 && abs(norm(p)-d)< 1e-5
            d = min(2*d,dmax);
        end
    end
    if rho > eta 
        w(:) = w+p;
    else
        w(:) = w;
    end
    [r, J] = fun(w); 
    g = J'*r; 
    nor = norm(g); 
    normgrad(iter+1) = nor; 
    f(iter+1) = 0.5*norm(r)^2; 
    if mod(iter+1,1)==0
        fprintf('iter = %d, f = %d, ||g|| = %d\n',iter+1, f(iter+1),normgrad(iter+1));
    end
    iter = iter+1; 
end
fprintf('LM: %d iterations, norm(g) = %d\n',iter,nor);


end


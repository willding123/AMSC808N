%% Q1
x = [0:5].*(pi/10);
g = 1-cos(x);
a = linspace(-1,1,50);
b = linspace(-1,1,50);
[AA B] = meshgrid(a,b);
Z = zeros(size(AA)); 
for i=1:size(AA,1)
    for j=1:size(AA,2)
        Z(i,j) = sum((relu(AA(i,j)*x-B(i,j))-g).^2)/12;
    end
end
figure(1)
surf(AA,B,Z)
figure(2)

imagesc(Z)
colorbar
%%
p = zeros(6,2);
f = zeros(6,1);
R = zeros(6,6);
for i=1:6
    A = [sum(x(i:6).^2) -sum(x(i:6)); sum(x((i:6))) -(7-i)];
    b= [sum(g(i:6).*x(i:6)); sum(g(i:6))]; 
    p(i,:) = A\b;
    err = norm(A*p(i,:)'-b)/norm(b)
    R(i,:) = relu(p(i,1)*x-p(i,2));
    f(i) = sum((relu(p(i,1)*x-p(i,2))-g).^2)/12;
end

R= [[0:5]' R];

%% GD
% q =(a,b); q = p-alpha*d
iter = 0; 
tol = 1e-4;
iter_max = 10^4;
q = [1;0];
alpha = 1.29;
d = del(q,x,g);
nor = norm(d);
err = zeros(100,1);
while nor > tol && iter < iter_max
    q(:) = q-alpha*d; 
    d(:) = del(q,x,g);
    nor = norm(d);
    iter = iter+1; 
    err(iter) = norm(q-p)/norm(p);
end

%% SG 

iter = 0; 
tol = 1e-4;
iter_max = 10^3;
q = [1;0];
alpha = @(iter) 1.3/iter;
e = randi([1,6]);
d = dels(q,x,g,e);
nor = norm(d);
err = zeros(1000,1);

while iter < iter_max
    q(:) = q-alpha(iter+1)*d; 
    e = randi([1,6]);
    d(:) = dels(q,x,g,e);
    nor = norm(d);
    iter = iter+1; 
    err(iter) = norm(q-p)/norm(p);
end

%% derivative
function d = del(p,x,g)
k = ceil(10*p(2)/(p(1)*pi)) +1;
if k>6
    d = zeros(2,1);
    return
end
try 
    delf = sum((p(1)*x(k:6)-p(2)-g(k:6)).*x(k:6))/6;
catch 
    warning("Array indices must be positive integers or logical values.")
    k
end
delg = -sum(p(1)*x(k:6)-p(2)-g(k:6))/6;
d= [delf;delg];
end 

%% SG derivative
function d = dels(p,x,g,e)
if p(1)*x(e)-p(2)>0
    delf = (p(1)*x(e)-p(2)-g(e))*x(e);
    delg = -(p(1)*x(e)-p(2)-g(e));
    d = [delf;delg];
    return
else 
    d = [0;0];
    return 
end
    
end


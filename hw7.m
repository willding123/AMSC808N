s=linspace(1,100,100); % component size
Ps = zeros(size(s));
N =100; % number of partitions for trapozoidal rule 
ip = linspace(0,2*pi,N+1); % integral partitions  
ip(end) = []; % avoid calculating end points twice 
p =1-.875;
for k=1:length(s)
z = @(p) 3-2*p; % mean degree
mu = @(x,p) (z(p)-sqrt(z(p)^2-12*x.^2*(1-p)*p))./(6*x*(1-p)); % quadratic root 
G = @(x,p) p*x + (1-p)*x.^3; % generating function for degree distribution of a randomly selected vertex
H = @(x,p) x.*G(mu(x,p),p); % H0 - generating function for component size distribution when randomly select a vertext
f = @(x) H(x,p)./x.^(s(k)); 

% composite trapozoidal 
y = exp(1i*ip); 
Ps(k)= real(sum(f(y))/N);
end 
plot(s,Ps,'-+b')
hold on
xlabel("component size")
ylabel("probability")
title("Component Size Distribution")
sum(Ps)
% test accuracy of the above code using integral of 1/z along the unit circle from
% 0 to 2pi
% it is proven that this integral is equal to 2*pi*1i
I = 0;
ft = @(x) 1/x; 

for j =1:N 
    I =I+ft(exp(2*pi*j*1i/N))*exp(2*pi*j*1i/N)*1i;
end
I = I*2*pi/N; 
err = I - 2*pi*1i; % very good accuracy order of 1e-16 when N=10; 

%%
rng(111);
n= 1e4; % number of nodes 
% generate vertices with given degree distribution
k = rand(n,1); 
k(k>p)=3; 
k(k<p)=1;
S = sum(k); % number of edges
while mod(S,2)~=0
k(:) = rand(n,1); 
k(k>p)=3; 
k(k<p)=1;
S = sum(k);
end
label = zeros(S,1); 
st =1; 
for i =1:n
    label(st:st+k(i)-1) = i;
    st = st+k(i);
end

label = label(randperm(S));

A = sparse(label(1:S/2), label(S/2+1:end),1,n,n);
A = A+A'; % symmetric

A = min(A,1);
A =  A- diag(diag(A));

%% DFS

G=graph(A);
m = ones(n,1);
L = []; % store component sizes 
while isempty(find(m))~=1
    i = find(m,1);
    v = dfsearch(G,i); 
    L = [L length(v)];
    m(v) = 0;
end
Ps1 = zeros(size(s));
for i=1:100 
    Ps1(i) = i*sum(L==i)/n; % estimate probabilities for component-size distribution
end
plot(s,Ps1)
legend("Cauchy","Randgraph")

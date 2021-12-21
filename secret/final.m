%% Q1 & Q2
data = readmatrix("lastfm_asia_edges.csv");
n = 7624; % number of nodes 
A = zeros(n,n); % build adjacency matrix
for i=1:length(data)
    A(data(i,1)+1, data(i,2)+1) =1; 
end
A = A+A'; % make symmetric
G=graph(A); % make graph
m = ones(n,1);
L = []; % store component sizes 
while isempty(find(m))~=1
    i = find(m,1);
    v = dfsearch(G,i); 
    L = [L length(v)];
    m(v) = 0;
end
%% Q3. figures for degree distribution  
did = degree(G); % degree for each id
pk = zeros(max(did),1); % degree distribution
for i =1:max(did)
    pk(i) = sum(did==i)/length(did);
end
scatter(1:length(pk),pk);
set(gca,'xscale','log','yscale','log')
hold on 
nmax = floor(log2(max(did)))-1;
logbin = zeros(nmax+1, 2); 
for i =0:nmax
    logbin(i+1,1) = [2^i:2^(i+1)-1]*pk(2^i:2^(i+1)-1)/sum(pk(2^i:2^(i+1)-1));
    logbin(i+1,2) = sum(pk(2^i:2^(i+1)-1))/2^i;
end
scatter(logbin(:,1), logbin(:,2),"filled");
hold on 

%% Q4. optimization 
lp = [ones(7,1) -log(logbin(:,1)) -(logbin(:,1))];
c= lp\log(logbin(:,2)); % c(1) = C, c(2) = T, c(3) = a; 
c(1) = exp(c(1));
fp = @(k) c(1)*exp(-c(3)*k).*k.^(-c(2)); 
scatter(logbin(:,1),fp(logbin(:,1)),"filled")

legend("loglog",'log-binning','power law')

%% Q5 average path lengths
d = distances(G); 
d_avg = sum(d,'all')/n/(n-1); % average shortest path length 

% z1 = mean(degree(G));
% z2 = (1:216).^2*pk-z1;
z1 = fp(1:300)*(1:300)';
z2 = fp(1:300)*((1:300)'.^2)-z1;
l = log(n/z1)/log(z2/z1)+1; 
% l = (log((n-1)*(z2-z1)+z1^2)-log(z1^2))/log(z2/z1);

%% Q6 clustering coefficient 
C = trace(A^3)/(sum(A,2)'*(sum(A,2)-1));
Cr = z2^2/z1^3/n; % clustering coefficient for the random graph 

%% Q8 & Q9 
rng(0)
Tc = ((1:216)*pk)/(((1:216).*(0:215))*pk);
T= 0.4;
E = find(triu(A)) ; % all undirected edges' linear indices
Te = ceil(T*length(E)); % number of edges in transmission graph 
% [row col] = ind2sub([n,n],E(randperm(length(E),Te))); % transmission graph edges
% Tg = [row col]; 
Tg = zeros(n,n);
idx = E(randperm(length(E),Te));
Tg(idx)=1; 
Tg  = Tg+Tg';
sick = 1026;

count=1; % count of sick people 
cp = 0; % count of sick people from previous time step 
add = 1; 
vcount = [1]; % infected number after each time step 
for i =1:10 
    new = []; 
   for j=(cp+1):(cp+add)
    row  = sick(j);
    if j~=1
        if any(sick(1:j-1)==row)
             continue   
        end    
    end 
    idx = find(Tg(row, :)); 
    new = [new;idx'];
   end
   sick = [sick; new]; 
   add = length(new);
   cp = count;
   count = length(unique(sick));
   vcount = [vcount; count];
end
% ind = sub2ind([n,n], sick(:,1), sick(:,2));
% any(Tg(ind)==0) % check if all edges in "sick" are correct
figure(1);plot(vcount(2:end))
Pe = count/n; % fraction of nodes infected
%     
% for i =1:10
%     
% end
%% Q10 random graph with same degree distribution
% Tcr = ((1:216)*fp(1:216)')/(((1:216).*(0:215))*fp(1:216)'); % critical transmissibility for random graph 
Tcr = real(polylog(c(2)-1, exp(-1*c(3)))/(polylog(c(2)-2, exp(-1*c(3)))-polylog(c(2)-1,exp(-1*c(3)) )));
fpl = @(k) k.^(-c(2)).*exp(-k*c(3))/polylog(c(2), exp(-1*c(3)) );
% fq = @(k) (k+1).*fpl(k+1)/z1; 
% G1 = @(x) fq((1:216))*x.^((1:216)');
G0 = @(x) polylog(c(2),x*exp(-1*c(3)))/polylog(c(2),exp(-1*c(3)));
G1 = @(x) polylog(c(2)-1, x*exp(-1*c(3)))/(x*polylog(c(3)-1,exp(-1*c(3))));
dG1 =  @(x)1.0./x.^2.*polylog(-1.004256392651763,x.*9.566005994957725e-1).*(2.302252249979772e-3+4.426584198965011e-27i)+1.0./x.^2.*polylog(-4.256392651762586e-3,x.*9.566005994957725e-1).*(-2.302252249979772e-3-4.426584198965011e-27i);


% solve u = G1(1+(u-1)T) where T = 0.4 using Newton 
mu = 1; 
tol = 1e-3;
alpha =@(k) 0.01/sqrt(k+100);
k =1; 
kmax = 1e4; 
while abs(real(mu-G1(1+(mu-1)*T)))>=tol && k<kmax && mu>0
    p = real(-G1(1+(mu-1)*T)/(dG1(1+(mu-1)*T)*T)); 
    mu = mu+ alpha(k) * p; 
    k = k+1; 
end
S = 1-G0(1+(mu-1)*T);
% x = x + alpha*p; 


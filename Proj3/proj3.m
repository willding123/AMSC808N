% diffusion map for emoji dataset

% set up dataset 
data = data3; 
[m,n] = size(data); 

% find epsilon
delta = zeros(m,n); 
drowmin  = zeros(m,1);
for i=1:m 
    for j=1:m
        delta(i,j) = norm(data(i,:) - data(j,:)); 
    end
end

for i=1:m 
    drowmin(i) = min(delta(i,setdiff(1:m,i)));
end


%%
eps = 2.5*mean(drowmin);
delta = 0.1; 
d =3 ;
[Xd] = diffusion(data,d+1, eps,delta);
plot3(Xd(:,1),Xd(:,2),Xd(:,3))
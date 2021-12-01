load FaceData.mat
% set up dataset 
data = data3; 
[m,n] = size(data); 
% plot color and markersize
c= linspace(1,10,m); 
sz = 25;
% find epsilon for diffusion map 
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

%% diffusion 
eps = 1e2*mean(drowmin);
delta = 0.9; 
d =3 ;
[Xd] = diffusion(data,d+1, eps,delta);
% [Xd] = diffMap(data,eps,delta,dim);
scatter3(Xd(:,1),Xd(:,2),Xd(:,3),sz,c);

%% PCA 
% center the data 
Y = data - ones(m,1)* mean(data,1);
[~, ~, V] = svd(Y,'econ'); 
scatter3(Y*V(:,1),Y*V(:,2), Y*V(:,3),sz,c);

%% Isomap 
isomap

%% LLE 
k=5;
d =1;
X =  data3'+.1*rand(size(data3')); 
Xd = lle(X, d, k);
% scatter3(Xd(:,1),Xd(:,2),Xd(:,3),sz,c)

%% TSNE
[Yt_sne, loss] = tsne(data3,'Algorithm','exact','Perplexity',6);
scatter(Yt_sne(:,1),Yt_sne(:,2),sz,c)
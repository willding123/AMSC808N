function [Xd] = diffusion(A,d, epsilon,delta)
%   Diffusion maps 
[m,~] = size(A); 

P = zeros(m,m);
rowsum  = zeros(m,1); 
for i=1:m 
    for j=1:m
       P(i,j)= exp(-norm(A(i,:)-A(j,:),2)^2/epsilon);
       rowsum(i) = rowsum(i)+ P(i,j);
    end
    P(i,:) = P(i,:)/rowsum(i);
end
rowsum = rowsum./sum(rowsum);
pie = diag(rowsum);

% [V S] = spectralfact(pie^(0.5)*P*pie^(-0.5));
[V S] =eig(pie^(0.5)*P*pie^(-0.5)); 
[absevals,ind] = sort(abs(diag(S)),'descend');
V = V(:,ind);
S = S(ind,ind);
R = pie^(-0.5)*V; 
fprintf("norm(R'PieR-I)=%d \n",norm(R'*pie*R-eye(m)));
t = ceil( log(1/delta) / log(absevals(2)/absevals(d)) );
Xd = R(:,2:d)*S(2:d,2:d).^t; 
end


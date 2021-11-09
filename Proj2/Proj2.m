%% Read Matrix and set Param
M = readmatrix("MovRankData.csv");
M = M(:,2:end);
Omg = isnan(M)==0;
M(isnan(M)==1) =0;
lambda = 1;
var = {'Home Alone'; 'The Lion King'; 'The Princess Bride'; 'Titanic'; 'Beauty and the Beast'; 'Cinderella'; 'Shrek'; 'Forrest Gump'; 'Aladdin'; 'Ferris Buellers Day Off'; 'Finding Nemo'; 'Harry Potter and the Sorcerers Stone';'Back to the Future'; 'UP';	'The Breakfast Club'; 'The Truman Show'; 'Avengers: Endgame'; 'The Incredibles'; 'Coraline'; 'Elf'};


%% Low rank approx
k=5; 
[X, Y, f, stopIter] = AI(M, .5, k, Omg);
var = {'Home Alone'; 'The Lion King'; 'The Princess Bride'; 'Titanic'; 'Beauty and the Beast'; 'Cinderella'; 'Shrek'; 'Forrest Gump'; 'Aladdin'; 'Ferris Buellers Day Off'; 'Finding Nemo'; 'Harry Potter and the Sorcerers Stone';'Back to the Future'; 'UP';	'The Breakfast Club'; 'The Truman Show'; 'Avengers: Endgame'; 'The Incredibles'; 'Coraline'; 'Elf'};
P = X*Y';
T = table( var, P(55,:)','VariableNames',{'Movies','User'})
plot(f(1:stopIter))
xlabel("Iteration")
ylabel("Frobenius norm")
%% Nuclear trick
[M1, f, stopIter] = Nuc(M, 0.5, Omg); 

var = {'Home Alone'; 'The Lion King'; 'The Princess Bride'; 'Titanic'; 'Beauty and the Beast'; 'Cinderella'; 'Shrek'; 'Forrest Gump'; 'Aladdin'; 'Ferris Buellers Day Off'; 'Finding Nemo'; 'Harry Potter and the Sorcerers Stone';'Back to the Future'; 'UP';	'The Breakfast Club'; 'The Truman Show'; 'Avengers: Endgame'; 'The Incredibles'; 'Coraline'; 'Elf'};
T = table( var, M1(55,:)','VariableNames',{'Movies','User'})

plot(f(1:stopIter))
xlabel("Iteration")
ylabel("Frobenius norm")
%% Kmeans 
k = 3; 
[D,R, f] = kmeans(M1,k);
table(var, R(1,:)', R(2,:)',R(3,:)','VariableNames',{'Movies','G1', 'G2','G3'})

%% NMF 
% projected GD 
k=3;
[W,H,f] = projGD(M1,k);
figure(1);
plot(f);
hold on 
[W,H,f1] = LS(M1,k);
plot(f1);;
legend("projGD", "LS");
xlabel("iteration")
ylabel("Frobenium norm squared")

%% CUR
[M2,Mcounts,y,words] = readdata(); 
L1 = [];
L2 = [];
t = [0.5, 1, 2, 4, 8, 16];
fc = zeros(length(t),1); 
fr = zeros(length(t),1); 

% for k = 2:20 
%     [C,U, R, l1, l2] = CUR(M,k,t);
%     L1 = [L1;l1'];
%     L2  = [L2;l2'];
%     
% 
% end

k=7; 
[U S V] = svd(M2, 'econ');
Mk = U(:,1:k)*S(1:k,1:k)*V(:,1:k)'; 
for i = 1:length(t)
    [C,U, R, l1, l2] = CUR(M2,k,t(i));
    fc(i) = norm(M2-C*U*R','fro')/norm(M2-Mk,'fro');
end

%%
figure(1);
for i=1:19
    plot(L1(i,:))
    hold on
end
legend("k=2","k=3","k=4","k=5","k=6","k=7","k=8","k=9","k=10","k=11","k=12","k=13","k=14","k=15","k=16","k=17","k=18","k=19","k=20")
xlabel("indices")
ylabel("leverage score")
figure(2);
for i=1:19
    plot(L2(i,:))
    hold on
end
legend("k=2","k=3","k=4","k=5","k=6","k=7","k=8","k=9","k=10","k=11","k=12","k=13","k=14","k=15","k=16","k=17","k=18","k=19","k=20")
xlabel("indices")
ylabel("leverage score")
%% top 5 pearson informaiton gain
[M,Mf,y,words] = readdata();

% preprocessing and get rid of stopwords
stopwords={'the','gif','and','for','jpg','our','with','you','your','this','page','com','2003','are','links','y100-fm','that','after','before','need','561-305-3558','135','For','will','all','may','what','here','please','other','has','have','can','about','from','more','information','one','not','click','also','their','who','only','just','copyright','most'};
Mu = [];
wordsu = {};
for i =1:length(M)
    if ~any(strcmp(stopwords,words(i)))
        Mu = [Mu M(:,i)];
        wordsu{end+1} = char(words(i));
    end
end
%%
% pearson information gain
[n,d] = size(Mu);
i1 = find(y==-1);
i2 = find(y==1);
ii = find(Mu>0);
n1 = length(i1);
n2 = length(i2);

M = full(Mu);
Mfreq = sum(M,1)/n;
M1freq = sum(M(i1,:),1)/n1;
M2freq = sum(M(i2,:),1)/n2;
IG = abs(M1freq-M2freq).*Mfreq; % information gain
q = IG'/sum(IG);
[ig, ind]= maxk(IG,5);


[~,~, ~, l1, ~] = CURadj(Mu,20,1,q);
[ls, ind1] = maxk(l1,20);
wordsu(ind1)
wordsu(ind)

%% project onto 2D using PCA
[~, ind5] = maxk(l1,5);

C = Mu(:,ind5); 
[U S ~] = svd(C);
% for i = 1:length(y)
%     if y(i) ==1; 
%         break;
%     end
% end
P = U(:,1:2)*S(1:2,1:2);

plot(P(1:72,1), P(1:72,2),'.','Markersize',20,'color','k')
hold on 
plot(P(73:end,1), P(73:end,2),'.','Markersize',20,'color','b')
legend("Indiana","Florida")


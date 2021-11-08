%% Read Matrix and set Param
M = readmatrix("MovRankData.csv");
M = M(:,2:end);
Omg = isnan(M)==0;
M(isnan(M)==1) =0;
lambda = 0.1;
var = {'Home Alone'; 'The Lion King'; 'The Princess Bride'; 'Titanic'; 'Beauty and the Beast'; 'Cinderella'; 'Shrek'; 'Forrest Gump'; 'Aladdin'; 'Ferris Buellers Day Off'; 'Finding Nemo'; 'Harry Potter and the Sorcerers Stone';'Back to the Future'; 'UP';	'The Breakfast Club'; 'The Truman Show'; 'Avengers: Endgame'; 'The Incredibles'; 'Coraline'; 'Elf'};


%% Low rank approx
k=5; 
[X, Y, f, stopIter] = AI(M, lambda, k, Omg);
var = {'Home Alone'; 'The Lion King'; 'The Princess Bride'; 'Titanic'; 'Beauty and the Beast'; 'Cinderella'; 'Shrek'; 'Forrest Gump'; 'Aladdin'; 'Ferris Buellers Day Off'; 'Finding Nemo'; 'Harry Potter and the Sorcerers Stone';'Back to the Future'; 'UP';	'The Breakfast Club'; 'The Truman Show'; 'Avengers: Endgame'; 'The Incredibles'; 'Coraline'; 'Elf'};
P = X*Y';
T = table( var, P(55,:)','VariableNames',{'Movies','User'})

%% Nuclear trick
[M1, f, stopIter] = Nuc(M, lambda, Omg); 

var = {'Home Alone'; 'The Lion King'; 'The Princess Bride'; 'Titanic'; 'Beauty and the Beast'; 'Cinderella'; 'Shrek'; 'Forrest Gump'; 'Aladdin'; 'Ferris Buellers Day Off'; 'Finding Nemo'; 'Harry Potter and the Sorcerers Stone';'Back to the Future'; 'UP';	'The Breakfast Club'; 'The Truman Show'; 'Avengers: Endgame'; 'The Incredibles'; 'Coraline'; 'Elf'};
T = table( var, M1(55,:)','VariableNames',{'Movies','User'})

%% Kmeans 
k = 3; 
[D,R, f] = kmeans(M1,k);
table(var, R(1,:)', R(2,:)',R(3,:)','VariableNames',{'Movies','G1', 'G2','G3'})

%% NMF 
% projected GD 
k=3;
[W,H,f] = projGD(M1,k);
% [W,H,f] = LS(M1,k);



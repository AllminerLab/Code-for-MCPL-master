function [ACC, MIhat, Purity] = MCPL(X, truthY, k, c, m, n, niter)

lambda = randperm(30,1);
%initialize weighted_distX
SUM = zeros(n);
for v = 1:m
    distX_initial =  L2_distance_1(X{v},X{v}) ;                 
    SUM = SUM + distX_initial;
end
distX = SUM/m;
[distXs, idx] = sort(distX,2);

%initialize S
S = zeros(n);
rr = zeros(n,1);
for i = 1:n
    di = distXs(i,2:c+2);
    rr(i) = 0.5*(c*di(c+1)-sum(di(1:c)));
    id = idx(i,2:c+2);
    S(i,id) = (di(c+1)-di)/(c*di(c+1)-sum(di(1:c))+eps); 
end
alpha = mean(rr);

%% initialize F
S = (S+S')/2;                                                         % initialize F
D = diag(sum(S));
Ls = D - S;
F = my_eig(Ls, k, 0);

%% initialize w^v
for v = 1:m
	w{v} = 1/m;
end

for iter = 1:niter
%     fprintf('Iteration %d...\n', iter);

    %% Upate U^v
    U = cell(m, 1);
    for v = 1:m
        U{v}=X{v}/(eye(n)+2*w{v}*Ls);
    end
    
    %% w^v
    w = cell(m, 1);
    distU_updated = cell(m, 1);
    SUM = zeros(n);
    for v = 1:m
        distU_updated{v} =  L2_distance_1(U{v}, U{v}) ;
        w{v} = (0.5) / sqrt(sum(sum( distU_updated{v}.*S)));
        SUM = SUM + w{v}*distU_updated{v};
    end
    distU = SUM;
    
    %% update S
    distf = L2_distance_1(F',F');
    S = zeros(n);
    for i=1:n                                                         
        idxa0 = idx(i,2:k+1);
        dfi = distf(i,idxa0);
        dui = distU(i,idxa0);
        ad = -(dui+lambda*dfi)/(2*alpha);
        S(i,idxa0) = EProjSimplex_new(ad);
    end
    
    %% Update F
    S = (S+S')/2;                                                       
    D = diag(sum(S));
    Ls = D-S;
    F_old = F;
    [F, ~, ev]=my_eig(Ls, k, 0);
    evs(:,iter+1) = ev;
    
    %update lambda
    thre = 1*10^-10;
    fn1 = sum(ev(1:k));                                             
    fn2 = sum(ev(1:k+1));
    if fn1 > thre
        lambda = 2*lambda;
    elseif fn2 < thre
        lambda = lambda/2;  F = F_old;
    else
        break;
    end
    
end

[clusternum, y] = graphconncomp(sparse(S)); 
y = y';
if clusternum ~= k
    sprintf('Can not find the correct cluster number: %d', clusternum)
end
[ACC, MIhat, Purity] = ClusteringMeasure(truthY, y);
end


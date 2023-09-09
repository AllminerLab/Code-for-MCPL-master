clear
close all

%% Mfeat
load('Mfeat.mat');
X = {fac, fou, kar, mor, pix, zer};
dataname = 'Mfeat';
viewname = {'fac', 'fou', 'kar','mor', 'pix', 'zer'};
c = 25;
fprintf('Datasets: Mfeat, c = %d ...\n', c);

tic; %timing

niter = 20; 
k = length(unique(y));
m = length(X);
n = size(X{1}, 1);

for v = 1:m
    for i = 1:n
        normItem = std(X{v}(i,:));
        if (0 == normItem)
            normItem = eps;
        end
        X{v}(i,:) = (X{v}(i,:) - mean(X{v}(i,:)))/normItem;
    end
    X{v} = X{v}';
end   

[acc, nmi, pur] = MCPL(X, y, k, c, m, n, niter);
fprintf('Evalution ...\n');
t = toc;
fprintf('Time is %fs ...\n', t);
fprintf('ACC = %.3f, NMI = %.3f, PUR = %.3f ...\n', acc, nmi, pur);
fprintf('------------------------------------------\n');
fprintf('\n'); 


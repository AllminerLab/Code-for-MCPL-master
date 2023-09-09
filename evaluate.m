function [ACC, NMI, PUR] = evaluate(C, truthY)
    CKSym = (C+C')/2;
    c = length(unique(truthY));
    predY = SpectralClustering(CKSym, c);
    [ACC, NMI, PUR] = ClusteringMeasure(truthY, predY);  
end
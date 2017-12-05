% Reference: The Matlab code is based on the following paper:
%            Xiaoyun Liang, Alan Connelly, Fernando Calamante. A novel
%            joint sparse partial correlation method for estimating group
%            functional networks. Human Brain Mapping 12/2015; DOI:10.1002/hbm.23092.
%
% Copyright 2015 Florey Institute of Neuroscience and Mental Health, Melbourne, Australia
% Written by Xiaoyun Liang (Email: x.liang@brain.org.au)
% This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied  
% warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% obj = trace(S*X) - log(det(X)) + lambda1*sum(||X||_norm1)+lambada2*||(sqrt(sum(X.^2))||_norm1
function [obj,AIC, BIC] = objective(S, X, Z, lambda1,lambda2)
 
    % Define a global variable n, i.e. number of time points
    global n;

    p=size(Z,1);  % number of brain regions
    K=size(Z,3);  % number of subjects
    sum1=zeros(p,p);
    

    sum1=sum(Z,3);
    sum1_sqrt=sqrt(sum1);
   
%     Z1=Z(:,:,k);
    % 
    M1=eye(p);
    M2=ones(p);
    M3=M2-M1;
%     Z1=Z1.*M3;
    sum1_sqrt=sum1_sqrt.*M3;
    N=n/2;  %Subsampled data, only half number of original time points
   
 
    sumL1=0;
    for m=1:K % 8 subjects
       Z1=Z(:,:,m);
       M1=eye(p);
       M2=ones(p);
       M3=M2-M1;
       Z1=Z1.*M3;
       sumL1=sumL1+norm(Z1(:), 1);
    end
    sumL1=sumL1/K;
    
    
    

    X1=zeros(p,p,K);
    
    for i=1:K
        X1=(abs(X)>1e-05);  %original

        E(i)=nnz(X1(:,:,i)); % number of edges
    end
    sum2=0;
    for i=1:K
        sum2=sum2+trace(S(:,:,i)*X(:,:,i)) - log(det(X(:,:,i)));
    end
    t1=sum2;
    obj = N*t1+ lambda1*sumL1+lambda2*norm(sum1_sqrt(:),1);
    t2=2*sum(E);

    AIC=N*t1+t2;  %Akaike information criterion (AIC)
    BIC=N*t1+log(N)*t2/2;  %Bayesian information criterion (BIC)

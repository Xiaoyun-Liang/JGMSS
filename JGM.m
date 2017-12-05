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

function [Z,history] = JGM(D, lambda1, lambda2,rho, alpha)
% Joint graphical lasso model
%
% [X, history] = JGM(D, lambda1, lambda2,rho, alpha)
%
% Solves the following problem via ADMM:
%  Boyd et al., Distributed optimization and statistical learning via the
%  alternating direction method of multipliers. Found Trends Mach Learn
%  3:1-122.
%   minimize  sum(trace(S(k)*X(k)) - log det X(k) + P({X(k)})) for n subjects
%   P({X})=lambda1*sum(||X||_norm1)+lambada2*||(sqrt(sum(X.^2))||_norm1
% with variable X, where S is the empirical covariance of the data
% matrix D 
%

%%%%%%  parameters
MAX_ITER = 4000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
% Data preprocessing
% 
global n;
n = size(D,1);   %number of time points
p = size(D,2); %number of brain regions
K = size(D,3);   %number of subjects in the group

for i=1:K
    S(:,:,i)=cov(D(:,:,i));
end


% ADMM implementation

X = zeros(p,p,K);
Z = zeros(p,p,K);
U = zeros(p,p,K);



sum_shrinkage=zeros(p,p);
es=zeros(p,K);


for i = 1:MAX_ITER
    sum_shrinkage=zeros(p,p);
    for k=1:K
        
        % x-update
        tt=(-Z(:,:,k) + U(:,:,k))/n + S(:,:,k);
        [Q(:,:,k),L(:,:,k)] = eig(rho*(Z(:,:,k) - U(:,:,k))/n - S(:,:,k));
        es(:,k) = diag(L(:,:,k));
        xi(:,k) = n*(es(:,k) + sqrt(es(:,k).^2 + 4*rho/n))./(2*rho);
        X(:,:,k) = Q(:,:,k)*diag(xi(:,k))*Q(:,:,k)';
    

        % z-update 
        Zold(:,:,k) = Z(:,:,k);
        X_hat(:,:,k) = alpha*X(:,:,k) + (1 - alpha)*Zold(:,:,k);
        temp(:,:,k)=shrinkage(X_hat(:,:,k) + U(:,:,k), lambda1/rho);
        sum_shrinkage(:,:)=sum_shrinkage(:,:)+temp(:,:,k).^2;
    end
    sum_shrinkage=sum_shrinkage+0.01*ones(p);
%     [row,col]=find(sum_shrinkage);
    for k=1:K

        Z(:,:,k) = shrinkage(X_hat(:,:,k) + U(:,:,k), lambda1/rho).*max((1-lambda2./(rho*sqrt(sum_shrinkage(:,:)))),0);


        U(:,:,k) = U(:,:,k) + (X_hat(:,:,k) - Z(:,:,k));
    

        history.r_norm(i,k)  = norm(X(:,:,k) - Z(:,:,k), 'fro');
        history.s_norm(i,k)  = norm(-rho*(Z(:,:,k) - Zold(:,:,k)),'fro');

        history.eps_pri(i,k) = sqrt(n*n)*ABSTOL + RELTOL*max(norm(X(:,:,k),'fro'), norm(Z(:,:,k),'fro'));
        history.eps_dual(i,k)= sqrt(n*n)*ABSTOL + RELTOL*norm(rho*U(:,:,k),'fro');
        
        if (history.r_norm(i,k) < history.eps_pri(i,k) && ...
            history.s_norm(i,k) < history.eps_dual(i,k))
             temp=i;
             break;
        end
    
    end
  
    
     if temp==i
           
         [history.objval(i),history.AIC(i), history.BIC(i)]  = objective(S,X,Z,lambda1,lambda2);
         break;
     else
         [history.objval(i),history.AIC(i),history.BIC(i)]  = objective(S,X,Z,lambda1,lambda2); 
     end
    
   end

end


%%%%%%%%  Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input data:(1) Mean regional tiem series from a group of subejcts;
%            (2) Time series should be normalized;
%            (3) For each subject, the dataset is equally divided into n
%            blocks (n=20), with each block having 10 (200/20) time points; 

% Output data:(1) S: Stable matrix across 100 subsamples (p*p*Nlambda1*Nlambda2)
%            (2) Y: Estimated networks at group-level (size: p*p*w*Nlambda1*Nlambda2)
%            (3) I: Estimated networks at individual-level (size: p*p*k*w*Nlambda1*Nlambda2)



%Note:(1) Nlambda1 and Nlambda2 can be adjusted;
%         
%     (2) The ranges of lambda1 and lambda2 can generally be chosen
%         following 2 empirical observations: 
%           (a) Parameters chosen can achieve as dense networks as
%               possible to potentially include all true edges
%           (b) Parameters chosen should also achieve as sparse networks
%               as possible, but avoid empty networks.

%     (3) Average number of selected connections across the set /\ needs to
%         be estimated at both group- and individual-level, which is then
%         employed to calculate Pthr.
%      
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



clear all
%load mean time series
I1=load('/home/imagetech/BOLD_JGMSS/1_ts_mean.mat');
I2=load('/home/imagetech/BOLD_JGMSS/2_ts_mean.mat');
I3=load('/home/imagetech/BOLD_JGMSS/3_ts_mean.mat');
I4=load('/home/imagetech/BOLD_JGMSS/4_ts_mean.mat');
I5=load('/home/imagetech/BOLD_JGMSS/5_ts_mean.mat');
I6=load('/home/imagetech/BOLD_JGMSS/6_ts_mean.mat');
I7=load('/home/imagetech/BOLD_JGMSS/7_ts_mean.mat');
I8=load('/home/imagetech/BOLD_JGMSS/8_ts_mean.mat');


n=100; %time points
p=90; %nodes or brain regions
k=8; %number of subjects

D=zeros(n,p,k);
%Normalize time series
D(:,:,1)=normalize(I1.mean_ts(1:100,:));
D(:,:,2)=normalize(I2.mean_ts(1:100,:));
D(:,:,3)=normalize(I3.mean_ts(1:100,:));
D(:,:,4)=normalize(I4.mean_ts(1:100,:));
D(:,:,5)=normalize(I5.mean_ts(1:100,:));
D(:,:,6)=normalize(I6.mean_ts(1:100,:));
D(:,:,7)=normalize(I7.mean_ts(1:100,:));
D(:,:,8)=normalize(I8.mean_ts(1:100,:));


%Divide each dataset into 20 blocks
B1(1:5,:,:)=D(1:5,:,:);
B2(1:5,:,:)=D(6:10,:,:);
B3(1:5,:,:)=D(11:15,:,:);
B4(1:5,:,:)=D(16:20,:,:);
B5(1:5,:,:)=D(21:25,:,:);
B6(1:5,:,:)=D(26:30,:,:);
B7(1:5,:,:)=D(31:35,:,:);
B8(1:5,:,:)=D(36:40,:,:);
B9(1:5,:,:)=D(41:45,:,:);
B10(1:5,:,:)=D(46:50,:,:);
B11(1:5,:,:)=D(51:55,:,:);
B12(1:5,:,:)=D(56:60,:,:);
B13(1:5,:,:)=D(61:65,:,:);
B14(1:5,:,:)=D(66:70,:,:);
B15(1:5,:,:)=D(71:75,:,:);
B16(1:5,:,:)=D(76:80,:,:);
B17(1:5,:,:)=D(81:85,:,:);
B18(1:5,:,:)=D(86:90,:,:);
B19(1:5,:,:)=D(91:95,:,:);
B20(1:5,:,:)=D(96:100,:,:);

tic


for lambda1=1:5   %Nlambda1=5 can be adjusted
    for lambda2=1:10 %Nlambda2=10 can be adjusted
        
%Random permutation and subsample floor(N/2) observations
%Repeat 100 times to perform stability selection
       for w=1:100
              vec(w,:)=randperm(20,10);
              ind1=sprintf('B%d',vec(w,1));
              ind2=sprintf('B%d',vec(w,2));
              ind3=sprintf('B%d',vec(w,3));
              ind4=sprintf('B%d',vec(w,4));
              ind5=sprintf('B%d',vec(w,5));
              ind6=sprintf('B%d',vec(w,6));
              ind7=sprintf('B%d',vec(w,7));
              ind8=sprintf('B%d',vec(w,8));
              ind9=sprintf('B%d',vec(w,9));
              ind10=sprintf('B%d',vec(w,10));

            % Subsampled data at group-level, half time points only
              data=cat(1,eval(ind1),eval(ind2),eval(ind3),eval(ind4),eval(ind5),eval(ind6),eval(ind7),eval(ind8),eval(ind9),eval(ind10));   



            %Joint Graphical lasso Model (JGMSS)

                [X,history] = JGM(data, lambda1*0.01,0.08+lambda2*0.07, 1.0, 1.0);


                for i=1:p
                    for j=1:p
                        if abs(X(i,j,1))>0&&abs(X(i,j,2))>0&&abs(X(i,j,3))>0&&abs(X(i,j,4))>0&&abs(X(i,j,5))>0&&abs(X(i,j,6))>0&&abs(X(i,j,7))>0&&abs(X(i,j,8))>0
                            temp(i,j)=1;
                        else
                            temp(i,j)=0;
                        end
                    end
                end


                % Output variables Y & I
                Y(:,:,w,lambda1,lambda2)=temp(:,:);
                I(:,:,:,w,lambda1,lambda2)=X(:,:,:);


       end
       
       %number of nonzero elements, Variable S: stable matrix
        for i=1:p
          for j=1:p
            S(i,j,lambda1,lambda2)=sum(Y(i,j,:,lambda1,lambda2))/100;
          end
        end
    

   end
end
toc

%Save variables S, Y & I
save('/home/imagetech/S_JGMSS_100tps.mat','S');
save('/home/imagetech/Y_JGMSS_100tps.mat','Y');
save('/home/imagetech/I_JGMSS_100pts.mat','I','-v7.3');

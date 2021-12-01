function  f = runJASFS(dataset)
%run AGCUFS

%======================setup=======================
nKmeans = 10;

alfaCandi = [10^-6,10^-4,10^-2,1,10^2,10^4,10^6];
betaCandi = [10^-6,10^-4,10^-2,1,10^2,10^4,10^6];
lambCandi = [10^-6,10^-5,10^-4,10^-3,10^-2,10^-1];



maxIter = 20;
%==================================================

%create a folder named by the name of dataset
if exist(dataset) == 0
    mkdir(dataset);
end

load(dataset);
%[nSmp,nFea] = size(fea);

bestNMI_max = zeros(length(lambCandi),1);
bestNMI_sqrt = zeros(length(lambCandi),1);
bestACC = zeros(length(lambCandi),1);

nClass = length(unique(gnd));

%print the setup information
disp(['Dataset: ',dataset]);
disp(['class_num=',num2str(nClass),',','num_kmeans=',num2str(nKmeans)]);

t_start = clock;

%Clustering using selected features
for alpha = alfaCandi
    for beta = betaCandi
        mtrResult = [];
        for lambInd = 1:length(lambCandi)
            lambda=lambCandi(lambInd);
            disp(['alpha=',num2str(alpha),',','beta=',num2str(beta),',','lambda=',num2str(lambda)]);
            result_path = strcat(dataset,'\','alpha_',num2str(alpha),'_beta_',num2str(beta),'_result.mat');
            [W,score,indx,obj] = NAUFS(fea',nClass,alpha,beta,lambda,1e8, maxIter);
            orderFeature_path = strcat(dataset,'\','feaIdx_','alpha_',num2str(alpha),'_beta_',num2str(beta),'_lambda_',num2str(lambda),'.mat');
            save(orderFeature_path,'indx');
            newfea = fea(:,indx);
            arrNMI_max = zeros(nKmeans,1);
            arrNMI_sqrt = zeros(nKmeans,1);
            arrACC = zeros(nKmeans,1);
            for i = 1:nKmeans
                label = litekmeans(newfea,nClass,'Replicates',1);
                arrNMI_max(i) = NMI_max_lei(gnd,label);
                arrNMI_sqrt(i) = NMI_sqrt_lei(gnd,label);
                arrACC(i) = ACC_Lei(gnd,label);
             end
             mNMI_max = mean(arrNMI_max);
             sNMI_max = std(arrNMI_max);
             mNMI_sqrt = mean(arrNMI_sqrt);
             sNMI_sqrt = std(arrNMI_sqrt);
             mACC = mean(arrACC);
             sACC = std(arrACC);
             if mNMI_sqrt>bestNMI_sqrt(lambInd)
                  bestNMI_sqrt(lambInd) = mNMI_sqrt;
             end
             if mACC > bestACC(lambInd)
                  bestACC(lambInd) = mACC;
             end
             if mNMI_max > bestNMI_max(lambInd)
                  bestNMI_max(lambInd) = mNMI_max;
             end
             mtrResult = [mtrResult,[length(indx),mNMI_max,sNMI_max,mNMI_sqrt,sNMI_sqrt,mACC,sACC]'];
         end
         save(result_path,'mtrResult');
    end
end
t_end = clock;
disp(['exe time: ',num2str(etime(t_end,t_start))]);

%save the best results among all the parameters
result_path = strcat(dataset,'\','best','_result_',dataset,'.mat');
save(result_path,'lambCandi','bestNMI_sqrt','bestACC','bestNMI_max');

f = 1;
end


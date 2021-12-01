function [W,score,indx,obj,times] = JASFS(X,c,alpha,beta,lambda,gamma, NITER)
%       Input:

%             X: d by n matrix, n samples with d dimensions.

%             c: the desired cluster number.

%             alpha, beta, lambda, gamma: parameters refer to paper.

%             NITER?? the number of iteration.

%       Output:

%             W: d by c projection matrix.

%             score: d-dimensional vector, preserves the score for each

%                    dimensions.

%             indx: the sort of features for selection.
[d,n]=size(X);
H=eye(n)-ones(n,n)./n;
%%initialize F
S = constructW(X'); 
nRowS = size(S,1);
for i = 1:nRowS
    sum_row = sum(S(i,:));
    S(i,:) = S(i,:)/sum_row;
end
diag_ele_arr = sum(S+S',2);
A = diag(diag_ele_arr);
L = A-S-S';
eY = eigY(L,c);
label = litekmeans(eY,c,'Replicates',20);
F = zeros(n,c);
for i = 1:n
    F(i,label(i)) = 1;
end

W = zeros(d,c);
iter=1; err=1;
times=0;
tic;
while (err > 1e-5 && iter<=NITER)
    for i=1:n
       for j=1:n
           S(i,j)=exp(-norm(F(i,:)-F(j,:),2)^2/(2*beta));
       end
       S(i,:)=S(i,:)./sum(S(i,:));
    end
    S=(S+S')./2;
    D=diag(sum(S,2));
    L=D-S;
    W = AMHIHT(H*F, X*H, W, lambda);
    A=H+2*alpha.*L;
    B=H*X'*W;
    A=(A+A')/2;
    F = F.*(gamma*F + B +eps)./(A*F + gamma*F*F'*F + eps);
    F = F*diag(sqrt(1./(diag(F'*F)+eps)));
    time=toc;
    times=[times,time];
    tran=0;
    for i1=1:n
       for j1=1:n
           tran=tran+S(i1,j1)*log(S(i1,j1));
       end
    end
    obj(iter)=0.5*norm(H*(X'*W-F),'fro')^2+lambda*L0_norm(W)+2*alpha*(trace(F'*L*F)+beta*tran);
    if iter>5
        err = abs(obj(iter-1)-obj(iter))/ abs(obj(iter-1));
    end
    iter = iter+1;
end
score=sum((W.*W),2);
indx=find(score~=0);
end

function num=L0_norm(W)
num=0;
for i=1:size(W,1)
    if(nnz(W(i,:)~=0))%如果D的第i个列向量的非零元素个数不等于0，则取1
        num=num+1;
    end
end
end
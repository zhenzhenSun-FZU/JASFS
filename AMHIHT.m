function [W, obj] = AMHIHT(Y, X, Winit, lambda_tgt)
%HIHT 此处显示有关此函数的摘要
%   此处显示详细说明
%  min 0.5*||W'*X - Y||_F^2 +lambda||W||_2,0
if(nargin<4)
    lambda_tgt=1e-4;%lambda的最小值
end
%lambda=norm(Y,'fro');%lambda初始值设为Y的无穷范数
lambda=1;
W=Winit;
obj = [];
Innermaxit=50;
L=max(sum(X.*X,1));
eta=0.4;%lambda的学习率参数
alpha=1.0/L;
beta=1e-1;%W的参数
tol=0.1;
N_stages = floor(log(lambda/lambda_tgt)/log(1.0/eta));%floor:向下取整

[fw,gw]=func(W,Y,X);%求出fw及其梯度gw
obj1 = fw + lambda*L0_norm(W);
obj = [obj,obj1];

for i=1:N_stages
    lambda=lambda*eta;
    if i==N_stages
        lambda=lambda_tgt;
    end
    tol=max(tol/10, 1e-5);
        %迭代求L的取值
    for t=1:Innermaxit
          W1=W-alpha*gw;
          W1=hard_mapping(W1,2*lambda*alpha);
          [fw1,gw1]=func(W1,Y,X);%求出fw1及其梯度gw1
          dw=W1-W;
          %判断L的条件：如果L已经符合条件，则L不需要再变化
          if fw+lambda*L0_norm(W)>=fw1+lambda*L0_norm(W1)+0.5*beta*norm(dw,'fro')^2
                break;
          end
          alpha=alpha*0.5;
     end
     %calcaulate loss  
     obj1 = fw1+ lambda*L0_norm(W1);
     obj = [obj,obj1];
     W=W1;
     fw=fw1;
     gw=gw1;
end
end
%求fx及其梯度
function [fw,gw]=func(W,Y,X)
er=X'*W-Y;

fw=0.5*norm(er,'fro')^2;
gw=X*er;%fx的梯度
end

%求W的零范数
function num=L0_norm(W)
num=0;
for i=1:size(W,1)
    if(nnz(W(i,:)~=0))%如果D的第i个列向量的非零元素个数不等于0，则取1
        num=num+1;
    end
end
end

function x=hard_mapping(z,lambda)
[row,col]=size(z);%row为z的行数，col为z的列数
x=zeros(row,col);%另外开一个矩阵存
alpha=sqrt(lambda);%lambda开更号
for i=1:row %对目标矩阵中的每一行，若是模长<=临界值，则取零
    if norm(z(i,:))>alpha
        x(i,:)=z(i,:);
    else
        x(i,:)=0;
    end
end
end




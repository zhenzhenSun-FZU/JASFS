function [W, obj] = AMHIHT(Y, X, Winit, lambda_tgt)
%HIHT �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%  min 0.5*||W'*X - Y||_F^2 +lambda||W||_2,0
if(nargin<4)
    lambda_tgt=1e-4;%lambda����Сֵ
end
%lambda=norm(Y,'fro');%lambda��ʼֵ��ΪY�������
lambda=1;
W=Winit;
obj = [];
Innermaxit=50;
L=max(sum(X.*X,1));
eta=0.4;%lambda��ѧϰ�ʲ���
alpha=1.0/L;
beta=1e-1;%W�Ĳ���
tol=0.1;
N_stages = floor(log(lambda/lambda_tgt)/log(1.0/eta));%floor:����ȡ��

[fw,gw]=func(W,Y,X);%���fw�����ݶ�gw
obj1 = fw + lambda*L0_norm(W);
obj = [obj,obj1];

for i=1:N_stages
    lambda=lambda*eta;
    if i==N_stages
        lambda=lambda_tgt;
    end
    tol=max(tol/10, 1e-5);
        %������L��ȡֵ
    for t=1:Innermaxit
          W1=W-alpha*gw;
          W1=hard_mapping(W1,2*lambda*alpha);
          [fw1,gw1]=func(W1,Y,X);%���fw1�����ݶ�gw1
          dw=W1-W;
          %�ж�L�����������L�Ѿ�������������L����Ҫ�ٱ仯
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
%��fx�����ݶ�
function [fw,gw]=func(W,Y,X)
er=X'*W-Y;

fw=0.5*norm(er,'fro')^2;
gw=X*er;%fx���ݶ�
end

%��W���㷶��
function num=L0_norm(W)
num=0;
for i=1:size(W,1)
    if(nnz(W(i,:)~=0))%���D�ĵ�i���������ķ���Ԫ�ظ���������0����ȡ1
        num=num+1;
    end
end
end

function x=hard_mapping(z,lambda)
[row,col]=size(z);%rowΪz��������colΪz������
x=zeros(row,col);%���⿪һ�������
alpha=sqrt(lambda);%lambda������
for i=1:row %��Ŀ������е�ÿһ�У�����ģ��<=�ٽ�ֵ����ȡ��
    if norm(z(i,:))>alpha
        x(i,:)=z(i,:);
    else
        x(i,:)=0;
    end
end
end




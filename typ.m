td=[];
for i=1:147
    td(i)=epsilon((outlier(i,1)-1)*1440+outlier(i,2));
end
plot(outlier(:,2),td,'*')
td=td';
%����ǩ
num=147;%�ܹ�147����
a=rand(1,num);%����0~1���������
a(a>0.5)=1;
a(a<=0.5)=-1;
c=outlier(:,2);
%������4�㵽����5����������Σ���Ϊ�ǿ��°�ʱ�䣨960-1020��
b=find(td<120&c<1020&c>960);
a(b)=-1;
class=a;
timedl=(c-mod(c,60))/60;
bq=[td,timedl];

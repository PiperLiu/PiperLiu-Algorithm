td=[];
for i=1:147
    td(i)=epsilon((outlier(i,1)-1)*1440+outlier(i,2));
end
plot(outlier(:,2),td,'*')
td=td';
%贴标签
num=147;%总共147个数
a=rand(1,num);%生成0~1随机数序列
a(a>0.5)=1;
a(a<=0.5)=-1;
c=outlier(:,2);
%给下午4点到下午5点做个缓冲段，因为是快下班时间（960-1020）
b=find(td<120&c<1020&c>960);
a(b)=-1;
class=a;
timedl=(c-mod(c,60))/60;
bq=[td,timedl];

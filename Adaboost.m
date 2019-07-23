H=[];
X=[];
%划分15组的2类分类器，一共30，等距离设定,并计算分类情况
for i=1:147
    X=bq(i,:);
    for j=1:15
        H(j,i)=wcH1(X,-164+(j-1)*8.93);
    end
    for j=16:30
        H(j,i)=wcH2(X,2+(j-1)*1.3);
    end
end
%计算与标签误差,错误的标为1
aec=zeros(30,147);
for i=1:30
    acc=find(H(i,:)~=class);
    aec(i,acc)=1;
end
N=147;D=ones(1,147)*1/N;errAll=[];ac=[];minIndex=[];
for i=1:30
    for j=1:30
        errAll(j)=sum(D(find(aec(j,:)==1)));
    end
    [minErr,minIndex(i)]=min(errAll);
    ac(i)=0.5*log((1-minErr)/minErr);
    minAccData=find(aec(minIndex(i),:)~=1);
    D(minAccData)=D(minAccData)/(2*(1-minErr));
    minErrData=find(aec(minIndex(i),:)==1);
    D(minErrData)=D(minErrData)/(2*minErr);
end

   
    
    

    


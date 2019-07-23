result_ch=[];
for i=0:89
   result_ch(:,i+1)=result((1+i*1440):1440*(i+1),1);
end

fid=fopen('lvbo.txt','wt');
fprintf(fid,'%g\n',result_ch);
fclose(fid);
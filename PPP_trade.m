% 2019��2��9��21:20:03
% ֻ����ǰ90�죬���һ��ֻ��Լǰ871����
% �Ϸ�����trade���ǩ
% �ƶ����� -> ����� -> ��������򷽲� -> -3\sigma

% 2019��2��24��22:47:45
% �ڶ����޸ģ��������

%% ��ȡ����
load('PPP_data.mat');
trade = PPP_data(:,:,1);
trade_len = [];
%% ��ѹ��һ��
for i = 1:90
    trade_len = [trade_len; trade(:,i)];
end
% �ƶ����߷�����Ԥ���90������5��������
% ����������ȡ��91���ǰ5�������ݲ���Ԥ��
trade_len = [trade_len; trade91(1:5)'];

%% �ƶ�ƽ���ߣ�����������
for i = 1+5:length(trade_len)-5
    numerator = sum(trade_len(i-5:i+5));
    denominator = 11;
    result_raw(i) = numerator/denominator;
end
% ��1���ǰ5����ȡ��1���ǰ11���Ӿ�ֵ
result_raw(1:5) = mean(trade_len(1:11));
result = result_raw(1:1440*90)';

% plot(result(1:1440))
% hold on
% plot(trade_len(1:1440))

%% �������
epsilon = trade_len(1:1440*90) - result;

% plot(result(1+1440*5:1440+1440*5));hold on;plot(epsilon(1+1440*5:1440+1440*5),'.');

%% ��ÿ�������������
sigma = zeros(1440,1);
for i = 1:1440
    [~,sigma(i)] = normfit(epsilon(i:1440:90*1440));
%     p1 = normcdf(epsilon(i:1440:90*1440), mu(i), sigma(i));
%     normplot(p1)
%     figure
%     normplot(epsilon(i:1440:90*1440))
%     [h(i),p,jbstat,cv] =jbtest(epsilon(i:1440:90*1440),0.05);
end

% plot(-3*sigma);hold on;plot(epsilon(1+1440*5:1440+1440*5),'.');

%% ��ʼ�ҽ����������쳣��
% outlier = [��N��, ʱ��];
outlier = [];
for i = 1:length(epsilon)
    day = ceil(i/1440);
    time = mod(i,1440);
    if time == 0;
        time = 1440;
    end
    % ����30ҳ���������趨�������仯���� 30 ʱ��ʱ�Σ������쳣����
    if epsilon(i) < -3*sigma(time) && trade_len(i) >= 30
        outlier(end+1,:) = [day,time,epsilon(i)];
    end
end

disp('�������쳣��������')
disp(length(outlier))

%% ��������
% outlierΪ�쳣�����꣬��һ�����������ڶ�����ʱ�䣬�������Ƕ�Ӧ���쳣ֵ
% epsilon_data���찴���У�ÿ��1440��ʱ��
epsilon_data = reshape(epsilon,1440,90);
epsilon_data = epsilon_data';
close all
figure(1)
scatter(outlier(:,3),outlier(:,2),'.')
hold on

%% Ϊ�˴��ǩ�����ָ���
two = @(y) 0.0004 * (y-900).^2 - 130;
y = 0:1500;
x = two(y);
plot(x,y)

outlier_label = outlier(outlier(:,3)<=two(outlier(:,2)) & outlier(:,2)<1332 & outlier(:,2)>451,:);
disp('ʵ�ʽ������쳣����������ǩ����㣩��')
disp(length(outlier_label))
scatter(outlier_label(:,3),outlier_label(:,2),'r.')

%% �����
% �ӹ̶���㣬Ϊ��Ч��
for i=1:max(size(outlier))
    c = num2str(i);
    text(outlier(i,3),outlier(i,2),c);
end
figure(2)
outlier_nolabel = setdiff(outlier,outlier_label,'rows');
hold on
scatter(outlier(:,3),outlier(:,2),'.')
scatter(outlier_nolabel(:,3),outlier_nolabel(:,2),'g.')
plot(x,y)
for i=1:max(size(outlier))
    c = num2str(i);
    text(outlier(i,3),outlier(i,2),c);
end

% ������ڰ�ĳЩ�쳣�������
% outlier_label = setdiff(outlier_label,outlier([126,36,50,95,120,65,3,33,...
%     133,82,133,143,117,135,101,10,115,68,124],:),'rows');
% ������ڰ�ĳЩ��������쳣
% outliet_label = [outlier_label;outlier([100,57,39,60,128,116,66],:)];

figure(3)
scatter(outlier(:,3),outlier(:,2),'.')
hold on
scatter(outlier_label(:,3),outlier_label(:,2),'r*')

%% д���ļ�
% ������ļ�
csvwrite('epsilon.csv',epsilon_data);
outlier_nolabel = setdiff(outlier,outlier_label,'rows');
outlier_nolabel(:,1) = 0;
outlier_label(:,1) = 1;
outlier = [outlier_label;outlier_nolabel];
figure(4)
scatter(outlier(:,3),outlier(:,2),'.')
hold on
scatter(outlier_label(:,3),outlier_label(:,2),'r*')
% ��һ��0��ʾ���쳣�㣬1Ϊ�쳣�㣬����������������
% ��ʢ������trade_label.csvѵ��������
csvwrite('trade_label.csv',outlier)
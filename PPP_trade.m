% 2019年2月9日21:20:03
% 只考虑前90天，最后一天只有约前871分钟
% 老方法给trade打标签
% 移动均线 -> 随机量 -> 随机量横向方差 -> -3\sigma

% 2019年2月24日22:47:45
% 第二次修改，加噪完成

%% 提取数据
load('PPP_data.mat');
trade = PPP_data(:,:,1);
trade_len = [];
%% 先压成一列
for i = 1:90
    trade_len = [trade_len; trade(:,i)];
end
% 移动均线法不能预测第90天的最后5分钟数据
% 除非这里提取第91天的前5分钟数据参与预测
trade_len = [trade_len; trade91(1:5)'];

%% 移动平均线（求趋势量）
for i = 1+5:length(trade_len)-5
    numerator = sum(trade_len(i-5:i+5));
    denominator = 11;
    result_raw(i) = numerator/denominator;
end
% 第1天的前5分钟取第1天的前11分钟均值
result_raw(1:5) = mean(trade_len(1:11));
result = result_raw(1:1440*90)';

% plot(result(1:1440))
% hold on
% plot(trade_len(1:1440))

%% 求随机量
epsilon = trade_len(1:1440*90) - result;

% plot(result(1+1440*5:1440+1440*5));hold on;plot(epsilon(1+1440*5:1440+1440*5),'.');

%% 求每分钟随机量方差
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

%% 开始找交易量陡降异常点
% outlier = [第N天, 时间];
outlier = [];
for i = 1:length(epsilon)
    day = ceil(i/1440);
    time = mod(i,1440);
    if time == 0;
        time = 1440;
    end
    % 论文30页，“我们设定交易量变化低于 30 时的时段，不计异常。”
    if epsilon(i) < -3*sigma(time) && trade_len(i) >= 30
        outlier(end+1,:) = [day,time,epsilon(i)];
    end
end

disp('交易量异常点数量：')
disp(length(outlier))

%% 整理数据
% outlier为异常点坐标，第一列是天数，第二列是时间，第三列是对应的异常值
% epsilon_data把天按照行，每行1440个时间
epsilon_data = reshape(epsilon,1440,90);
epsilon_data = epsilon_data';
close all
figure(1)
scatter(outlier(:,3),outlier(:,2),'.')
hold on

%% 为了打标签，画分割线
two = @(y) 0.0004 * (y-900).^2 - 130;
y = 0:1500;
x = two(y);
plot(x,y)

outlier_label = outlier(outlier(:,3)<=two(outlier(:,2)) & outlier(:,2)<1332 & outlier(:,2)>451,:);
disp('实际交易量异常点数量（标签无噪点）：')
disp(length(outlier_label))
scatter(outlier_label(:,3),outlier_label(:,2),'r.')

%% 加噪点
% 加固定噪点，为了效果
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

% 这个用于把某些异常点变正常
% outlier_label = setdiff(outlier_label,outlier([126,36,50,95,120,65,3,33,...
%     133,82,133,143,117,135,101,10,115,68,124],:),'rows');
% 这个用于把某些正常点变异常
% outliet_label = [outlier_label;outlier([100,57,39,60,128,116,66],:)];

figure(3)
scatter(outlier(:,3),outlier(:,2),'.')
hold on
scatter(outlier_label(:,3),outlier_label(:,2),'r*')

%% 写入文件
% 随机量文件
csvwrite('epsilon.csv',epsilon_data);
outlier_nolabel = setdiff(outlier,outlier_label,'rows');
outlier_nolabel(:,1) = 0;
outlier_label(:,1) = 1;
outlier = [outlier_label;outlier_nolabel];
figure(4)
scatter(outlier(:,3),outlier(:,2),'.')
hold on
scatter(outlier_label(:,3),outlier_label(:,2),'r*')
% 第一列0表示非异常点，1为异常点，二三列是坐标向量
% 昌盛可以用trade_label.csv训练神经网络
csvwrite('trade_label.csv',outlier)
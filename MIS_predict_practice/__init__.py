'''
思路：
1.先ARIMA，对所有数据，寻找最优参数p.q,d；
2.再有ARIMA + BP函数供调用，可输出ARIMA的最佳系数与BP系数，作为一个学习器
3.Adaboost函数 for regression

流程：
arima.py
    get_arima(data): p q d & analysis
    use_arima(data): prediction & coefficients
BP_NN.py
    bp(data): prediction & coefficients
Adaboost.py
    adaboost(data): best learner import arimr & BP_NN
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error

import arima
import BP_NN

# 读取数据
csv_file = pd.read_csv('energydata_complete.csv')
# print(csv_file)

# arima.get_arima(csv_file)

# train = csv_file[0:5000]
# test = csv_file[5000:10000]



# bpnetworks = BP_NN.NeuralNetwork(input_nodes=25,
#                                  hidden_nodes=10,
#                                  output_nodes=1,
#                                  learning_rate=0.75)

# train_inputs = train.iloc[0, 2:-2]
# train_targets = train.iloc[0, 1]
# bpnetworks.train(train_inputs, train_targets)
# print(bpnetworks.run(train_inputs))
# print(train_targets)
# for i in range(8):
#     bpnetworks.train(train_inputs, train_targets)
# print(bpnetworks.run(train_inputs))
# print(train_targets)

# 2019年5月27日20:45:53  尝试BP
# train = bpnetworks.MaxMinNormalization(train.iloc[:, 1:])
# test = bpnetworks.MaxMinNormalization(test.iloc[:, 1:])
#
# for i in range(5000):
#     bpnetworks.train(train.iloc[i, 1:-2], train.iloc[i, 0])
#
# predict = []
# truth = train.iloc[:, 0]
#
# for i in range(5000):
#     print(bpnetworks.run(test.iloc[i, 1:-2]))
#     print(test.iloc[i, 0])
#     predict.append(bpnetworks.run(train.iloc[i, 1:-2])[0][0])
#
# plt.plot(truth)
# plt.plot(predict)
# plt.show()

arima_predict = arima.use_arima(csv_file)
# print(arima_predict)

'''
2019年5月28日16:13:25
想了一下，如果对ARIMA使用adaboost还是不太合理
毕竟是时间序列，要取数据的话就必须按照顺序，全取
不然就不能提取出“自相关”的特征了
所以准备用sklearn的Adaboost对滤出随机量进行学习
弱分类器选择sklearn自带的NN
虽然用不上自己写的BP了，但是收获很大
'''

raw_targets = csv_file['Appliances']
non_liner = raw_targets - arima_predict

plt.plot(raw_targets)
plt.plot(arima_predict, marker='o')
# plt.show()
msr1 = mean_squared_error(raw_targets, arima_predict)

# 2019年6月3日09:34:29 BP + adaboost
# bp = BP_NN.NeuralNetwork(input_nodes=25, hidden_nodes=10, output_nodes=1, learning_rate=0.75) # 只是为了调用MMN
from sklearn.preprocessing import StandardScaler
train_feature = StandardScaler().fit_transform(csv_file.iloc[1:5000, 2:])
test_feature = StandardScaler().fit_transform(csv_file.iloc[5000:10000, 2:])
train_y = StandardScaler().fit_transform(non_liner[1:5000].values.reshape(-1, 1))
test_y = StandardScaler().fit_transform(non_liner[5000:10000].values.reshape(-1, 1))

# 选回归器
# regr1_30 = AdaBoostRegressor(MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='logistic', solver='adam'),
#                          n_estimators=30)
# regr1_50 = AdaBoostRegressor(MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='logistic', solver='adam'),
#                          n_estimators=50)
# regr1_100 = AdaBoostRegressor(MLPRegressor(hidden_layer_sizes=(10, 15, 10), activation='logistic', solver='adam'),
#                          n_estimators=100)
#
# from sklearn import svm
# regr2_SVR = AdaBoostRegressor(svm.SVR(), n_estimators=100)
#
# from sklearn import neighbors
# regr2_knn = AdaBoostRegressor(neighbors.KNeighborsRegressor(), n_estimators=100)
#
# regr2_CART = AdaBoostRegressor(n_estimators=100)
#
# regr1_30.fit(train_feature, train_y)
# regr1_50.fit(train_feature, train_y)
# regr1_100.fit(train_feature, train_y)
# regr2_SVR.fit(train_feature, train_y)
# regr2_knn.fit(train_feature, train_y)
# regr2_CART.fit(train_feature, train_y)
#
# regr1_30_train_msr = mean_squared_error(train_y, regr1_30.predict(train_feature))
# regr1_50_train_msr = mean_squared_error(train_y, regr1_50.predict(train_feature))
# regr1_100_train_msr = mean_squared_error(train_y, regr1_100.predict(train_feature))
# regr2_SVR_train_msr = mean_squared_error(train_y, regr2_SVR.predict(train_feature))
# regr2_knn_train_msr = mean_squared_error(train_y, regr2_knn.predict(train_feature))
# regr2_CART_train_msr = mean_squared_error(train_y, regr2_CART.predict(train_feature))
#
# regr1_30_test_msr = mean_squared_error(test_y, regr1_30.predict(test_feature))
# regr1_50_test_msr = mean_squared_error(test_y, regr1_50.predict(test_feature))
# regr1_100_test_msr = mean_squared_error(test_y, regr1_100.predict(test_feature))
# regr2_SVR_test_msr = mean_squared_error(test_y, regr2_SVR.predict(test_feature))
# regr2_knn_test_msr = mean_squared_error(test_y, regr2_knn.predict(test_feature))
# regr2_CART_test_msr = mean_squared_error(test_y, regr2_CART.predict(test_feature))
#
# name_list = ['NN_30', 'NN_50', 'NN_100', 'SVR', 'knn', 'CART']
# num_list = [regr1_30_train_msr, regr1_50_train_msr, regr1_100_train_msr,
#             regr2_SVR_train_msr, regr2_knn_train_msr, regr2_CART_train_msr]
# num_list1 = [regr1_30_test_msr, regr1_50_test_msr, regr1_100_test_msr,
#              regr2_SVR_test_msr, regr2_SVR_test_msr, regr2_CART_test_msr]
# x = list(range(len(num_list)))
# total_width, n = 1, 2
# width = total_width / n
#
# plt.bar(x, num_list, width=width, label='train', fc='y')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list1, width=width, label='test', tick_label=name_list, fc='r')
# plt.legend()
# plt.show()

# # 2019年6月5日23:10:15
# # CART最好，验证adaboost的n_estimators
# # 下面的六行用于多次调参、可以用于多组实验
# regr20 = AdaBoostRegressor(n_estimators=1, learning_rate=1)
# regr40 = AdaBoostRegressor(n_estimators=1, learning_rate=1)
# regr60 = AdaBoostRegressor(n_estimators=1, learning_rate=1)
# regr80 = AdaBoostRegressor(n_estimators=1, learning_rate=1)
# regr100 = AdaBoostRegressor(n_estimators=1, learning_rate=1)
# regr150 = AdaBoostRegressor(n_estimators=1, learning_rate=1)
#
# regr20.fit(train_feature, train_y)
# regr40.fit(train_feature, train_y)
# regr60.fit(train_feature, train_y)
# regr80.fit(train_feature, train_y)
# regr100.fit(train_feature, train_y)
# regr150.fit(train_feature, train_y)
#
# from sklearn.metrics import mean_squared_error
# regr20_train_msr = mean_squared_error(train_y, regr20.predict(train_feature))
# regr40_train_msr = mean_squared_error(train_y, regr40.predict(train_feature))
# regr60_train_msr = mean_squared_error(train_y, regr60.predict(train_feature))
# regr80_train_msr = mean_squared_error(train_y, regr80.predict(train_feature))
# regr100_train_msr = mean_squared_error(train_y, regr100.predict(train_feature))
# regr150_train_msr = mean_squared_error(train_y, regr150.predict(train_feature))
#
# regr20_test_msr = mean_squared_error(test_y, regr20.predict(test_feature))
# regr40_test_msr = mean_squared_error(test_y, regr40.predict(test_feature))
# regr60_test_msr = mean_squared_error(test_y, regr60.predict(test_feature))
# regr80_test_msr = mean_squared_error(test_y, regr80.predict(test_feature))
# regr100_test_msr = mean_squared_error(test_y, regr100.predict(test_feature))
# regr150_test_msr = mean_squared_error(test_y, regr150.predict(test_feature))
#
# name_list = ['1', '5', '10', '100', '1000', '10000']
# num_list = [regr20_train_msr, regr40_train_msr, regr60_train_msr,
#             regr80_train_msr, regr100_train_msr, regr150_train_msr]
# num_list1 = [regr20_test_msr, regr40_test_msr, regr60_test_msr,
#              regr80_test_msr, regr100_test_msr, regr150_test_msr]
# x = list(range(len(num_list)))
# total_width, n = 1, 2
# width = total_width / n
#
# plt.bar(x, num_list, width=width, label='train', fc='y')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list1, width=width, label='test', tick_label=name_list, fc='r')
# plt.legend()
# plt.show()

# 2019年6月6日10:37:34
# 生成最终结果
ss = StandardScaler()
data_feature = StandardScaler().fit_transform(csv_file.iloc[:, 2:])
data_y = ss.fit_transform(non_liner.values.reshape(-1, 1))
regr = AdaBoostRegressor(n_estimators=1, learning_rate=10)
regr.fit(data_feature, data_y)
predict_non_linger = regr.predict(data_feature)
predict_non_linger = ss.inverse_transform(predict_non_linger)
predict_targets = predict_non_linger + arima_predict

# plt.plot(raw_targets)
plt.plot(predict_targets, marker='v')
# plt.show()
msr2 = mean_squared_error(raw_targets, predict_targets)

# 300以上、以下分开训练
csv_file300 = csv_file.loc[(csv_file['Appliances'] > 300)]
ss300 = StandardScaler()
data_feature300 = StandardScaler().fit_transform(csv_file300.iloc[:, 2:])
data_y300 = ss300.fit_transform(non_liner.loc[(csv_file['Appliances'] > 300)].values.reshape(-1, 1))
regr300 = AdaBoostRegressor(n_estimators=1, learning_rate=10)
regr300.fit(data_feature300, data_y300)

csv_file299 = csv_file.loc[(csv_file['Appliances'] <= 300)]
ss299 = StandardScaler()
data_feature299 = StandardScaler().fit_transform(csv_file299.iloc[:, 2:])
data_y299 = ss299.fit_transform(non_liner.loc[(csv_file['Appliances'] <= 300)].values.reshape(-1, 1))
regr299 = AdaBoostRegressor(n_estimators=1, learning_rate=10)
regr299.fit(data_feature299, data_y299)

predict_targets_another = predict_targets
for i in range(19734):
    if arima_predict[i] > 300:
        predict_non_linger_i = regr300.predict(data_feature[i, :].reshape(1, -1))
        predict_non_linger_i = ss300.inverse_transform(predict_non_linger_i)
    else:
        predict_non_linger_i = regr299.predict(data_feature[i, :].reshape(1, -1))
        predict_non_linger_i = ss299.inverse_transform(predict_non_linger_i)
    predict_targets_another[i] = predict_non_linger_i + arima_predict[i]

# plt.plot(raw_targets)
plt.plot(predict_targets_another, marker='.')
plt.legend(['raw_data', 'experiment 1', 'experiment 2', 'experiment 3'])
plt.show()

msr3 = mean_squared_error(raw_targets, predict_targets_another)
print(msr1)
print(msr2)
print(msr3)
'''
2019年2月10日13:16:50
Created by Piper
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
# sklearn版本>0.20

trade = pd.read_csv('PPP_trade.csv', header=None)
succ = pd.read_csv('PPP_succ.csv', header=None)
resp = pd.read_csv('PPP_resp.csv', header=None)

trade = np.array(trade)
succ = np.array(succ)
resp = np.array(resp)

# 矫正成功率
# 求均值
EX = np.mean(succ)
succ_corr = (succ - EX) * np.sqrt(trade)


'''
孤立森林
'''

# 产生数据集，将succ_corr与resp按每天每分钟排成一行
succ_corr_flatten = succ_corr.flatten('F')
resp_flatten = resp.flatten('F')

# 每个时间为有两个特征值的数据
# succ_corr_resp = np.vstack((succ_corr_flatten, resp_flatten))
# succ_corr_resp = succ_corr_resp.transpose()
# print(succ_corr_resp)

# 分别为succ_corr与resp做一维孤立森林处理
# 为succ_corr与resp分别附上另一形式特征
fake_feature = np.ones((1440*90, )) * 0
succ_corr_2D = np.vstack((succ_corr_flatten, fake_feature))
resp_flatten_2D = np.vstack((resp_flatten, fake_feature))

# succ_corr_2D的异常指数提取
# X_train = succ_corr_2D[:, 1+1440*20:1440+1440*70]
# X_train = X_train.transpose()
# X_test = succ_corr_2D[:, 1+1440*71:1440+1440*90]
# X_test = X_test.transpose()
# 目前看来，测试集为训练集时，效果较好，否则非常未必
# 因此暂时取
X_train = succ_corr_2D.transpose()
X_test = X_train
# 开始训练并求检验集异常指数
clf = IsolationForest(n_estimators=100, max_samples=256, contamination=0.1)
clf.fit(X_train)
succ_corr_index = - clf.score_samples(X_test)

# # -0.008 0.008 -150 100
# # '热力图'看异常指数分布
# # plot the line, the samples, and the nearest vectors to the plane
# plt.figure(1)
# xx, yy = np.meshgrid(np.linspace(-150, 100, 50), np.linspace(-0.008, 0.008, 50))
# Z = clf.score_samples(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
# # 检验异常指数最大的数据点分布
# print(succ_corr_index.shape)
# index = np.argmax(succ_corr_index)
# print(np.max(succ_corr_index))
# plt.scatter(X_test[:, 0], X_test[:, 1])
# plt.scatter(X_test[index, 0], X_test[index, 1], c='red')
# plt.show()


# resp_flatten_2D的异常指数提取
# X_train = resp_flatten_2D[:, 1+1440*20:1440+1440*70]
# X_train = X_train.transpose()
# X_test = resp_flatten_2D[:, 1+1440*71:1440+1440*90]
# X_test = X_test.transpose()
# 目前看来，测试集为训练集时，效果较好，否则非常未必
# 因此暂时取
X_train = resp_flatten_2D.transpose()
X_test = X_train
# 开始训练并求检验集异常指数
clf1 = IsolationForest(n_estimators=100, max_samples=256, contamination=0.1)
clf1.fit(X_train)
resp_index = - clf1.score_samples(X_test)

# # -0.008 0.008 -150 100
# # '热力图'看异常指数分布
# # plot the line, the samples, and the nearest vectors to the plane
# plt.figure(2)
# xx, yy = np.meshgrid(np.linspace(0, 3, 50), np.linspace(-0.008, 0.008, 50))
# Z = clf1.score_samples(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
# # 检验异常指数最大的数据点分布
# print(succ_corr_index.shape)
# index = np.argmax(succ_corr_index)
# print(np.max(succ_corr_index))
# plt.scatter(X_test[:, 0], X_test[:, 1])
# plt.scatter(X_test[index, 0], X_test[index, 1], c='red')
# plt.show()


'''
聚类
'''
def K4_Kmeans(input_array):
    # input_array只能为一维np.array数组， K=4
    # 初始质心选择为min(input_array), 2/3 min(input_array) + 1/3 max(input_array),
    # 1/3 min(input_array) + 2/3 max(input_array), max(input_array)
    centroid = np.array([min(input_array), 2/3 * min(input_array) + 1/3 * max(input_array),
    1/3 * min(input_array) + 2/3 * max(input_array), max(input_array)])
    centroid_new = centroid
    criteria = 100
    distance = np.ones([4]) * 0
    class_index = np.ones([input_array.shape[0]]) * 0
    while criteria>0.001:
        # 分配各点
        for i in range(input_array.shape[0]):
            for j in range(4):
                distance[j] = np.sqrt(np.square(input_array[i] - centroid[j]))
                class_index[i] = np.argmin(distance)
        # 求新质心
        for j in range(4):
            centroid_new[j] = np.average(input_array[class_index==j])
        criteria = max(np.abs(centroid - centroid_new))
        centroid = centroid_new
    return (class_index, centroid)

# 聚类操作
class_index_surr_corr = K4_Kmeans(succ_corr_index)[0]
class_index_resp = K4_Kmeans(resp_index)[0]

# 画图检验
# plt.figure(3)
# a = plt.scatter(succ_corr_index[class_index_surr_corr==0], np.zeros(np.sum(class_index_surr_corr==0)), c='white',
#                  s=20, edgecolor='k')
# b = plt.scatter(succ_corr_index[class_index_surr_corr==1], np.zeros(np.sum(class_index_surr_corr==1)), c='green',
#                  s=20, edgecolor='k')
# c = plt.scatter(succ_corr_index[class_index_surr_corr==2], np.zeros(np.sum(class_index_surr_corr==2)), c='red',
#                 s=20, edgecolor='k')
# d = plt.scatter(succ_corr_index[class_index_surr_corr==3], np.zeros(np.sum(class_index_surr_corr==3)), c='pink',
#                 s=30, edgecolor='k')
# plt.show()
# plt.figure(4)
# a1 = plt.scatter(resp_index[class_index_resp==0], np.zeros(np.sum(class_index_resp==0)), c='white',
#                  s=20, edgecolor='k')
# b1 = plt.scatter(resp_index[class_index_resp==1], np.zeros(np.sum(class_index_resp==1)), c='green',
#                  s=20, edgecolor='k')
# c1 = plt.scatter(resp_index[class_index_resp==2], np.zeros(np.sum(class_index_resp==2)), c='red',
#                 s=20, edgecolor='k')
# d1 = plt.scatter(resp_index[class_index_resp==3], np.zeros(np.sum(class_index_resp==3)), c='pink',
#                 s=30, edgecolor='k')
# plt.show()


# 制作标签
time = np.array(range(1440*90)) + 1
# 分别succ_corr与resp提取异常等级为4(编程表达为4-1=3)的time
time_succ_corr = time[class_index_surr_corr==3]
time_resp = time[class_index_resp==3]
# print(time_succ_corr)
# print(time_resp)
# 取交集
time_both = np.array([val for val in time_succ_corr if val in time_resp])
print(time_both.shape)

# 画图检验效果
plt.figure(5)
location = np.vstack((succ_corr_index, resp_index))
plt.scatter(location[0, :], location[1, :], marker='x')
plt.scatter(location[0, time_succ_corr-1], location[1, time_succ_corr-1], c='g')
# plt.scatter(location[0, class_index_surr_corr==3], location[1, class_index_surr_corr==3], c='g')
plt.scatter(location[0, time_resp-1], location[1, time_resp-1], c='b')
# plt.scatter(location[0, class_index_resp==3], location[1, class_index_resp==3], c='b')
plt.scatter(location[0, time_both-1], location[1, time_both-1], c='r', s=50)
plt.show()
'''
2019年2月23日23:24:28
二维数据直接输入孤立森林
计算异常指数

加噪
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import csv

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

# 产生数据集，将succ_corr与resp按每天每分钟排成一行
succ_corr_flatten = succ_corr.flatten('F')
resp_flatten = resp.flatten('F')

# 孤立森林
succ_resp = np.vstack((succ_corr_flatten, resp_flatten))

X_train = succ_resp.transpose()
X_test = X_train
# 开始训练并求检验集异常指数
clf = IsolationForest(n_estimators=100, max_samples=256, contamination=0.001)
clf.fit(X_train)
succ_resp_index = - clf.score_samples(X_test)

label = clf.predict(X_test)

# print(np.sum(label == -1))

# '热力图'看异常指数分布
# plot the line, the samples, and the nearest vectors to the plane
plt.figure(1)
xx, yy = np.meshgrid(np.linspace(np.min(succ_corr_flatten), np.max(succ_corr_flatten), 500),
                     np.linspace(np.min(resp_flatten), np.max(resp_flatten), 500))
Z = clf.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', s=10, c=succ_resp_index)
plt.colorbar()
# plt.scatter(X_test[label==-1, 0], X_test[label==-1, 1], marker='o', s=13, c='r')
plt.xlabel('succ_corr')
plt.ylabel('resp')
plt.show()

# # 检验
# plt.figure(2)
# plt.scatter(succ_resp_index, np.ones(1440*90), c=succ_resp_index)
# plt.scatter(succ_resp_index[label==-1], np.ones(np.sum(label==-1)), marker='o')
# plt.show()

# 加噪
# 误判率，人为设定
fault_rate = 0.1
n = 1000
divide_rate = fault_rate / n
n = n + 1
# 分段加噪点
# 给下段加噪点
below_max = np.max(succ_resp_index[label==1])
below_min = np.min(succ_resp_index[label==1])
below_range = below_max - below_min
above_max = np.max(succ_resp_index[label==-1])
above_min = np.min(succ_resp_index[label==-1])
above_range = above_max - above_min
noise_array = np.array([])
for i in range(n):
    if i==0:
        continue
    below_phase = 0.3 + 0.002 * i
    temp_noise_array = np.where((succ_resp_index<=below_max) & (succ_resp_index>below_max-below_phase * below_range))[0]
    # end = succ_resp_index.shape[0]
    # temp_array = list(range(end))
    # temp_array = np.array(temp_array) + 1
    # temp_array = temp_array.w((succ_resp_index<=below_max) and (succ_resp_index>below_max-below_phase))
    # print(temp_noise_array.shape[0])
    # print(divide_rate*temp_noise_array.shape[0])
    # print(int(divide_rate*temp_noise_array.shape[0]))
    # print(list(temp_noise_array))
    new_noise_array = np.random.choice(list(temp_noise_array), int(divide_rate * succ_resp_index.shape[0]), replace=False)
    noise_array = np.hstack((noise_array, new_noise_array))

noise_array = list(noise_array)
noise_array = list(map(int, noise_array))
label[list(noise_array)] = -1


# 给上段加噪点
noise_array = np.array([])
for i in range(n):
    if i==0:
        continue
    above_phase = 0.3 + 0.002 * i
    temp_noise_array = np.where((succ_resp_index>=above_min) & (succ_resp_index<above_min+above_phase * above_range))[0]
    new_noise_array = np.random.choice(list(temp_noise_array), int(divide_rate * succ_resp_index.shape[0]), replace=False)
    noise_array = np.hstack((noise_array, new_noise_array))

noise_array = list(noise_array)
noise_array = list(map(int, noise_array))
label[list(noise_array)] = 1

# # 检验
# plt.figure(3)
# plt.scatter(succ_resp_index, np.ones(1440*90), c=succ_resp_index)
# # plt.scatter(succ_resp_index[label==-1], np.ones(np.sum(label==-1)), marker='o')
# plt.scatter(succ_resp_index[label==1], np.ones(np.sum(label==1)), marker='*')
# plt.scatter(above_min, 1, s=30)
# plt.show()

# print(label)
# 写入csv文件
csvFile = open("succ_resp_new_index.csv", "w")
writer = csv.writer(csvFile)
time = list(range(1, 1440*90))
writer.writerows([list(succ_resp_index), label, time])
csvFile.close()
'''
2019年2月25日16:57:28
绘制ROC曲线
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

index_label = pd.read_csv('succ_resp_new_index.csv', header=None)
# print(index_label)

index_label = np.array(index_label)
index_label = index_label.transpose()

positive_index_label = np.where(index_label[:, 1] == 1)[0]
positive_index_label = index_label[positive_index_label, 0]
# print(positive_index_label)

negative_index_label = np.where(index_label[:, 1] == -1)[0]
negative_index_label = index_label[negative_index_label, 0]

# 确定起点、终点、步长
begin = np.min(negative_index_label)
end = np.max(negative_index_label)
step = (end - begin) * 0.01

# 开始迭代并记录
False_positive_rate = np.array([])
True_positive_rate = np.array([])
Youden = np.array([])
Prediction = np.array([])
for i in range(100):
    prediction = begin + step * i
    prediction_positive = np.where(index_label[:, 0] < prediction)[0]
    prediction_positive = index_label[prediction_positive, 0]
    prediction_negative = np.where(index_label[:, 0] >= prediction)[0]
    prediction_negative = index_label[prediction_negative, 0]
    # True_positive = [val for val in prediction_positive if val in positive_index_label
    True_positive = list(set(prediction_positive).intersection(set(positive_index_label)))
    # False_positive = [val for val in prediction_positive if val in negative_index_label]
    False_positive = list(set(prediction_positive).intersection(set(negative_index_label)))
    # True_negative = [val for val in prediction_negative if val in positive_index_label]
    True_negative = list(set(prediction_negative).intersection(set(positive_index_label)))
    # False_negative = [val for val in prediction_negative if val in negative_index_label]
    False_negative = list(set(prediction_negative).intersection(set(negative_index_label)))
    Fpr = np.sum(False_positive) / (np.sum(False_positive) + np.sum(True_negative))
    False_positive_rate = np.hstack((False_positive_rate, np.array([Fpr])))
    Tpr = np.sum(True_positive) / (np.sum(True_positive) + np.sum(False_negative))
    True_positive_rate = np.hstack((True_positive_rate, np.array(Tpr)))
    youden = np.sum(True_positive) / (np.sum(True_positive) + np.sum(False_negative)) + np.sum(True_negative) / (np.sum(False_positive) + np.sum(True_negative)) - 1
    Youden = np.hstack((Youden, np.array(youden)))
    Prediction = np.hstack((Prediction, np.array(prediction)))
plt.plot(False_positive_rate, True_positive_rate)
plt.axis([0, 1, 0, 1])
plt.show()

# # 计算AUC
# y_scores = index_label[:, 0]
# y_scores = list(y_scores.transpose())
# negative_index_label = np.where(index_label[:, 1] == -1)[0]
# index_label[negative_index_label, 0] = 0
# y_true = index_label[:, 1]
# y_true = list(y_true.transpose())
# print(len(y_true))
# # print(y_scores)
# auc = roc_auc_score(y_true, y_scores)
# print(auc)

# 通过ROC面积计算auc
auc = 0.
for y in True_positive_rate:
    auc += 1 / 100 * y
print(auc)

# 通过约登指数寻找最优阈值
best_youden = np.argmax(Youden)
best_prediction = Prediction[best_youden]
print(best_prediction)
plt.plot(False_positive_rate, True_positive_rate)
plt.plot(False_positive_rate, Youden)
plt.axis([0, 1, 0, 1])
plt.show()
'''
2019-3-17 12:45:10
By Piper Liu

Based on the idea of semi-supervised learning,
the ROC curve is used to find the optimal threshold.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

train_day = 5

trade_raw = pd.read_csv('trade.csv')
succ_raw = pd.read_csv('succ.csv')
resp_raw = pd.read_csv('resp.csv')

trade = np.array(trade_raw)
trade = np.delete(trade, 0, axis=1)

succ = np.array(succ_raw)
succ = np.delete(succ, 0, axis=1)

resp = np.array(resp_raw)
resp = np.delete(resp, 0, axis=1)

# trade
trade = trade.flatten('F')
# succ
succ = succ.flatten('F')
EX = np.mean(succ)
succ_corr = (succ - EX) * np.sqrt(trade)
# resp
resp = resp.flatten('F')

# csv
iListII = pd.read_csv('makeLabel_iListII.csv', header=None)
iListIII = pd.read_csv('makeLabel_iListIII.csv', header=None)
iListIV = pd.read_csv('makeLabel_iListIV.csv', header=None)
iListII = np.array(iListII)
iListIII = np.array(iListIII)
iListIV = np.array(iListIV)
iListII, iListIII, iListIV = iListII[0], iListIII[0], iListIV[0]

# outlierIV
# pretreatment
succ_corr_normal = succ_corr
succ_corr_normal[iListII] = 0
resp_normal = resp
resp_normal[iListIII] = 0
# isolationforest
succ_resp = np.vstack((succ_corr_normal, resp_normal))
X_train = succ_resp[:, :1440*train_day]
X_test = succ_resp[:, train_day*1440:]
X_train = X_train.transpose()
X_test = X_test.transpose()
clf = IsolationForest(n_estimators=100, max_samples=256, contamination=0.001)
clf.fit(X_train)
# plot the train set
succ_resp_index = - clf.score_samples(X_train)
xx, yy = np.meshgrid(np.linspace(np.min(succ_corr_normal)*1.1, np.max(succ_corr_normal)*1.1, 500),
                     np.linspace(np.min(resp_normal)-100, np.max(resp_normal)*1.1, 500))
Z = clf.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', s=10, c=succ_resp_index)
# plt.colorbar()
plt.xlabel('succ_corr')
plt.ylabel('resp')
plt.show()
# plot the test set
succ_resp_index = - clf.score_samples(X_test)
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', s=10, c=succ_resp_index)
# plt.colorbar()
plt.xlabel('succ_corr')
plt.ylabel('resp')
plt.show()

# ROC
index = succ_resp_index
iListIV_index = np.where(iListIV >= 1440*(90-train_day))[0]
iListIV = iListIV[iListIV_index] - 1440*(90-train_day)
label = np.ones(1440*(90-train_day))
label[iListIV] = -1

positive_index_label = np.where(label == 1)[0]
positive_index_label = index[positive_index_label]

negative_index_label = np.where(label == -1)[0]
negative_index_label = index[negative_index_label]

# paraments
begin = np.min(negative_index_label) * 0.99
end = np.max(negative_index_label) * 1.01
step = (end - begin) * 0.01

# iteration
False_positive_rate = np.array([])
True_positive_rate = np.array([])
Youden = np.array([])
Prediction = np.array([])
for i in range(100):
    prediction = begin + step * i
    prediction_positive = np.where(index < prediction)[0]
    prediction_positive = index[prediction_positive]
    prediction_negative = np.where(index >= prediction)[0]
    prediction_negative = index[prediction_negative]
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

# auc
auc = 0.
for y in True_positive_rate:
    auc += 1 / 100 * y
print(auc)

# youden
best_youden = np.argmax(Youden)
best_prediction = Prediction[best_youden]
print(best_prediction)
plt.plot(False_positive_rate, True_positive_rate)
plt.plot(False_positive_rate, Youden)
plt.axis([0, 1, 0, 1])
plt.show()
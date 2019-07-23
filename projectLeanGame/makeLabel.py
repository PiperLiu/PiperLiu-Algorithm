'''
2019-3-14 08:31:54
By Piper Liu

All data are processed to prepare for
subsequent verification and semi-supervised learning.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import csv

trade_raw = pd.read_csv('trade.csv')
succ_raw = pd.read_csv('succ.csv')
resp_raw = pd.read_csv('resp.csv')

trade = np.array(trade_raw)
trade = np.delete(trade, 0, axis=1)

succ = np.array(succ_raw)
succ = np.delete(succ, 0, axis=1)

resp = np.array(resp_raw)
resp = np.delete(resp, 0, axis=1)

timeList = trade_raw['Date']
timeList = np.array(timeList)
dateList = trade_raw.columns
dateList = dateList[1:]
dateList = np.array(dateList)   # type(dateList) == 'str'

# trade of a certain day
dayIndex = 0
plt.plot(trade[:, dayIndex])
plt.xlabel('time/minute')
plt.ylabel('trade')
plt.show()

# trade
trade = trade.flatten('F')
trade = np.append(trade, np.ones(5)*120)    # float64

# 90 days
plt.plot(trade)
plt.xlabel('time/minute')
plt.ylabel('trade')
plt.show()

# Moving Average
trade_trend = np.ones(trade.shape[0])
denominator = 11
for i in range(0+5, trade.shape[0]-5):
    numerator = np.sum(trade[i-5:i+6])
    trade_trend[i] = numerator / denominator
trade_trend[0:5] = np.mean(trade[0:11])
trade_random = trade - trade_trend
trade_random = trade_random[0:-5]

# t t r
plt.plot(trade[1440:1440*2], label='trade')
plt.plot(trade_trend[1440:1440*2], label='trade_trend')
plt.scatter(range(1440), trade_random[1440:1440*2], c='g', s=5, label='trade_random')
plt.xlabel('time/minute')
plt.ylabel('trade')
plt.legend(prop={'size':15})
plt.show()

# t t r
plt.plot(trade[1440:1440+780+5])
plt.plot(trade_trend[1440:1440+780])
plt.xlabel('time/minute')
plt.ylabel('trade')
plt.show()

# sigma of every minute
sigma = np.zeros(1440)
for i in range(1440):
    sigma[i] = np.std(trade_random[i:90*1440:1440])

# search for outliersI
outlierI = np.array([0, 0])
iListI = []
for i in range(trade_random.shape[0]):
    day = int(np.ceil(i / 1440)) - 1
    time = int(np.mod(i, 1440)) - 1
    if trade_random[i] < -3 * sigma[time]:
        new_outlierI = np.array([dateList[day], timeList[time]])
        outlierI = np.vstack((outlierI, new_outlierI))
        iListI = iListI + [i]
outlierI = outlierI[1:, :]

# succ
succ = succ.flatten('F')
trade = trade[0:-5]
EX = np.mean(succ)
succ_corr = (succ - EX) * np.sqrt(trade)

# succ plot normal
plt.plot(trade[1440*10:1440*11] / 20, label='trade / * 20')
plt.plot(succ[1440*10:1440*11], label='succ')
plt.xlabel('time/minute')
plt.ylabel('succ or trade')
plt.legend(prop={'size':20})
plt.show()

# - 3 sigma
sigma_succ = np.std(succ_corr)

# succ contrast
plt.plot(succ[1440*30:1440*31], label='succ')
plt.plot(succ_corr[1440*30:1440*31], label='succ_corr')
plt.plot(range(1440), -3 * np.ones(1440) * sigma_succ)
plt.xlabel('time/minute')
plt.ylabel('succ')
plt.legend(prop={'size':15})
plt.show()

# search for outliersII
outlierII = np.array([0, 0])
iListII = []
for i in range(succ_corr.shape[0]):
    day = int(np.ceil(i / 1440)) - 1
    time = int(np.mod(i, 1440)) - 1
    if succ_corr[i] < - 3 * sigma_succ:
        new_outlierII = np.array([dateList[day], timeList[time]])
        outlierII = np.vstack((outlierII, new_outlierII))
        iListII = iListII + [i]
outlierII = outlierII[1:, :]

# resp
resp = resp.flatten('F')
EX_resp = np.mean(resp)
sigma_resp = np.std(resp)

# wired outliersIII
outlierIII = np.array([0, 0])
iListIII = []
for i in range(resp.shape[0]):
    day = int(np.ceil(i / 1440)) - 1
    time = int(np.mod(i, 1440)) - 1
    if resp[i] > 3 * sigma_resp + EX_resp:
        new_outlierIII = np.array([dateList[day], timeList[time]])
        outlierIII = np.vstack((outlierIII, new_outlierIII))
        iListIII = iListIII + [i]
outlierIII = outlierIII[1:, :]

plt.scatter(range(1440*90), resp, s=7)
plt.scatter(iListIII, resp[iListIII], s=7)
plt.plot(range(1440*90), 3 * np.ones(1440*90) * sigma_resp + EX_resp, c='g')
plt.ylim([0, 400000])
plt.show()

# second time for 3 sigma
resp_temp = list(set(resp).difference(set(resp[iListIII])))
resp_temp = np.array(resp_temp)
EX_resp_corr = np.mean(resp_temp)
sigma_resp_corr = np.std(resp_temp)

resp_new = np.copy(resp)
resp_new[iListIII] = 0

# outliersIII
for i in range(resp_new.shape[0]):
    day = int(np.ceil(i / 1440)) - 1
    time = int(np.mod(i, 1440)) - 1
    if resp_new[i] > 3 * sigma_resp_corr + EX_resp_corr:
        new_outlierIII = np.array([dateList[day], timeList[time]])
        outlierIII = np.vstack((outlierIII, new_outlierIII))
        iListIII = iListIII + [i]

# resp contrast
plt.scatter(range(1440*90), resp, s=7)
plt.scatter(iListIII, resp[iListIII], s=7)
plt.plot(range(1440*90), 3 * np.ones(1440*90) * sigma_resp_corr + EX_resp_corr, c='g')
plt.ylim([0, 400000])
plt.show()

# outlierIV
# pretreatment
succ_corr_normal = succ_corr
succ_corr_normal[iListII] = 0
resp_normal = resp
resp_normal[iListIII] = 0
# isolationforest
succ_resp = np.vstack((succ_corr_normal, resp_normal))
X_train = succ_resp.transpose()
X_test = X_train
clf = IsolationForest(n_estimators=100, max_samples=256, contamination=0.001)
clf.fit(X_train)
succ_resp_index = - clf.score_samples(X_test)
# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(np.min(succ_corr_normal)*1.1, np.max(succ_corr_normal)*1.1, 500),
                     np.linspace(np.min(resp_normal)-100, np.max(resp_normal)*1.1, 500))
Z = clf.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', s=10, c=succ_resp_index)
# plt.colorbar()
plt.xlabel('succ_corr')
plt.ylabel('resp')
plt.show()
# outlierIV
label = clf.predict(X_test)
outlierIV = np.array([0, 0])
iListIV = []
for i in range(succ_resp_index.shape[0]):
    day = int(np.ceil(i / 1440)) - 1
    time = int(np.mod(i, 1440)) - 1
    if label[i] < 0:
        new_outlierIV = np.array([dateList[day], timeList[time]])
        outlierIV = np.vstack((outlierIV, new_outlierIV))
        iListIV = iListIV + [i]

# csv for recording
csvFile = open("makeLabel_outlier.csv", "w")
writer = csv.writer(csvFile)
writer.writerows([outlierI, outlierII, outlierIII, outlierIV])
csvFile.close()
# for iList
csvFile = open("makeLabel_iListI.csv", "w")
writer = csv.writer(csvFile)
writer.writerows([iListI])
csvFile.close()
csvFile = open("makeLabel_iListII.csv", "w")
writer = csv.writer(csvFile)
writer.writerows([iListII])
csvFile.close()
csvFile = open("makeLabel_iListIII.csv", "w")
writer = csv.writer(csvFile)
writer.writerows([iListIII])
csvFile.close()
csvFile = open("makeLabel_iListIV.csv", "w")
writer = csv.writer(csvFile)
writer.writerows([iListIV])
csvFile.close()

# plot trade_trend
dayIndex = 0
plt.plot(trade[dayIndex*1440:(dayIndex+1)*1440])
plt.plot(trade_trend[dayIndex*1440:(dayIndex+1)*1440])
plt.show()

# trade_trend
trade_trend = trade_trend[:-5]
trade_trend = trade_trend.reshape(90, 1440)
csvFile = open("makeLabel_tradetrend.csv", "w")
writer = csv.writer(csvFile)
writer.writerows(trade_trend)
csvFile.close()
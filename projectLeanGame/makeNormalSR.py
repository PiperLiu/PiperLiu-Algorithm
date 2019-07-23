'''
2019-3-17 12:09:45
By Piper Liu

The success rate is not preprocessed
and the abnormal points of response time are not removed.
Used to highlight our proposed approach.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trade_raw = pd.read_csv('trade.csv')
succ_raw = pd.read_csv('succ.csv')
resp_raw = pd.read_csv('resp.csv')

trade = np.array(trade_raw)
trade = np.delete(trade, 0, axis=1)

succ = np.array(succ_raw)
succ = np.delete(succ, 0, axis=1)

resp = np.array(resp_raw)
resp = np.delete(resp, 0, axis=1)

# succ_corr
trade = trade.flatten('F')
succ = succ.flatten('F')
EX = np.mean(succ)
succ_corr = (succ - EX) * np.sqrt(trade)

# sigma
sigma_succ = np.std(succ)
sigma_succ_corr = np.std(succ_corr)

# plot for Contrast
dayIndex = 0
sigma_succ_flatten = np.ones(1440) * (EX - 3 * sigma_succ)
plt.plot(succ[dayIndex * 1440:(dayIndex + 1) * 1440])
plt.plot(sigma_succ_flatten)
plt.show()
sigma_succ_corr_flatten = np.ones(1440) * (- 3 * sigma_succ_corr)
plt.plot(succ_corr[dayIndex * 1440:(dayIndex + 1) * 1440])
plt.plot(sigma_succ_corr_flatten)
plt.show()

# correct of resp
resp = resp.flatten('F')
EX_resp = np.mean(resp)
sigma_resp = np.std(resp)

# wired outliersIII
iListIII = []
for i in range(resp.shape[0]):
    if resp[i] > 3 * sigma_resp + EX_resp:
        iListIII = iListIII + [i]

plt.scatter(range(1440*90), resp)
plt.scatter(iListIII, resp[iListIII])
resp_flatten = np.ones(1440*90) * 3 * sigma_resp
plt.plot(resp_flatten)
plt.show()

print(iListIII.__len__() / 143)
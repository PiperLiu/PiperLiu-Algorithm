import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

def get_arima(data):
    data['diff_1'] = data['Appliances'].diff(1)
    data['diff_2'] = data['diff_1'].diff(2)
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(131)
    ax1.plot(data['Appliances'])
    ax2 = fig.add_subplot(132)
    ax2.plot(data['diff_1'])
    ax3 = fig.add_subplot(133)
    ax3.plot(data['diff_2'])
    plt.show()

    train = data['Appliances']

    # # 输出结果图示
    # p_min = 0
    # d_min = 0
    # q_min = 0
    # p_max = 8
    # d_max = 0
    # q_max = 8
    # # Initialize a DataFrame to store the results, 以BIC准则
    # results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
    #                            columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
    # for p, d, q in itertools.product(range(p_min, p_max + 1),
    #                                  range(d_min, d_max + 1),
    #                                  range(q_min, q_max + 1)):
    #     if p == 0 and d == 0 and q == 0:
    #         results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
    #         continue
    #     try:
    #         model = sm.tsa.ARIMA(train, order=(p, d, q),
    #                              # enforce_stationarity=False,
    #                              # enforce_invertibility=False,
    #                              )
    #         results = model.fit()
    #         results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    #     except:
    #         continue
    # results_bic = results_bic[results_bic.columns].astype(float)
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax = sns.heatmap(results_bic,
    #                  mask=results_bic.isnull(),
    #                  ax=ax,
    #                  annot=True,
    #                  fmt='.2f',
    #                  )
    # ax.set_title('BIC')
    # plt.show()
    # # BIC(7,6)

    # train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)
    # print('AIC', train_results.aic_min_order)
    # print('BIC', train_results.bic_min_order)
    # AIC(7，8)
    # BIC(7, 8)

    model = sm.tsa.ARIMA(train, order=(7, 0, 6))
    results = model.fit()
    resid = results.resid  # 赋值
    plt.figure(figsize=(12, 8))
    sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40)
    plt.show()

def use_arima(data):
    train = data['Appliances']
    model = sm.tsa.ARIMA(train, order=(7, 0, 6))
    results = model.fit()
    # print(results.params)
    predict_sunspots = results.predict(start=0, end=19734, dynamic=False)
    # plt.plot(train)
    # plt.plot(predict_sunspots)
    # plt.show()
    return predict_sunspots
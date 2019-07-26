import pandas as pd
import numpy as np
from sklearn.cluster import k_means
from sklearn import metrics

def read(path):
    df = pd.read_hdf(path, '1.0')
    return df

def baselinePromote(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    train_test = pd.concat([train, test], ignore_index=True)
    temp = train_test.groupby(['buyer_admin_id', 'item_id']).size().reset_index()
    temp.columns = ['buyer_admin_id', 'item_id', 'item_cnts']
    train_test = pd.merge(train_test, temp)
    train_test = train_test.sort_values(['buyer_admin_id', 'item_cnts', 'irank'])

    train_test_t = train_test.loc[train_test.buyer_country_id == 'yy']
    train_test_t = train_test_t.drop_duplicates(subset=['buyer_admin_id', 'item_id'], keep='first')

    item_hot = train_test_t.groupby(['item_id']).size().reset_index()
    item_hot.columns = ['item_id', 'cnts']
    # 按照cnts降序排序
    item_hot = item_hot.sort_values('cnts', ascending=False)
    items = item_hot['item_id'].values.tolist()

    # 很多admin的历史行为不够30个item，所以就需要填充够30个
    # 这里使用train下yy的数据构造item_id频次排序，然后依次填充
    def item_fillna(tmp_):
        tmp = tmp_.copy()
        l = len(tmp)
        if l == 30:
            tmp = tmp
        elif l < 30:
            m = 30 - l
            items_t = items.copy()
            for i in range(m):
                for j in range(50):
                    # 从items_t中取出第一个值（最热门的），并删去
                    it = items_t.pop(0)
                    if it not in tmp:
                        tmp.append(it)
                        break
        elif l > 30:
            tmp = tmp[:30]

        return tmp

    # 获取top30的item
    def get_item_list(df_):
        df = df_.copy()
        dic = {}
        flag = 0
        for item in df[['buyer_admin_id', 'item_id']].values:
            # 此时item: ['buyer_admin_id', 'item_id'] dtype=int
            try:
                dic[item[0]].append(item[1])
                # 对于第n个用户，如果是第1次进行append，则报错，执行except语句块
                # except中，对第n-1个用户进行后续进行操作，即，执行item_fillna()
            except:
                if flag != 0:
                    # 去重
                    tmp = []
                    for i in dic[flag]:
                        if i not in tmp:
                            tmp.append(i)
                    # 填充
                    tmp = item_fillna(tmp)
                    dic[flag] = tmp

                    # 以下三行是保证迭代中flag的值总为用户id
                    flag = item[0]
                else:
                    # 用于第一次确认flag的值
                    flag = item[0]

                dic[item[0]] = [item[1]]

        return dic

    # 按照'b_a_i', 'item_cnts', 'irank'排序，制作标准test
    test = train_test.loc[train_test.is_train == 0]
    test = test.sort_values(['buyer_admin_id', 'item_cnts', 'irank'])
    # 结果
    dic = get_item_list(test)

    # 最终提交
    temp = pd.DataFrame({'lst': dic}).reset_index()
    for i in range(30):
        temp[i] = temp['lst'].apply(lambda x: x[i])
    del temp['lst']

    return temp


class K_means(object):
    def __init__(self, n_clusters, train_df):
        self.n_clusters = n_clusters.copy()
        self.train = train_df.copy()


if __name__ == '__main__':
    data = read('train_test.h5')
    train = data.loc[data.is_train == 1]
    test = data.loc[data.is_train == 0]
    temp = baselinePromote(train, test)
    temp.to_csv('submission_blPro.csv', index=False, header=None)

import numpy as np
import pandas as pd
# import os
# from tqdm import tqdm_notebook
# import lightgbm as lgb
# import xgboost as xgb
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

path = ''
item = pd.read_csv(path + 'Antai_AE_round1_item_attr_20190626.csv')
# submit = pd.read_csv(path + 'Antai_AE_round1_submit_20190715.csv', header=None)
test = pd.read_csv(path + 'Antai_AE_round1_test_20190626.csv')
train = pd.read_csv(path + 'Antai_AE_round1_train_20190626.csv')

def get_preprocessing(df_):
    df = df_.copy()
    # df['hour'] = df['create_order_time'].apply(lambda x: int(x[11:13]))
    # df['day'] = df['create_order_time'].apply(lambda x: int(x[8:10]))
    # df['month'] = df['create_order_time'].apply(lambda x: int(x[5:7]))
    # df['year'] = df['create_order_time'].apply(lambda x: int(x[0:4]))
    # df['date'] = (df['month'].values - 7) * 31 + df['day']
    # del df['create_order_time']
    return df


train = get_preprocessing(train)
test = get_preprocessing(test)

# 高频item_id
# 针对yy国，提取高频数据
temp = train.loc[train.buyer_country_id == 'yy']
# 去重（A用户多次购买a商品）
temp = temp.drop_duplicates(subset=['buyer_admin_id', 'item_id'], keep='first')
# item_cnts中存储item_id出现次数（购买人数）
# groupby()将df中的数据按照键‘item_id’进行分组
# size()返回每组中元素的个数，reset_index()为df添加索引列
item_cnts = temp.groupby(['item_id']).size().reset_index()
item_cnts.columns = ['item_id', 'cnts']
# 按照cnts降序排序
item_cnts = item_cnts.sort_values('cnts', ascending=False)
items = item_cnts['item_id'].values.tolist()

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

# 先按照'b_a_i'排序，再按'irank'排序，制作标准test
test = test.sort_values(['buyer_admin_id', 'irank'])
# 结果
dic = get_item_list(test)

# 最终提交
temp = pd.DataFrame({'lst': dic}).reset_index()
for i in range(30):
    temp[i] = temp['lst'].apply(lambda x: x[i])
del temp['lst']
temp.to_csv('submission.csv', index=False, header=None)

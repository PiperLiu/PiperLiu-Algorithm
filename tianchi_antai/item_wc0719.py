import pandas as pd
# from numba import jit
import math
# import json
from operator import itemgetter
dataset = {}
sim_num = 40
rec_num = 30
item_popular = {}
item_count = 0

sim_matrix = {}

def data_processing():
    record = pd.read_csv('Antai_AE_round1_train_20190626.csv')
    attr = pd.read_csv('Antai_AE_round1_item_attr_20190626.csv')
    attr = attr[['item_id', 'cate_id']]
    record = record[['buyer_admin_id', 'item_id', 'irank']]
    df = pd.merge(record, attr).sort_values(by=['buyer_admin_id', 'irank'], ascending=(True, True)).reset_index(drop=True)
    # df['hour'] = df['create_order_time'].apply(lambda x: int(x[11:13]))
    # df['day'] = df['create_order_time'].apply(lambda x: int(x[8:10]))
    # df['month'] = df['create_order_time'].apply(lambda x: int(x[5:7]))
    # df['year'] = df['create_order_time'].apply(lambda x: int(x[0:4]))
    # df['date'] = (df['month'].values - 7) * 31 + df['day']
    # del df['create_order_time']
    return df


def get_dataset(df_):
    df = df_.copy()
    for row in df.itertuples(index=True, name='Pandas'):
        admin = getattr(row, "buyer_admin_id")
        dataset.setdefault(admin, {})
        dataset[admin][getattr(row, "item_id")] = 1


def process():
    for user, items in dataset.items():
        for item in items:
            if item not in item_popular:
                item_popular[item] = 0
            item_popular[item] += 1
    print("用户-物品矩阵已完毕")  # 物品-次数矩阵
    for user, items in dataset.items():
        for i1 in items:
            for i2 in items:
                if i1 == i2:
                    continue
                sim_matrix.setdefault(i1, {})
                sim_matrix[i1].setdefault(i2, 0)
                sim_matrix[i1][i2] += 1
    print("物品物品矩阵已完毕")  # 物品-物品矩阵（两者被一个人同时购买次数）
    for m1, related_movies in sim_matrix.items():
        for m2, count in related_movies.items():
            # 注意0向量的处理，即某电影的用户数为0
            if item_popular[m1] == 0 or item_popular[m2] == 0:
                sim_matrix[m1][m2] = 0
            else:
                sim_matrix[m1][m2] = count / math.sqrt(item_popular[m1] * item_popular[m2])
    print("物品相似度矩阵已完毕")
    # jsObj = json.dumps(sim_matrix)
    # fileObject = open(r'F:\工程\untitled\1.json', 'w')
    # fileObject.write(jsObj)
    # fileObject.close()



def recommend(user):
    K = 50
    N = 50
    rank = {}
    items = dataset[user]
    for item, rating in items.items():
        for related_movie, w in sorted(sim_matrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
            if related_movie in items:
                continue
            rank.setdefault(related_movie, 0)
            rank[related_movie] += w * float(rating)
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]


if __name__ == '__main__':
    get_dataset(data_processing())
    process()
    print(recommend(1))
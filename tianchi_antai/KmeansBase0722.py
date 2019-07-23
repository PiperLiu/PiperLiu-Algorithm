import pandas as pd
import math
import random

def read(path):
    df = pd.read_hdf(path, '1.0')
    return df

# 聚类距离
def distance(list_1_, list_2_, cate_cnts):
    list_1 = list_1_.copy()
    list_2 = list_2_.copy()
    distance = 0
    for i in list_1:
        if i in list_2:
            distance += math.sqrt(math.log(cate_cnts.loc[cate_cnts['cate_id'] == i].cnts.values))
            list_2.remove(i)
    return distance

# 将0-1质心转为cate列表形式
def cenToList(centroid, len_cate):
    centroid_new = []
    for centroid_k in centroid:
        centroid_new.append([])
        for cate in range(len_cate):
            if centroid_k[cate] >= 0.5:
                centroid_new[-1].append(cate + 1)
    return centroid_new

# 聚类停止判据 ================》》》》明早改
def flag(init_centroid, centroid):
    flag = True
    list_1 = init_centroid
    list_2 = centroid
    for i in list_1:
        if i in list_2:
            list_1.remove(i)
            list_2.remove(i)
            continue
    if list_1 == [] and list_2 ==[]:
        flag = False
    return flag

# 求新质心
def averageCent(dict, block_index, K, len_cate):
    centroid = list(range(K))

    for k in range(K):
        dict_k = {}
        for user in block_index:
            if block_index[user] == k:
                dict_k.update(dict[user])
        len_user_k = len(dict_k)

        cate_cnts_k = list(range(len_cate)) * 0
        for user_k in dict_k:
            for cate in dict_k[user_k]:
                cate_cnts_k[cate] += 1

        for cate in range(len_cate):
            centroid[k][cate] = cate_cnts_k[cate] / len_user_k

        centroid = cenToList(centroid, len_cate)

    return centroid

def Kmeans(dict_, K, cate_cnts):
    dict = dict_.copy()
    len_cate = cate_cnts.shape[0]
    p = 50 / len_cate
    init_centroid = []
    centroid = []
    for k in range(K):
        init_centroid.append([])
        for j in range(len_cate):
            if random.random() < p:
                init_centroid[k].append(1)
            else:
                init_centroid[k].append(0)
    init_centroid = cenToList(init_centroid, len_cate)
    for k in range(K):
        centroid.append([])
        for j in range(len_cate):
            if random.random() < p:
                centroid[k].append(1)
            else:
                centroid[k].append(0)
    centroid = cenToList(centroid, len_cate)
    block_index = []

    while flag(init_centroid, centroid):
        # 分配各点
        for user in dict:
            dist = []
            for k in range(K):
                dist.append(distance(dict[user], centroid[k], cate_cnts))
            block_index.append(dist.index(min(dist)))
        # 求新质心
        centroid = averageCent(dict, block_index, K, len_cate)

    return block_index, centroid

def classification(df_):
    df = df_.copy()

    # block = {uesr_1:[]. user_2:[]}
    block = {}
    for row in df.itertuples(index=True, name='Pandas'):
        admin = getattr(row, "buyer_admin_id")
        block.setdefault(admin, [])
        block[admin].append(getattr(row, "cate_id"))

    cate_cnts = df.groupby(['cate_id']).size().reset_index()
    cate_cnts.columns = ['cate_id', 'cnts']
    block_index, centroid = Kmeans(block, 5, cate_cnts)

    # blocks = {{df's sebset_1}, {df's sebset_2}, {df's sebset_3}}
    blocks = {}
    for user in block_index:
        blocks.setdefault(user, {})
        blocks[user].update(block[user])

    return blocks, centroid

# ================================= 下面是基于baseline的推荐 =================================
# ================================= 原思路 =================================
# 求yy国热度商品
def hot_yy(df_):
    df = df_.copy()
    # 针对yy国，提取高频数据
    temp = df.loc[df.buyer_country_id == 'yy']
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
    return items

# 很多admin的历史行为不够30个item，所以就需要填充够30个
# 这里使用train下yy的数据构造item_id频次排序，然后依次填充
def item_fillna(tmp_, items):
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

# 填充至30个商品
def get_item_list(df_, df_test):
    items = hot_yy(df_)
    df = df_test.copy()
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
                tmp = item_fillna(tmp, items)
                dic[flag] = tmp

            # 以下三行是保证迭代中flag的值总为用户id
                flag = item[0]
            else:
                # 用于第一次确认flag的值
                flag = item[0]

            dic[item[0]] = [item[1]]

    return dic

if __name__ == '__main__':
    data = read('train_test.h5')
    blocks, centroid = classification(data)
    print(blocks)
    get_item_list(data, data.loc[data.is_train == '0'])


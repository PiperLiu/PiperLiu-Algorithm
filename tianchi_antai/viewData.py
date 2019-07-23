# from https://nbviewer.jupyter.org/github/RainFung/awesome-visualization/blob/master/Electronic-Commerce

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gc
    # 禁用科学计数法
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    item = pd.read_csv('Antai_AE_round1_item_attr_20190626.csv')
    train = pd.read_csv('Antai_AE_round1_train_20190626.csv')
    test = pd.read_csv('Antai_AE_round1_test_20190626.csv')
    submit = pd.read_csv('Antai_AE_round1_submit_20190715.csv')

    # 数据预处理
    # 合并train和test文件
    # 提取日期年月日等信息
    # 关联商品价格、品类、店铺
    # 转化每列数据类型为可存储的最小值，减少内存消耗
    # 保存为hdf5格式文件，加速读取

    df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)])

    df['create_order_time'] = pd.to_datetime(df['create_order_time'])
    df['date'] = df['create_order_time'].dt.date
    df['day'] = df['create_order_time'].dt.day
    df['hour'] = df['create_order_time'].dt.hour

    df = pd.merge(df, item, how='left', on='item_id')

    memory = df.memory_usage().sum() / 1024 ** 2
    print('Before memory usage of properties dataframe is :', memory, " MB")

    dtype_dict = {'buyer_admin_id': 'int32',
                  'item_id': 'int32',
                  'store_id': pd.Int32Dtype(),
                  'irank': 'int16',
                  'item_price': pd.Int16Dtype(),
                  'cate_id': pd.Int16Dtype(),
                  'is_train': 'int8',
                  'day': 'int8',
                  'hour': 'int8',
                  }

    df = df.astype(dtype_dict)
    memory = df.memory_usage().sum() / 1024 ** 2
    print('After memory usage of properties dataframe is :', memory, " MB")
    del train, test
    gc.collect()

    # Before memory usage of properties dataframe is : 1292.8728713989258  MB
    # After memory usage of properties dataframe is : 696.1623153686523  MB

    for col in ['store_id', 'item_price', 'cate_id']:
        df[col] = df[col].fillna(0).astype(np.int32).replace(0, np.nan)
    df.to_hdf('train_test.h5', '1.0')

    df = pd.read_hdf('train_test.h5', '1.0')

    # 文件内存占用从1200M减少至600M
    # 采用hdf5格式存储，读取时间减少2/3

"""
    20220-05-04
    依赖于SKU-Data数据, 预测SKU Embedding(title + point)属于哪个品牌？
    brand_label = index  如果不存在则为-1
    color_label = index  如果不存在则为-1

    预训练模型区分大小写的， 这里品牌、型号没有区分大小写了
"""
import numpy as np
from collections import OrderedDict
from scipy.special import factorial
import pandas as pd
import string
from random import randrange, uniform
import datetime
import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    os.environ['PYTHONHASHSEED'] = '12345'
    os.execv(sys.executable, [sys.executable] + sys.argv)


############################ 1.创建brand、color 词表数据
"""
    删除特殊字符、转化为小写
"""
productid_file = "/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
df = pd.read_csv(productid_file, na_values="", keep_default_na=True)
# print(df[df['product_locale'] == 'us'][df['product_id'] == 'B003O0MNGC'])

brand_list = df['product_brand'].fillna('').to_list()
color_list = df['product_color_name'].fillna('').to_list()
brand_list = [str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip() for item in brand_list if item and not item.isspace() ]
color_list = [str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip() for item in color_list if item and not item.isspace() ]

brand_dict = dict()
color_dict = dict()
for item in brand_list:
    if item and not item.isspace():
        if item in brand_dict:
            brand_dict[item] += 1
        else:
            brand_dict[item] = 1

for item in color_list:
    if item and not item.isspace():
        if item in color_dict:
            color_dict[item] += 1
        else:
            color_dict[item] = 1
brand_list = [item for item,value in brand_dict.items() if value > 1 ]
color_list = [item for item,value in color_dict.items() if value > 1 ]        # 颜色字段不要求大于1次。 本来覆盖度就只有70%
brand_set = set(brand_list)
color_set = set(color_list)

print(len(brand_list), len(color_list), len(brand_set), len(color_set))   # 1672228 1123930 292608 234762

count = 0
with open("./extra_vocab/brand.txt", mode='w') as fout:
    fout.write('[UNK]' + '\n')
    fout.write('[UNK]' + '\n')
    fout.write('[UNK]' + '\n')
    for item in list(brand_set):
        if item and not item.isspace():
            fout.write(item + '\n')
            count += 1
        else:
            print('brand_nan_item=', item)
print('brand_vocab_len=', count + 3)

count = 0
with open("./extra_vocab/color.txt", mode='w') as fout:
    fout.write('[UNK]' + '\n')
    fout.write('[UNK]' + '\n')
    fout.write('[UNK]' + '\n')
    for item in list(color_set):
        if item and not item.isspace():
            fout.write(item + '\n')
            count += 1
        else:
            print('color_nan_item=', item)
print('color_vocab_len=', count + 3)

############################################ 2.对于商品集合添加brand_label, color_label
brand_name_set = set()
brand_name_idx = dict()
color_name_idx = dict()
idx = 0
with open('./extra_vocab/brand.txt', mode='r') as fin:
    for line in fin:
        item = line.strip()
        if item != '[UNK]':
            brand_name_set.add(item)
            brand_name_idx[item] = idx
        idx += 1

idx = 0
with open('./extra_vocab/color.txt', mode='r') as fin:
    for line in fin:
        item = line.strip()
        if item != '[UNK]':
            color_name_idx[item] = idx
        idx += 1
print(len(brand_name_idx.keys()), len(color_name_idx.keys()), len(brand_name_set))
print('idx=', brand_name_idx['mungoo mach mal anders'])

productid_file = "/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
product_df = pd.read_csv(productid_file, na_values="", keep_default_na=True)

count_brand = 0
def func_brand2idx(brand_name):
    brand_name_lower = str(brand_name).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
    if brand_name_lower and not brand_name_lower.isspace():
        if brand_name_lower in brand_name_idx:
            idx =  brand_name_idx[brand_name_lower]
        else: idx = -2
    else:
        idx =  -1
    return int(idx)

count_color = 0
def func_color2idx(color_name):
    color_name_lower = str(color_name).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
    if color_name_lower and not color_name_lower.isspace():
        if color_name_lower in color_name_idx:
            idx =  color_name_idx[color_name_lower]
        else: idx = -2
    else:
        idx =  -1
    return int(idx)

product_df['brand_vocab_idx'] = product_df['product_brand'].fillna('').apply(func_brand2idx)
product_df['color_vocab_idx'] = product_df['product_color_name'].fillna('').apply(func_color2idx)
print(product_df['product_brand'].iloc[10], product_df['brand_vocab_idx'].iloc[10], product_df['product_color_name'].iloc[10], product_df['color_vocab_idx'].iloc[100])
print(product_df['product_brand'].iloc[-1], product_df['brand_vocab_idx'].iloc[-1], product_df['product_color_name'].iloc[-1], product_df['color_vocab_idx'].iloc[-1])
print(product_df['product_brand'].iloc[0], product_df['brand_vocab_idx'].iloc[0], product_df['product_color_name'].iloc[0], product_df['color_vocab_idx'].iloc[0])


# -1 代表是空值,  -2代表其字段信息仅出现1次
print(product_df[(product_df['brand_vocab_idx'] != -1) & (product_df['brand_vocab_idx'] != -2)].shape)
print(product_df[(product_df['color_vocab_idx'] != -1) & (product_df['color_vocab_idx'] != -2)].shape)


#### str(color_name).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
#### 大约有800条数据, 没有被目前"清除表达符号、转化为小写"的方式没有匹配上
#### pretrain 获取到SKU Embedding后, 计算向量之间的相似度,  做交叉熵计算

################### 23w品牌词    23w颜色词     过大的词表是不是太难了?   64dim  先这样
# print(product_df[product_df['brand_vocab_idx'] != -1].shape)
# print(product_df[product_df['color_vocab_idx'] != -1].shape)
############### 有label的数据。品牌数据 和 颜色数据
# 1672180
# 1123196
############### 有label的数据。品牌数据 和 颜色数据出现次数大于一次     15w品牌词，4w的颜色词
# (1531190, 9)
# (935657, 9)

################ 考虑到词表量级、覆盖度。  品牌要求出现大于1次（size=15w），颜色也要求（size=4w）。   全部是128dim维度的词表。
"""
    title + point = sku embeding  将其信息压缩到  128维度的品牌、型号字段中
    sku信息补充一些品牌、型号信息
"""
# 1.删除brand_idx color_idx全为空的数据, 本身也不会产生Loss
less_product_df = product_df[ (product_df['brand_vocab_idx'] >= 0) | (product_df['color_vocab_idx'] >= 0) ]
print(less_product_df.shape)        # (1598616, 9)   160w存在brand or color数据

# 2.写入csv
headers = ['product_id', 'product_title', 'product_bullet_point', 'product_locale', 'product_brand', 'product_color_name', 'brand_vocab_idx', 'color_vocab_idx']
shuffled_all_df = less_product_df.sample(frac=1, random_state=12345).reset_index(drop=True)
less_product_df.to_csv("Product_shuffled_brand_color_vocab.csv", encoding='utf8', index=False, columns=headers)

sample_df = shuffled_all_df.iloc[0:1000]
sample_df.to_csv("Product_shuffled_brand_color_vocab-head1k.csv", encoding='utf8', index=False, columns=headers)

# 3.注意预训练阶段引入 sample-weight

import numpy as np
from collections import OrderedDict
from scipy.special import factorial
import pandas as pd
import string

productid_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
df = pd.read_csv(productid_file, na_values="", keep_default_na=True)
print(df[df['product_locale'] == 'us'][df['product_id'] == 'B003O0MNGC'])

brand_list = df['product_brand'].fillna('').to_list()
color_list = df['product_color_name'].fillna('').to_list()

"""
品牌、颜色字段的最大字符长度 == 100
"""
# max_len=0
# for item in brand_list:
#     max_len = max(max_len, len(item))
# print(max_len)
# max_len=0
# for item in color_list:
#     max_len = max(max_len, len(item))
# print(max_len)


brand_dict = dict()
color_dict = dict()
for item in brand_list:
    process_item = item
    # process_item = str(item).translate(str.maketrans('\n', ' ', string.punctuation))
    if process_item in brand_dict:
        brand_dict[process_item] += 1
    else:
        brand_dict[process_item] = 1

for item in color_list:
    process_item = item
    # process_item = str(item).translate(str.maketrans('\n', ' ', string.punctuation))
    if process_item in color_dict:
        color_dict[process_item] += 1
    else:
        color_dict[process_item] = 1

# brand_list = [item if len(item) < 20 else '' for item in brand_list]
# color_list = [item if len(item) < 20 else '' for item in color_list]

# brand_set = set(brand_list)
# color_set = set(color_list)
# print(len(brand_set), len(color_set))
# print(len(brand_set & color_set), len(brand_set | color_set))

"""
    2022.05.04 这里仅对于出现次数大于1次的, 品牌、颜色制作embedding.    所有的品牌数=30w, 有15w出现2次以上.  所有颜色数目25w,有4w出现2次以上。

    1.删除标点符号, 减少0.2%数据      (没有必要, 不使用)
    2.小写转化, 减少5%的重复数据

"""
brand_list = [item.lower() if value > 1 else '' for item,value in brand_dict.items()]
color_list = [item.lower() if value > 1 else '' for item,value in color_dict.items()]

brand_set = set(brand_list)
color_set = set(color_list)
# print(len(brand_set), len(color_set))
# print(len(brand_set & color_set), len(brand_set | color_set))

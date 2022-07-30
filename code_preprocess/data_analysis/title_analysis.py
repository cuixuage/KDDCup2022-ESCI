"""
    2022.03.21  title分析
        a. 产品词重复
"""
import numpy as np
from collections import OrderedDict
from scipy.special import factorial
import pandas as pd
import string
# #######################################################1.Title数据集合, productid-locale语言重复性
def title_analysis():
    productid_file = "/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
    df = pd.read_csv(productid_file, na_values="", keep_default_na=True)
    print(df[df['product_locale'] == 'us'][df['product_id'] == 'B003O0MNGC'])
    return df

df = title_analysis()       # 数据读取完成

# productid_list = df['product_id'].tolist()
# productid_locale = df['product_locale'].tolist()

# productid_locale_dict = dict()
# for idx, val in enumerate(productid_list):
#     if val in productid_locale_dict.keys():
#         productid_locale_dict[val].add(productid_locale[idx])
#     else:
#         tmp_set = set()
#         tmp_set.add(productid_locale[idx])
#         productid_locale_dict[val] = tmp_set

# single = 0
# double = 0
# triple = 0
# for key, val in productid_locale_dict.items():
#     tokens = list(val)
#     if len(tokens) == 1:
#         single += 1
#     elif len(tokens) == 2:
#         double += 1
#     else:
#         triple += 1
# print(single, double, triple, len(productid_locale_dict.keys()))
# """
# 1791495 10983 585 1803063        一共有1815216行(1791495 + 10983*2 + 585*3)商品, 其中有179w商品是独一无二的, 1w条商品是跨多语言的。

# 总计181w行, 180w独一无二的商品
# """

#######################################################2.header 字段分析

# print(df.isna().sum())
# """
# product_id                   0
# product_title              292
# product_description     878077      12.3%缺失数据
# product_bullet_point    304395      4.2%缺失数据
# product_brand           142988      7.0%缺失数据
# product_color_name      691267      38.5%缺失数据
# product_locale               0
# dtype: int64
# """

#######################################################2.1.description; 产品页面下方, 相当于产品详情
# df['len_title'] = df['product_title'].map(lambda x: len(str(x)) if str(x) != 'NaN' else 0)
# df['len_desc'] = df['product_description'].map(lambda x: len(str(x)) if str(x) != 'NaN' else 0)
# df['len_point'] = df['product_bullet_point'].map(lambda x: len(str(x)) if str(x) != 'NaN' else 0)
# df['len_brand'] = df['product_brand'].map(lambda x: len(str(x)) if str(x) != 'NaN' else 0)
# df['len_color'] = df['product_color_name'].map(lambda x: len(str(x)) if str(x) != 'NaN' else 0)
# print("len分位数: ", df['len_title'].quantile([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]))
# print("len分位数: ", df['len_desc'].quantile([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]))
# print("len分位数: ", df['len_point'].quantile([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]))
# print("len分位数: ", df['len_brand'].quantile([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]))
# print("len分位数: ", df['len_color'].quantile([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]))
#######################################################2.2 bullet_point字段分析; 很重要信息, 产品要点, 500个字符以内  https://www.cifnews.com/article/24377
"""
len分位数:  0.10     35.0
0.30     63.0
0.50     89.0
0.70    127.0
0.90    183.0
0.95    194.0
0.99    200.0
Name: len_title, dtype: float64
len分位数:  0.10       3.0
0.30       3.0
0.50      30.0
0.70     508.0
0.90    1385.0
0.95    1715.0
0.99    1975.0
Name: len_desc, dtype: float64
len分位数:  0.10       3.0
0.30     127.0
0.50     357.0
0.70     705.0
0.90    1239.0
0.95    1511.0
0.99    2045.0
Name: len_point, dtype: float64
len分位数:  0.10     3.0
0.30     6.0
0.50     8.0
0.70    10.0
0.90    16.0
0.95    20.0
0.99    30.0
Name: len_brand, dtype: float64
len分位数:  0.10     3.0
0.30     3.0
0.50     4.0
0.70     6.0
0.90    14.0
0.95    19.0
0.99    33.0
Name: len_color, dtype: float64
"""
#######################################################2.3 brand \ color id化统计
brand_list = df['product_brand'].to_list()
color_list = df['product_color_name'].to_list()
brand_dict = dict()
color_dict = dict()  #305168 253188

# https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
for item in brand_list:
    process_item = str(item).translate(str.maketrans('\n', ' ', string.punctuation))
    if process_item in brand_dict:
        brand_dict[process_item] += 1
    else:
        brand_dict[process_item] = 1

for item in color_list:
    process_item = str(item).translate(str.maketrans('\n', ' ', string.punctuation))
    if process_item in color_dict:
        color_dict[process_item] += 1
    else:
        color_dict[process_item] = 1

sorted_brand_dict = dict(sorted(brand_dict.items(), key=lambda item: item[1]))
sorted_color_dict = dict(sorted(color_dict.items(), key=lambda item: item[1]))
print("brand_nums=", len(sorted_brand_dict.keys()), " color_nums=", len(sorted_color_dict.keys()))

# tmp_str = ""
# for item in color_list:
#     if "Negro" in str(item):
#         tmp_str += str(item) + ','
# print(tmp_str)
# tmp_str = ""
# for item in sorted_color_dict.keys():
#     if "Negro" in str(item):
#         tmp_str += str(item) + ','
# print(tmp_str)

"""
brand\color 没有删除特殊字符
brand_nums= 305168  color_nums= 253188

brand\color 删除特殊字符后
brand_nums= 303626  color_nums= 244102
"""

# #######################################################3.训练集合、公开测试集合，两者title重复性分析
# def title_analysis():
#     train_file = "/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2.csv"
#     test_public_file = "/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/test_public-v0.2.csv"
    
#     train_pid_list = []
#     test_public_pid_list = []
#     unique_pid_dict = dict()
#     unique_pid_public_dict = dict()

#     with open(train_file, mode='r') as fin:
#         for line in fin:
#             tokens = line.strip().split(',')
#             productid = tokens[2]
#             locale = tokens[3]
#             if locale == 'query_locale': continue    # 跳过csv header
#             train_pid_list.append(productid)
#             if productid in unique_pid_dict.keys():
#                 unique_pid_dict[productid].add(locale)
#             else:
#                 locale_set = set()
#                 locale_set.add(locale)
#                 unique_pid_dict[productid] = locale_set

#     with open(test_public_file, mode='r') as fin:
#         for line in fin:
#             tokens = line.strip().split(',')
#             productid = tokens[2]
#             locale = tokens[3]
#             if locale == 'query_locale': continue    # 跳过csv header
#             test_public_pid_list.append(productid)
#             if productid in unique_pid_public_dict.keys():
#                 unique_pid_public_dict[productid].add(locale)
#             else:
#                 locale_set = set()
#                 locale_set.add(locale)
#                 unique_pid_public_dict[productid] = locale_set
    
#     return train_pid_list, test_public_pid_list, unique_pid_dict, unique_pid_public_dict


# train_pid_list, test_public_pid_list, unique_pid_dict, unique_pid_public_dict = title_analysis()

# #1. 计算train、test productid共现情况
# train_set_pid = set(train_pid_list)
# test_set_public_pid = set(test_public_pid_list)
# pid_jiaoji = train_set_pid & test_set_public_pid
# pid_bingji = train_set_pid | test_set_public_pid
# print(len(train_pid_list), len(test_public_pid_list), len(train_set_pid), len(test_set_public_pid))
# print(len(pid_jiaoji), len(pid_bingji))
# """
# 1834744 394367 1345447 352596           #183w行训练集,覆盖134w sku;  39w行验证集,覆盖35w sku                ==> 训练集sku重复率明显更高, 数据集不均衡
# 119967 1578076                          # 训练集、验证集两者的SKU交集是11w, 两者的并集是157W
# """



####################################################### 4.热门度 曝光分析
###  query互相不可见, 分析query有什么用处呢?
###  title
#####  title这个分布信息似乎无法使用, 因为训练集 = 134/183 = 73%    验证集: 35/39 = 89%;   某些通用热门的商品召回出现的概率并不一定多
#####  但是很有可能出现训练集、验证集，正负比例不均衡的问题。   
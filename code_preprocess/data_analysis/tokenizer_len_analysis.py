"""
Query   Titlte  Brand  Color   Bullet  Desc  分词后的长度，百分位统计

bullet卖点信息: 需要信息抽取
desc描述信息: 需要信息抽取
"""

import numpy as np
from collections import OrderedDict
from scipy.special import factorial
import pandas as pd
import string
import os, sys
from transformers import  AutoTokenizer

hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    os.environ['PYTHONHASHSEED'] = '12345'
    os.execv(sys.executable, [sys.executable] + sys.argv)
pd.set_option('display.max_colwidth', None)


tokenizer = AutoTokenizer.from_pretrained('/home/kddcup_2022/huggingface_models/kddcup_2022/infoxlm-base', use_fast=False, truncation_side='right')
def get_tokens_len(item_str):
    return len(tokenizer.tokenize(item_str))
def get_chars_len(item_str):
    return len(item_str)

######################   Title 侧数据分析
task2_train_file = '/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2.csv'
task2_product_file = '/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv'
df = pd.read_csv(task2_product_file, na_values="", keep_default_na=True)
df = df[df['product_locale'] == 'jp'].iloc[10000:20000]

df['product_title_lens'] = df['product_title'].fillna('').map(lambda x: get_tokens_len(x))
df['product_title_chars_lens'] = df['product_title'].fillna('').map(lambda x: get_chars_len(x))
print('111111')
# df['product_description_lens'] = df['product_description'].fillna('').map(lambda x: get_tokens_len(x))
# print('111111')
# df['product_bullet_point_lens'] = df['product_bullet_point'].fillna('').map(lambda x: get_tokens_len(x))
# print('111111')
# df['product_brand_lens'] = df['product_brand'].fillna('').map(lambda x: get_tokens_len(x))
# print('111111')
# df['product_color_name_lens'] = df['product_color_name'].fillna('').map(lambda x: get_tokens_len(x))
# print('111111')


# df_product = df[['product_title_lens','product_description_lens','product_bullet_point_lens','product_brand_lens', 'product_color_name_lens']].copy()
# df_product.describe(include='all', percentiles=[0.5, 0.75, 0.99]).to_csv('destribution_toeknzier.csv')
print(df['product_title_lens'].describe(percentiles=[0.5, 0.75, 0.99]))
print(df['product_title_chars_lens'].describe(percentiles=[0.5, 0.75, 0.99]))

############  采样10w条数据, 数据如下:
# ,product_title_lens,product_description_lens,product_bullet_point_lens,product_brand_lens,product_color_name_lens
# count,100000.0,100000.0,100000.0,100000.0,100000.0
# mean,33.96731,151.93778,136.89341,2.90681,1.79388
# std,16.104784227199385,190.33638923779728,130.95638222061703,1.5948075300833577,2.5165408550982744
# min,0.0,0.0,0.0,0.0,0.0
# 50%,32.0,43.0,100.0,3.0,1.0
# 75%,47.0,287.0,218.0,3.0,2.0
# 99%,68.0,642.0,519.0,9.0,11.0                     # 标题=68, 描述词=642, 卖点词=519, 品牌=9  颜色=11
# max,212.0,3660.0,1063.0,38.0,62.0

# #######################   Query-侧数据分析
# task2_train_file = '/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2.csv'
# df = pd.read_csv(task2_train_file, na_values="", keep_default_na=True)
# df = df[df['query_locale'] == 'jp'].iloc[1000:10000]

# df['query_lens'] = df['query'].fillna('').map(lambda x: get_tokens_len(x))
# df['query_char_lens'] = df['query'].fillna('').map(lambda x: get_chars_len(x))

# print(df['query_lens'].describe(percentiles=[0.5, 0.75, 0.99]))
# print(df['query_char_lens'].describe(percentiles=[0.5, 0.75, 0.99]))


"""
英语每4个字符， 对应一个token
count    9000.000000
mean        8.944111
std         3.266115
min         3.000000
50%         9.000000
75%        11.000000
99%        18.000000
max        21.000000
Name: query_lens, dtype: float64
count    9000.000000
mean       32.454778
std        12.044098
min         7.000000
50%        32.000000
75%        42.000000
99%        57.000000
max        60.000000
Name: query_char_lens, dtype: float64

西班牙每4个字符， 对应一个token
count    9000.000000
mean        5.151667
std         2.183899
min         1.000000
50%         5.000000
75%         7.000000
99%        12.000000
max        15.000000
Name: query_lens, dtype: float64
count    9000.000000
mean       19.068333
std         7.600853
min         3.000000
50%        19.000000
75%        24.000000
99%        39.000000
max        53.000000
Name: query_char_lens, dtype: float64

日语每2个字符， 对应一个token
count    9000.000000
mean        4.736333
std         2.419898
min         1.000000
50%         4.000000
75%         6.000000
99%        12.000000
max        17.000000
Name: query_lens, dtype: float64
count    9000.000000
mean       10.459778
std         5.062488
min         2.000000
50%        10.000000
75%        13.000000
99%        26.000000
max        34.000000
Name: query_char_lens, dtype: float64
"""
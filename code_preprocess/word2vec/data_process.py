"""
    2022.07.14 word2vec 预处理
"""
import pandas as pd
from multiprocesspandas import applyparallel
import string
import os
import sys
import re
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    os.environ['PYTHONHASHSEED'] = '12345'
    os.execv(sys.executable, [sys.executable] + sys.argv)

# 1.data-frame, 清理desc字段、转小写。 拼接多个字段，写入文件
query_df = pd.read_csv('train-v0.2-with-task1.csv')
product_df = pd.read_csv('product_catalogue-v0.2.csv')
pre_df = pd.merge(query_df, product_df, how='left', left_on=['query_locale','product_id'], right_on=['product_locale', 'product_id'])
# pre_df = pre_df.iloc[:10000] 

def desc_cleaner(str_item):
    cleaner = re.compile('<.*?>')    # 删除网页标签, <>内容会被删除
    item_str = re.sub(cleaner, '', str(str_item))
    return item_str
pre_df["product_description_cleaner"] = pre_df["product_description"].fillna('').map(lambda x: desc_cleaner(x))
pre_df["SKU_TEXT"] =  pre_df["product_title"].fillna('') + '  ' + pre_df["product_brand"].fillna('') + '  ' + \
                        pre_df["product_color_name"].fillna('') + '  ' + pre_df["product_bullet_point"].fillna('') + '  ' + pre_df["product_description_cleaner"].fillna('')
# pre_df.to_csv("./sku_text.csv", , columns = ['SKU_TEXT'], encoding='utf8', index=False)

# 2. for循环，开始char3分词
def tokenizer_n_gram(item_str):
    char3_grams_list = []
    n = 3
    if not item_str or str(item_str).isspace() or len(str(item_str)) == 0:
        return []
    else:
        item_str = str(item_str).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
        if len(item_str) <= n:
            char3_grams_list.append(item_str)
        else:
            for i in range(len(item_str) - n):
                i_v = item_str[i:i+n].strip() # 注意这里, n-gram首尾删除空格
                char3_grams_list.append(i_v)
    return ' '.join(char3_grams_list)

print(pre_df.columns)
pre_df["char3_grams"] = pre_df['SKU_TEXT'].apply_parallel(tokenizer_n_gram, num_processes=16)
print(pre_df.columns)

# 3. 重新写入文件
pre_df_shuffle = pre_df.sample(frac=1, random_state=12345).reset_index(drop=True)
pre_df.to_csv("sku_text_char3_gram.csv", columns = ['char3_grams'], encoding='utf8', index=False)
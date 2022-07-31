"""
    20220-04-22
    依赖于SKU-Data数据, 制作Pretain数据集
"""
import numpy as np
from collections import OrderedDict
from scipy.special import factorial
import pandas as pd
import string
from random import randrange, uniform
import datetime
import os,re
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    os.environ['PYTHONHASHSEED'] = '12345'
    os.execv(sys.executable, [sys.executable] + sys.argv)

############################ 1.Test-Public数据集、Test-Private数据集  过采样
"""
    a. 公开测试集SKU - 11w = 20w
    b. 训练集、公开测试集,两者并集 == 157w, 推测出剩余20w SKU数据
    对于以上SKU, 过采样1倍
"""
def product_id_index():
    product_index_map = dict()
    productid_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
    df = pd.read_csv(productid_file, na_values="", keep_default_na=True, usecols=['product_id', 'product_locale'])
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        key = row['product_id'] + '+' + row['product_locale']
        product_index_map[key] = index
    print("all sku len=", len(product_index_map))               # all sku len= 1815216
    return product_index_map

def title_analysis():
    train_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2.csv"
    test_public_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/test_public-v0.2.csv"
    train_pid_list = []
    test_public_pid_list = []

    df = pd.read_csv(train_file, na_values="", keep_default_na=True, usecols=['product_id', 'query_locale'])
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        key = row['product_id'] + '+' + row['query_locale']
        train_pid_list.append(key)

    df = pd.read_csv(test_public_file, na_values="", keep_default_na=True, usecols=['product_id', 'query_locale'])
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        key = row['product_id'] + '+' + row['query_locale']
        test_public_pid_list.append(key)
    
    return train_pid_list, test_public_pid_list

def get_testpublic_20w_productid(product_index_map, train_pid_list, test_public_pid_list):
    product_index_list = []
    train_set_pid = set(train_pid_list)
    test_set_public_pid = set(test_public_pid_list)
    pid_jiaoji = train_set_pid & test_set_public_pid
    print("test-public sku len=", len(list(test_set_public_pid - pid_jiaoji)))   # test-public sku len= 235295
    for item in list(test_set_public_pid - pid_jiaoji):
        product_index_list.append(product_index_map[item])
    return product_index_list

def get_testprivate_20w_productid(product_index_map, train_pid_list, test_public_pid_list):
    product_index_list = []
    all_set_pid = set(product_index_map.keys())
    train_set_pid = set(train_pid_list)
    test_set_public_pid = set(test_public_pid_list)
    pid_bingji = train_set_pid | test_set_public_pid
    print("test-privete sku len=", len(list(all_set_pid - pid_bingji)))     # test-privete sku len= 223987
    for item in list(all_set_pid - pid_bingji):
        product_index_list.append(product_index_map[item])
    return product_index_list

product_index_map = product_id_index()
train_pid_list, test_public_pid_list = title_analysis()
public_index = get_testpublic_20w_productid(product_index_map, train_pid_list, test_public_pid_list)
private_index = get_testprivate_20w_productid(product_index_map, train_pid_list, test_public_pid_list)
productid_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
df = pd.read_csv(productid_file, na_values="", keep_default_na=True)

###数据a
public_df = df.iloc[public_index]
print("public_df=", len(public_df.index))   # public_df= 235295
###数据b
private_df = df.iloc[private_index] 
print("private_df=", len(private_df.index))  # private_df= 223987
###数据汇总
all_df = pd.concat([df, public_df, private_df], axis=0)
print("all_df=", len(all_df.index))        # all_df= 2274498

def desc_cleaner(str_item):
    cleaner = re.compile('<.*?>')    # 删除网页标签, <>内容会被删除
    item_str = re.sub(cleaner, '', str(str_item))
    return item_str

all_df["product_description_cleaner"] = all_df["product_description"].fillna('').map(lambda x: desc_cleaner(x))

############################ 2. 汇总文本信息, 到TEXT列.  2022.04.23 保留:product_id  product_title  product_brand product_color_name product_locale  TEXT  四列
all_df["TEXT"] = all_df["product_locale"].fillna('') + ' </s></s> ' + all_df["product_title"].fillna('') + ' </s></s> '  \
                + all_df["product_brand"].fillna('') + ' </s></s> ' + all_df["product_color_name"].fillna('') + ' </s></s> ' \
                + all_df["product_bullet_point"].fillna('') + ' </s></s> ' + all_df["product_description_cleaner"].fillna('')

all_df.drop(['product_title','product_description','product_bullet_point','product_brand','product_color_name', 'product_description_cleaner'], axis=1, inplace=True)
print("len(all_df.index)=", len(all_df.index))

all_df_us = all_df[all_df['product_locale'] == 'us'].copy()
all_df_jp = all_df[all_df['product_locale'] == 'jp'].copy()
all_df_es = all_df[all_df['product_locale'] == 'es'].copy()
all_df_language = pd.concat([all_df_us, all_df_jp, all_df_es], axis=0)
all_df_language.to_csv("Product_language_220W.csv", encoding='utf8', index=False)
shuffled_all_df = all_df.sample(frac=1, random_state=12345).reset_index(drop=True)
shuffled_all_df.to_csv("Product_shuffled_220W.csv", encoding='utf8', index=False)
print(shuffled_all_df.head())


########################### 3.sample 1w    -- 用于测试Train Valid
productid_file = "/home/kddcup_2022/data_process/Product_shuffled_220W.csv"
df = pd.read_csv(productid_file, na_values="", keep_default_na=True)
sample_df = df.iloc[0:10000]
sample_df.to_csv("Product_shuffled_220W-head1w.csv", encoding='utf8', index=False)
print(sample_df.head())


########################### 4.Eval集合, Fload-5-vaild.csv 计算其MLM Loss
"""
    MLM Loss: From title    Eval耗时23分钟
    NSP Loss:
"""
/home/kddcup_2022/data_process/flod_5_vaild.csv
example_id,query,product_id,query_locale,esci_label,part_id,product_title,product_locale



# ########################### 5.Faked Query集合, New DataLoader
"""
    Faked Query, Title按照泊松分布截取 、 Brand+Color按照泊松分布截取
    丰富Query多样性, 重复进行4遍 
"""
productid_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
# productid_file = "Product_shuffled_220W-head1w.csv"
all_df = pd.read_csv(productid_file, na_values="", keep_default_na=True, usecols=['product_id','product_title','product_brand','product_color_name','product_locale'])
all_df["TEXT"] =  all_df["product_title"].fillna('') + ' ' + all_df["product_brand"].fillna('') \
                + ' ' + all_df["product_color_name"].fillna('') + ' '

all_df = pd.concat([all_df]*4, ignore_index=True)
print(all_df.head(), all_df.shape)

def get_poisson_distribution(lower, upper, miu=8):
    """
        字符长度的采样概率, 服从泊松概率分布
        https://github.com/facebookresearch/SpanBERT/blob/main/pretraining/fairseq/data/masking.py#L104
    """
    len_distrib = [np.power(miu, k) * np.exp(-miu) / factorial(k) for k in range(lower, upper + 1)] if miu >= 0 else None
    len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
    lens = list(range(lower, upper + 1))
    span_len = np.random.choice(lens, p=len_distrib)
    return span_len

def get_rand_query_python_cpu(all_df, mask_in_title_prob=0.5):
    """
        1.获取随机的pos, len
        2.50%概率Query进行twice处理
        3.50%概率Title删除query字符串
    """
    title_str = all_df['TEXT']
    product_locale = all_df['product_locale']
    def get_pos_and_pad(title_len, min_len, max_len, pos_padding):
        pos = int(randrange(start=0, stop=2**30) % title_len)
        pos = max(0, min(pos, title_len-pos_padding))
        if product_locale == 'es' or product_locale == 'us':
            min_len = 8
        else:
            min_len = 4
        rand_lens = get_poisson_distribution(min_len, max_len)
        rand_lens = min(rand_lens, title_len-pos)
        return pos, rand_lens

    def random_span_as_query(title, title_len, min_len=2, max_len=20,
                             pos_padding=5, mask_in_title=False):
        pos, rand_lens = get_pos_and_pad(
            title_len, min_len, max_len, pos_padding)
        query = title[pos:pos+rand_lens]
        if mask_in_title:
            title = title.replace(query, ' ')
        return query, title

    def random_two_span_as_query(title, title_len, min_len=2, max_len=12,
                                 pos_padding=5, mask_in_title=False):
        pos1_org, rand_lens1_org = get_pos_and_pad(
            title_len, min_len, max_len, pos_padding)
        pos2_org, rand_lens2_org = get_pos_and_pad(
            title_len, min_len, max_len, pos_padding)
        if pos1_org < pos2_org:
            pos1 = pos1_org
            pos2 = pos2_org
            rand_lens1 = rand_lens1_org
            rand_lens2 = rand_lens2_org
        else:
            pos1 = pos2_org
            pos2 = pos1_org
            rand_lens1 = rand_lens2_org
            rand_lens2 = rand_lens1_org

        def true_fn(title):
            """merge two substr"""
            sub_len = 0
            if pos1 + rand_lens1 > pos2 + rand_lens2 :
                sub_len = rand_lens1
            else:
                sub_len = pos2 + rand_lens2 - pos1
            query = title[pos1:pos1+sub_len]
            if mask_in_title:
                title = title.replace(query, ' ')
            return query, title

        def false_fn(title):
            """concat three substr"""
            query1 = title[pos1:pos1+rand_lens1]
            query2 = title[pos2:pos2+rand_lens2]
            query = query1 + ' ' + query2
            if mask_in_title:
                title = title.replace(query1, ' ')
                title = title.replace(query2, ' ')
            return query, title

        if pos1 + rand_lens1 >= pos2:
             return true_fn(title)
        else:
            return false_fn(title)

    frand = uniform(0, 1)
    mask_in_title = (frand < mask_in_title_prob)
    title_len = len(title_str)

    frand = uniform(0, 1.0)
    if frand < 0.5:
        query_str, title_str = random_span_as_query(title_str, title_len, mask_in_title=mask_in_title)
    else:
        query_str, title_str = random_two_span_as_query(title_str, title_len, mask_in_title=mask_in_title)
    return query_str,title_str

all_df[['faked_query', 'title_str']] = all_df.apply(get_rand_query_python_cpu, axis=1, result_type="expand")
print(all_df.head(), all_df.shape)

shuffled_all_df = all_df.sample(frac=1, random_state=12345).reset_index(drop=True)
shuffled_all_df.to_csv("Product_shuffled_220W_x4_NSP.csv", encoding='utf8', index=False, columns=['product_id','faked_query', 'title_str', 'product_brand', 'product_color_name', 'product_locale'])
print(shuffled_all_df.head())

########################### 5.Faked Query集合, New DataLoader ---- 相同语言内, faked-query列shuffle, 作为负例
productid_file = "Product_shuffled_220W_x4_NSP.csv"
pos_df = pd.read_csv(productid_file, na_values="", keep_default_na=True)
pos_df['faked_query_label'] = 1
print("pos_df=", len(pos_df.index))
print(pos_df)

neg_df = pos_df.copy()
neg_df['faked_query_label'] = 0

new_df_us = neg_df[neg_df['product_locale'] == 'us'].copy()
new_df_us['faked_query'] = np.random.permutation(new_df_us['faked_query'].values)

new_df_jp = neg_df[neg_df['product_locale'] == 'jp'].copy()
new_df_jp['faked_query'] = np.random.permutation(new_df_jp['faked_query'].values)

new_df_es = neg_df[neg_df['product_locale'] == 'es'].copy()
new_df_es['faked_query'] = np.random.permutation(new_df_es['faked_query'].values)
neg_df = pd.concat([new_df_us, new_df_jp, new_df_es], axis=0)
print("neg_df=", len(neg_df.index))
print(neg_df)

all_df = pd.concat([pos_df, neg_df], axis=0)
shuffled_all_df = all_df.sample(frac=1, random_state=12345).reset_index(drop=True)
shuffled_all_df.to_csv("Product_shuffled_220W_x4_NSP_in_pretrain.csv", encoding='utf8', index=False)
print(shuffled_all_df.head())

shuffled_all_df = pd.read_csv("Product_shuffled_220W_x4_NSP_in_pretrain.csv", na_values="", keep_default_na=True)
sample_df = shuffled_all_df.iloc[0:1000]
sample_df.to_csv("Product_shuffled_220W_x4_NSP_in_pretrain-head1k.csv", encoding='utf8', index=False)
print(sample_df.head())



#####################################  2022.06.15 , FakedData追加Bullet、Desc数据字段
shuffled_all_df = pd.read_csv("Product_shuffled_220W_x4_NSP_in_pretrain.csv", na_values="", keep_default_na=True, usecols=['product_id','faked_query','product_locale', 'faked_query_label'])
sample_df = shuffled_all_df.iloc[0:1000]
# sample_df = shuffled_all_df

# productid_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
# product_df = pd.read_csv(productid_file, na_values="", keep_default_na=True)

# append_df = pd.merge(sample_df, product_df, how='left', left_on=['product_locale','product_id'], right_on=['product_locale', 'product_id'])
# append_df.to_csv("Product_shuffled_220W_x4_NSP_in_pretrain-Append.csv", encoding='utf8', index=False)
# print(sample_df.head())
# print(append_df.head())

# sample_append_df = append_df.iloc[0:1000]
# sample_append_df.to_csv("Product_shuffled_220W_x4_NSP_in_pretrain-Append-head1k.csv", encoding='utf8', index=False)
# print(sample_append_df.head(), sample_append_df.shape)

# print(sample_df.shape, append_df.shape, sample_append_df.shape)

# print(product_df.shape, shuffled_all_df.shape)


print(sample_df['faked_query_label'])

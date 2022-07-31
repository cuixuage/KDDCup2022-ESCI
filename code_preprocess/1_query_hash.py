"""
2022.04.06 划分训练集合、验证集合
query hash, 5折划分验证   --  cv变化更加稳定
"""
import pandas as pd
import string
import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    os.environ['PYTHONHASHSEED'] = '12345'
    os.execv(sys.executable, [sys.executable] + sys.argv)


train_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2.csv"
product_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
query_df = pd.read_csv(train_file)
# product_df = pd.read_csv(product_file, skipinitialspace=True, usecols=['product_id', 'product_title', 'product_locale'])
product_df = pd.read_csv(product_file, skipinitialspace=True)

# 1. query hash
query_df['part_id'] = query_df['query'].map(lambda x: hash(str(x)) % 10)
shuffled_df = query_df.sample(frac=1, random_state=12345).reset_index(drop=True)
print(query_df, shuffled_df)

# 2. join title   (这里先仅仅Join-title数据)
query_train_df = shuffled_df.loc[(shuffled_df['part_id'] != 0) & (shuffled_df['part_id'] != 1)]
query_vaild_df = shuffled_df.loc[shuffled_df['part_id'].isin([0,1])]
train_df = pd.merge(query_train_df, product_df, how='left', left_on=['query_locale','product_id'], right_on=['product_locale', 'product_id'])
vaild_df = pd.merge(query_vaild_df, product_df, how='left', left_on=['query_locale','product_id'], right_on=['product_locale', 'product_id'])
train_df["SKU_TEXT"] =  train_df["product_title"].fillna('') + ' </s> ' + train_df["product_brand"].fillna('') + ' </s> ' + \
                        train_df["product_color_name"].fillna('') + ' </s> ' + train_df["product_bullet_point"].fillna('') + ' </s> ' + train_df["product_description"].fillna('')
vaild_df["SKU_TEXT"] =  vaild_df["product_title"].fillna('') + ' </s> ' + vaild_df["product_brand"].fillna('') + ' </s> ' + \
                        vaild_df["product_color_name"].fillna('') + ' </s> ' + vaild_df["product_bullet_point"].fillna('') + ' </s> ' + vaild_df["product_description"].fillna('')


# 3. write to file
train_df.to_csv("flod_5_train.csv", encoding='utf8', index=False)
vaild_df.to_csv("flod_5_valid.csv", encoding='utf8', index=False)

# 4. generate public test file
query_test_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/test_public-v0.2.csv"
query_test_df = pd.read_csv(query_test_file)
public_test_df = pd.merge(query_test_df, product_df, how='left', left_on=['query_locale','product_id'], right_on=['product_locale', 'product_id'])
public_test_df['part_id'] = '-1'        # 2022.04.20 对齐train、vaild格式
public_test_df['esci_label'] = 'exact'
public_test_df["SKU_TEXT"] =  public_test_df["product_title"].fillna('') + ' </s> ' + public_test_df["product_brand"].fillna('') + ' </s> ' + \
                            public_test_df["product_color_name"].fillna('') + ' </s> ' + public_test_df["product_bullet_point"].fillna('') + ' </s> ' + public_test_df["product_description"].fillna('')
public_test_df.to_csv("flod_5_test.csv", encoding='utf8', index=False)



######################################## 采样1w条数据
productid_vaild_file = "/home/kddcup_2022/data_process/flod_5_train.csv"
df = pd.read_csv(productid_vaild_file, na_values="", keep_default_na=True)
sample_df = df.iloc[0:1000]
sample_df.to_csv("flod_5_train-head1k.csv", encoding='utf8', index=False)
print(sample_df.head())

productid_vaild_file = "/home/kddcup_2022/data_process/flod_5_valid.csv"
df = pd.read_csv(productid_vaild_file, na_values="", keep_default_na=True)
sample_df = df.iloc[0:1000]
sample_df.to_csv("flod_5_valid-head1k.csv", encoding='utf8', index=False)
print(sample_df.head())

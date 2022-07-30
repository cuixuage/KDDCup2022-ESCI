"""
task1 task2数据集  两者的diff

1. query侧diff     (训练集合, query+prodctid+query_locale diff)
2. product侧diff  (productid + locale   diff)
"""
import pandas as pd
import string
import os
import sys
import re, datetime

hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    os.environ['PYTHONHASHSEED'] = '12345'
    os.execv(sys.executable, [sys.executable] + sys.argv)


# # 1.计算Product侧 diff, 涉及到是否需要重新Pretrain
# task2_product_file = '/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv'
# task1_product_file = '/home/cuixuange/kddcup_2022/v0.2_task1/data/processed/public/task_1_query-product_ranking/product_catalogue-v0.2.csv'
# task2_df = pd.read_csv(task2_product_file, na_values="", keep_default_na=True)
# # task2_df = task1_df.iloc[5000:15000]
# task1_df = pd.read_csv(task1_product_file, na_values="", keep_default_na=True)
# # task1_df = task2_df.iloc[5000:15000]
# task1_pids = list()
# task2_pids = list()

# def task1_tokenizer_n_gram(df):
#     global task1_pids
#     pid = df['product_id'] + '+' + df['product_locale']
#     task1_pids.append(pid)
#     return
    
# def task2_tokenizer_n_gram(df):
#     global task2_pids
#     pid = df['product_id'] + '+' + df['product_locale']
#     task2_pids.append(pid)
#     return

# task1_df.apply(task1_tokenizer_n_gram, axis=1)
# task2_df.apply(task2_tokenizer_n_gram, axis=1)

# # print(len(task1_pids), len(task2_pids))

# # pid_jiaoji = set(task1_pids) & set(task2_pids)
# # pid_bingji = set(task1_pids) | set(task2_pids)
# # print(len(pid_jiaoji) / len(set(task1_pids)), len(pid_bingji) / len(set(task2_pids)) )

# # # 883868 1815216
# # # 1.0 1.0


# 2.训练集合两者的diff, 涉及到训练集重复划分的问题
task1_train_file = '/home/cuixuange/kddcup_2022/v0.2_task1/data/processed/public/task_1_query-product_ranking/train-v0.2.csv'
task2_train_file = '/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2.csv'
task3_train_file = '/home/cuixuange/kddcup_2022/v0.2_task3/data/processed/public/task_3_product_substitute_identification/train-v0.2.csv'
task1_df = pd.read_csv(task1_train_file, na_values="", keep_default_na=True)
# task1_df = task1_df.iloc[5000:15000]
task2_df = pd.read_csv(task2_train_file, na_values="", keep_default_na=True)
# task2_df = task2_df.iloc[5000:15000]
task3_df = pd.read_csv(task3_train_file, na_values="", keep_default_na=True)
# task3_df = task3_df.iloc[5000:15000]

task1_pids = list()
task2_pids = list()
task3_pids = list()

def task1_tokenizer_n_gram(df):
    global task1_pids
    pid = df['query'] + '+' + df['product_id'] + '+' + df['query_locale']
    task1_pids.append(pid)
    return
    
def task2_tokenizer_n_gram(df):
    global task2_pids
    pid = df['query'] + '+' + df['product_id'] + '+' + df['query_locale']
    task2_pids.append(pid)
    return

def task3_tokenizer_n_gram(df):
    global task3_pids
    pid = df['query'] + '+' + df['product_id'] + '+' + df['query_locale']
    task3_pids.append(pid)
    return

task1_df.apply(task1_tokenizer_n_gram, axis=1)
task2_df.apply(task2_tokenizer_n_gram, axis=1)
task3_df.apply(task3_tokenizer_n_gram, axis=1)

print(len(task1_pids), len(task2_pids), len(task3_pids))

pid_jiaoji = set(task1_pids) & set(task2_pids)
pid_bingji = set(task1_pids) | set(task2_pids)
print(len(pid_jiaoji) / len(set(task1_pids)), len(pid_bingji) / len(set(task2_pids)) , len(set(task2_pids))/len(set(task3_pids)))

# 781738 1834744 1834744
# 0.7002960071021237 1.127696288964564 1.0


# 3.合并task1、task2数据集合,  相比于task2应该是多了12%的数据量

# 3.1 task1_df 删除交集数据
def judge(df):
    global pid_jiaoji
    pid = df['query'] + '+' + df['product_id'] + '+' + df['query_locale']
    if pid in pid_jiaoji:
        return 'in'
    else:
        return 'out'
task1_df['filter'] = task1_df.apply(judge, axis=1, result_type="expand")
task1_df_filter = task1_df[task1_df['filter'] == 'out'].copy()
task1_df_filter.drop(['filter','query_id'], axis=1, inplace=True)
task1_df_filter['example_id'] = 'from_task1'
# task1_df_filter = task1_df_filter.iloc[100:110]

# 3.2 task1_df剩余数据  +  task2数据;  +task3数据
task2_df_extra = pd.concat([task2_df, task1_df_filter], axis=0)
print("task2-new=", task2_df.shape, task1_df_filter.shape, task2_df_extra.shape)
task2_df_extra.to_csv('/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2-with-task1.csv', encoding='utf8', index=False)


conversion_dict = {'irrelevant':'no_substitute', 'exact':'no_substitute' , 'substitute':'substitute', 'complement':'no_substitute'}
task1_df_filter['substitute_label'] = task1_df_filter['esci_label'].replace(conversion_dict)
# task1_df_filter = task1_df_filter.rename({'esci_label':'substitute_label'}, axis='columns')
# task1_df_filter.columns = ['example_id','query','product_id','query_locale','substitute_label']
task1_df_filter.drop('esci_label', axis=1, inplace=True)
task3_df_extra = pd.concat([task3_df, task1_df_filter], axis=0)
print("task3-new=", task3_df.shape, task1_df_filter.shape, task3_df_extra.shape)
task3_df_extra.to_csv('/home/cuixuange/kddcup_2022/v0.2_task3/data/processed/public/task_3_product_substitute_identification/train-v0.2-with-task1.csv', encoding='utf8', index=False)

# 3.3 重新划分训练集、验证集
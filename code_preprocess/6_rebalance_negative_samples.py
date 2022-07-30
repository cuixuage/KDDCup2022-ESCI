"""
    1. hard-negatives hard-postives 数据均难以构造
    2. 基于Fload-5-Train训练数据, 对于Complement\Irrelevant数据集, 扩充其数据样本
    3. 数据集平衡， Query词下样本同一个类目数据过多。


    "New-Data"  仅来自于product_bullet_point、product_description
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


###### 1.扩充Complement\Irrelevant数据集
flod_5_train_file = "./extra_data_with_task1/task2/flod_5_train_with_optm_trans.csv"
flod_5_valid_file = './extra_data_with_task1/task2/flod_5_valid.csv'
train_df = pd.read_csv(flod_5_train_file)
valid_df = pd.read_csv(flod_5_valid_file)
df = pd.concat([train_df, valid_df], axis=0)
print("init= task2_task1 train_df", df.shape)
# df = df.iloc[:100000]

exact_df = df[df['esci_label'] == 'exact'].copy()
substitute_df = df[df['esci_label'] == 'substitute'].copy()
complement_df = df[df['esci_label'] == 'complement'].copy()
irrelevant_df = df[df['esci_label'] == 'irrelevant'].copy()
print("class=", complement_df.shape, irrelevant_df.shape, substitute_df.shape, exact_df.shape)

"""
US、ES = 平均Title约33 Tokens  , Query约10个Token    => 200 Tokens  => 每4个字符约1个Token  => 800字符
JP =  平均Title约67 Tokens, Query约20个Token  =>  150 Tokens  => 每2个字符约1个Token  => 300字符

将BulletPoint + Desc    max_length = 800 或者 360得到一批新数据    
"""

c_append_df = pd.DataFrame(columns=['query','product_id','query_locale','esci_label','product_title','product_locale'])
i_append_df = pd.DataFrame(columns=['query','product_id','query_locale','esci_label','product_title','product_locale'])
s_append_df = pd.DataFrame(columns=['query','product_id','query_locale','esci_label','product_title','product_locale'])
e_append_df = pd.DataFrame(columns=['query','product_id','query_locale','esci_label','product_title','product_locale'])

def c_append_data(df):
    """
        区分语言、不相关的Label进行数据填充
    """
    global c_append_df
    query_locale = df['query_locale']
    title = df['product_title']
    p_b = df['product_bullet_point']
    p_b = p_b if p_b and not str(p_b).isspace() and len(str(p_b)) > 0 else ''
    p_d = df['product_description']
    p_d = p_d if p_d and not str(p_d).isspace() and len(str(p_d)) > 0 else ''
    cleaner = re.compile('<.*?>')
    p_d = re.sub(cleaner, '', str(p_d))
    enough_data = str(p_b) + ' ' + str(p_d)
    append_data = []
    split_len = 800

    if query_locale == 'jp':
        split_len = 300
    elif query_locale == 'us' or query_locale == 'es':
        split_len = 800
    else:
        return 
    if len(enough_data) < split_len:
        return 

    split_num = int(len(enough_data) / split_len) + 1
    chunks, chunk_size = len(enough_data), len(enough_data) // split_num
    append_data = [ enough_data[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    for data in append_data:
        new_row = [df['query'] ,  df['product_id'],  df['query_locale'] + '-append',  df['esci_label'],  str(title) + ' ' + str(data),  df['product_locale']]
        # new_row = [df['query'] ,  df['product_id'],  df['query_locale'] + '-append',  df['esci_label'],  str(data),  df['product_locale']]
        c_append_df.loc[len(c_append_df.index)] = new_row
    return 

def i_append_data(df):
    """
        区分语言、不相关的Label进行数据填充
    """
    global i_append_df
    query_locale = df['query_locale']
    title = df['product_title']
    p_b = df['product_bullet_point']
    p_b = p_b if p_b and not str(p_b).isspace() and len(str(p_b)) > 0 else ''
    p_d = df['product_description']
    p_d = p_d if p_d and not str(p_d).isspace() and len(str(p_d)) > 0 else ''
    cleaner = re.compile('<.*?>')
    p_d = re.sub(cleaner, '', str(p_d))
    enough_data = str(p_b) + ' ' + str(p_d)
    append_data = []
    split_len = 800

    if query_locale == 'jp':
        split_len = 300
    elif query_locale == 'us' or query_locale == 'es':
        split_len = 800
    else:
        return 
    if len(enough_data) < split_len:
        return 

    split_num = int(len(enough_data) / split_len) + 1
    chunks, chunk_size = len(enough_data), len(enough_data) // split_num
    append_data = [ enough_data[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    for data in append_data:
        new_row = [df['query'] ,  df['product_id'],  df['query_locale'] + '-append',  df['esci_label'],  str(title) + ' ' + str(data),  df['product_locale']]
        # new_row = [df['query'] ,  df['product_id'],  df['query_locale'] + '-append',  df['esci_label'],  str(data),  df['product_locale']]
        i_append_df.loc[len(i_append_df.index)] = new_row
    return 

def s_append_data(df):
    """
        区分语言、不相关的Label进行数据填充
    """
    global s_append_df
    query_locale = df['query_locale']
    title = df['product_title']
    p_b = df['product_bullet_point']
    p_b = p_b if p_b and not str(p_b).isspace() and len(str(p_b)) > 0 else ''
    p_d = df['product_description']
    p_d = p_d if p_d and not str(p_d).isspace() and len(str(p_d)) > 0 else ''
    cleaner = re.compile('<.*?>')
    p_d = re.sub(cleaner, '', str(p_d))
    enough_data = str(p_b) + ' ' + str(p_d)
    append_data = []
    split_len = 800

    if query_locale == 'jp':
        split_len = 300
    elif query_locale == 'us' or query_locale == 'es':
        split_len = 800
    else:
        return 
    if len(enough_data) < split_len:
        return 

    split_num = int(len(enough_data) / split_len) + 1
    chunks, chunk_size = len(enough_data), len(enough_data) // split_num
    append_data = [ enough_data[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    for data in append_data:
        new_row = [df['query'] ,  df['product_id'],  df['query_locale'] + '-append',  df['esci_label'],  str(title) + ' ' + str(data),  df['product_locale']]
        # new_row = [df['query'] ,  df['product_id'],  df['query_locale'] + '-append',  df['esci_label'],  str(data),  df['product_locale']]
        s_append_df.loc[len(s_append_df.index)] = new_row
    return 

def e_append_data(df):
    """
        区分语言、不相关的Label进行数据填充
    """
    global e_append_df
    query_locale = df['query_locale']
    title = df['product_title']
    p_b = df['product_bullet_point']
    p_b = p_b if p_b and not str(p_b).isspace() and len(str(p_b)) > 0 else ''
    p_d = df['product_description']
    p_d = p_d if p_d and not str(p_d).isspace() and len(str(p_d)) > 0 else ''
    cleaner = re.compile('<.*?>')
    p_d = re.sub(cleaner, '', str(p_d))
    enough_data = str(p_b) + ' ' + str(p_d)
    append_data = []
    split_len = 800

    if query_locale == 'jp':
        split_len = 400
    elif query_locale == 'us' or query_locale == 'es':
        split_len = 800
    else:
        return 
    if len(enough_data) < split_len:
        return 

    split_num = int(len(enough_data) / split_len) + 1
    chunks, chunk_size = len(enough_data), len(enough_data) // split_num
    append_data = [ enough_data[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    for data in append_data:
        new_row = [df['query'] ,  df['product_id'],  df['query_locale'] + '-append',  df['esci_label'],  str(title) + ' ' + str(data),  df['product_locale']]
        # new_row = [df['query'] ,  df['product_id'],  df['query_locale'] + '-append',  df['esci_label'],  str(data),  df['product_locale']]
        e_append_df.loc[len(e_append_df.index)] = new_row
    return 

# complement_df.apply(c_append_data, axis=1)
# print('c_append_df=', c_append_df.shape)

irrelevant_df.apply(i_append_data, axis=1)
print('i_append_df=', i_append_df.shape)

# substitute_df.apply(s_append_data, axis=1)
# print('s_append_df=', s_append_df.shape)

# exact_df.apply(e_append_data, axis=1)
# print('e_append_df=', e_append_df.shape)

c_append_df['example_id'] = '-1'
c_append_df['part_id'] = '-1'
c_append_df['product_description'] = ''
c_append_df['product_bullet_point'] = ''
c_append_df['product_brand'] = ''
c_append_df['product_color_name'] = ''

i_append_df['example_id'] = '-1'
i_append_df['part_id'] = '-1'
i_append_df['product_description'] = ''
i_append_df['product_bullet_point'] = ''
i_append_df['product_brand'] = ''
i_append_df['product_color_name'] = ''

s_append_df['example_id'] = '-1'
s_append_df['part_id'] = '-1'
s_append_df['product_description'] = ''
s_append_df['product_bullet_point'] = ''
s_append_df['product_brand'] = ''
s_append_df['product_color_name'] = ''

# e_append_df['example_id'] = '-1'
# e_append_df['part_id'] = '-1'
# e_append_df['product_description'] = ''
# e_append_df['product_bullet_point'] = ''
# e_append_df['product_brand'] = ''
# e_append_df['product_color_name'] = ''

all_df = pd.concat([df, c_append_df, i_append_df, s_append_df], axis=0)
print("result=", all_df.shape, c_append_df.shape, i_append_df.shape, s_append_df.shape)


# shuffled_all_df = all_df.sample(frac=1, random_state=12345).reset_index(drop=True)
# shuffled_all_df.to_csv("./extra_data_with_task1_rebalance/task2/rebalanced_public_data_sci.csv", encoding='utf8', index=False)          #  2022.06.13  训练集合、验证集合都在一起
# c_append_df.to_csv("./extra_data_with_task1_rebalance/task2/tmp_c_sci_append_df.csv", encoding='utf8', index=False)
i_append_df.to_csv("./extra_data_with_task1_rebalance/task2/tmp_i_append_df.csv", encoding='utf8', index=False)
# s_append_df.to_csv("./extra_data_with_task1_rebalance/task2/tmp_s_sci_append_df.csv", encoding='utf8', index=False)
# e_append_df.to_csv("./extra_data_with_task1_rebalance/task2/tmp_e_sci_append_df.csv", encoding='utf8', index=False)





#### 汇总 complement、substitute、irrelevant数据集
rebalanced_public_data_only_sc = pd.read_csv("./extra_data_with_task1_rebalance/task2/rebalanced_public_data_only_sc.csv")
tmp_i_append_df = pd.read_csv("./extra_data_with_task1_rebalance/task2/tmp_i_append_df.csv")
print(tmp_i_append_df.head)
print(rebalanced_public_data_only_sc.head)

rebalanced_public_data_sci = pd.concat([rebalanced_public_data_only_sc, tmp_i_append_df], axis=0)
shuffled_all_df = rebalanced_public_data_sci.sample(frac=1, random_state=12345).reset_index(drop=True)
print(rebalanced_public_data_only_sc.shape, tmp_i_append_df.shape, shuffled_all_df.shape)

shuffled_all_df.to_csv("./extra_data_with_task1_rebalance/task2/rebalanced_public_data_only_sci.csv", encoding='utf8', index=False)
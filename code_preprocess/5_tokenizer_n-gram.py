"""
    1.query2search  char 3-gram
    2.taobao search  1-gram、2-gram 

    InfoXLM 使用小写会使得效果变差 -0.1%
    但是对于char 3-ngram来说, 我们认为减少词表大小的收益, 能抵消lower()带来的效果
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

def write_dict_2_file(file_name, query_dict):
    with open(file_name, mode='w') as fout:
        for k,v in query_dict.items():
            fout.write(str(k) + '\t' + str(v) + '\n')

product_ngrams = dict()
def tokenizer_n_gram(df):
    wait_tokenizer = {'title':df['product_title'], 'brand':df['product_brand'], 'color':df['product_color_name'],
                        'point':df['product_bullet_point'], 'desc':df['product_description']}
    # wait_tokenizer = {'desc':df['product_description']}

    global product_ngrams
    n = 3

    cleaner = re.compile('<.*?>')    # 删除网页标签, <>内容会被删除
    for key, item_str in wait_tokenizer.items():  
        if not item_str or str(item_str).isspace() or len(str(item_str)) == 0:
            continue
        else:
            if key == 'desc':
                item_str = re.sub(cleaner, '', str(item_str))
            item_str = str(item_str).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
            if len(item_str) <= n:
                if item_str in product_ngrams.keys():
                    product_ngrams[item_str] += 1
                else:
                    product_ngrams[item_str] = 1 
            else:
                for i in range(len(item_str) - n):
                    i_v = item_str[i:i+n].strip() # 注意这里, n-gram首尾删除空格
                    if i_v in product_ngrams.keys():
                        product_ngrams[i_v] += 1
                    else:
                        product_ngrams[i_v] = 1  
    
    return 'nan'

# ################ 1.product char-3-gram分词     （小写、删除特殊字符、删除网页字符）
# productid_file = "/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/product_catalogue-v0.2.csv"
# df = pd.read_csv(productid_file, na_values="", keep_default_na=True)
# # df = df[df['product_locale'] == 'es'].iloc[5000:15000]
# print(df.shape)

# start_t = datetime.datetime.now()
# df['product_ngrams'] = df.apply(tokenizer_n_gram, axis=1)
# end_t = datetime.datetime.now()
# print((end_t - start_t).seconds)

# print(len(product_ngrams.keys()))
# print(list(product_ngrams.keys())[0:10], list(product_ngrams.keys())[-10:])
# write_dict_2_file('./extra_ngram/product_ngram.csv', product_ngrams)


################ 1.query char-3-gram分词     （小写、删除特殊字符）
query_train_ngrams = dict()
query_test_ngrams = dict()

def query_train_tokenizer_n_gram(df):
    item_str = df['query']
    global query_train_ngrams
    n = 3

    if not item_str or str(item_str).isspace() or len(str(item_str)) == 0:
        return 'nan'
    else:
        item_str = str(item_str).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
        if len(item_str) <= n:
            if item_str in query_train_ngrams.keys():
                query_train_ngrams[item_str] += 1
            else:
                query_train_ngrams[item_str] = 1 
        else:
            for i in range(len(item_str) - n): 
                i_v =item_str[i:i+n].strip() # 注意这里, n-gram首尾删除空格
                if i_v in query_train_ngrams.keys():
                    query_train_ngrams[i_v] += 1
                else:
                    query_train_ngrams[i_v] = 1               
    return 'nan'

def query_test_tokenizer_n_gram(df):
    item_str = df['query']
    global query_test_ngrams
    n = 3

    if not item_str or str(item_str).isspace() or len(str(item_str)) == 0:
        return 'nan'
    else:
        item_str = str(item_str).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
        if len(item_str) <= n:
            if item_str in query_test_ngrams.keys():
                query_test_ngrams[item_str] += 1
            else:
                query_test_ngrams[item_str] = 1 
        else:
            for i in range(len(item_str) - n): 
                i_v =item_str[i:i+n].strip() # 注意这里, n-gram首尾删除空格
                if i_v in query_test_ngrams.keys():
                    query_test_ngrams[i_v] += 1
                else:
                    query_test_ngrams[i_v] = 1               
    return 'nan'

# query_file = '/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2.csv'
# df = pd.read_csv(query_file, na_values="", keep_default_na=True)
# df['query_ngram'] = df.apply(query_train_tokenizer_n_gram, axis=1)
# print(len(query_train_ngrams.keys()))
# print(list(query_train_ngrams.keys())[0:10], list(query_train_ngrams.keys())[-10:])
# write_dict_2_file('./extra_ngram/query_ngram_train.csv', query_train_ngrams)

# query_file = '/home/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/test_public-v0.2.csv'
# df = pd.read_csv(query_file, na_values="", keep_default_na=True)
# df['query_ngram'] = df.apply(query_test_tokenizer_n_gram, axis=1)
# print(len(query_test_ngrams.keys()))
# print(list(query_test_ngrams.keys())[0:10], list(query_test_ngrams.keys())[-10:])
# write_dict_2_file('./extra_ngram/query_ngram_test.csv', query_test_ngrams)

# query_file = '/home/kddcup_2022/v0.2_task1/data/processed/public/task_1_query-product_ranking/train-v0.2.csv'
# df = pd.read_csv(query_file, na_values="", keep_default_na=True)
# df['query_ngram'] = df.apply(query_train_tokenizer_n_gram, axis=1)
# print(len(query_train_ngrams.keys()))
# print(list(query_train_ngrams.keys())[0:10], list(query_train_ngrams.keys())[-10:])
# write_dict_2_file('./extra_ngram/query_ngram_train_task1.csv', query_train_ngrams)

# query_file = '/home/kddcup_2022/v0.2_task1/data/processed/public/task_1_query-product_ranking/test_public-v0.2.csv'
# df = pd.read_csv(query_file, na_values="", keep_default_na=True)
# df['query_ngram'] = df.apply(query_test_tokenizer_n_gram, axis=1)
# print(len(query_test_ngrams.keys()))
# print(list(query_test_ngrams.keys())[0:10], list(query_test_ngrams.keys())[-10:])
# write_dict_2_file('./extra_ngram/query_ngram_test_task1.csv', query_test_ngrams)

################ 3.分析  Product 、 Query两侧的char-3-gram的重复度
product_ngrams_file = './extra_ngram/product_ngram.csv'
query_train_ngrams_file = './extra_ngram/query_ngram_train.csv'
query_test_ngrams_file = './extra_ngram/query_ngram_test.csv'
query_train_ngrams_file_taks1 = './extra_ngram/query_ngram_train_task1.csv'
query_test_ngrams_file_taks1 = './extra_ngram/query_ngram_test_task1.csv'

product_dict = dict()
query_train = dict()
query_test = dict()
query_train_task1 = dict()
query_test_task1 = dict()

with open(product_ngrams_file, mode='r') as fin:
    for line in fin:
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            print(tokens)
            continue
        product_dict[tokens[0]] = tokens[1]
with open(query_train_ngrams_file, mode='r') as fin:
    for line in fin:
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            print(tokens)
            continue
        query_train[tokens[0]] = tokens[1]
with open(query_test_ngrams_file, mode='r') as fin:
    for line in fin:
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            print(tokens)
            continue
        query_test[tokens[0]] = tokens[1]
with open(query_train_ngrams_file_taks1, mode='r') as fin:
    for line in fin:
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            print(tokens)
            continue
        query_train_task1[tokens[0]] = tokens[1]
with open(query_test_ngrams_file_taks1, mode='r') as fin:
    for line in fin:
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            print(tokens)
            continue
        query_test_task1[tokens[0]] = tokens[1]

product_set = set(product_dict.keys())
query_train_set = set(query_train.keys())
query_test_set = set(query_test.keys())
query_train_set_task1 = set(query_train_task1.keys())
query_test_set_task1 = set(query_test_task1.keys())

jiaoji_train = product_set & query_train_set
jiaoji_test = product_set & query_test_set 
jiaoji_train_test = query_test_set & query_train_set

print(len(product_set), len(query_train_set), len(query_test_set))
print(len(jiaoji_train) / float(len(query_train_set)),  len(jiaoji_test) / float(len(query_test_set)),  len(jiaoji_train_test) / float(len(query_test_set)))
# 5642462 50504 19583
# 0.9426184064628544 0.9692079865189195 0.6582750344686719

product_dict_more = dict()
query_train_more = dict()
query_test_more = dict()

for k, v in product_dict.items():
    if int(v) >= 14:
        product_dict_more[k] = v
for k, v in query_train.items():
    if int(v) >= 20:
        query_train_more[k] = v
for k, v in query_test.items():
    if int(v) >= 20:
        query_test_more[k] = v

product_dict = product_dict_more
# query_train = query_train_more
# query_test = query_test_more

product_set = set(product_dict.keys())
query_train_set = set(query_train.keys())
query_test_set = set(query_test.keys())
jiaoji_train = product_set & query_train_set
jiaoji_test = product_set & query_test_set 
jiaoji_train_test = query_test_set & query_train_set

print(len(product_set), len(query_train_set), len(query_test_set))
print(len(jiaoji_train) / float(len(query_train_set)),  len(jiaoji_test) / float(len(query_test_set)),  len(jiaoji_train_test) / float(len(query_test_set)))

###############
"""
Embedding Bag = 1.Product-n-grams出现次数>=5   2.所有的训练集合Query-n-grams
汇总有160w词表
"""

res_bag = set(product_dict_more.keys()) | set(query_train.keys()) | set(query_test.keys()) | set(query_train_task1.keys()) | set(query_test_task1.keys())
print(len(set(product_dict_more.keys())), len(set(query_train.keys())), len(res_bag))

with open('./extra_ngram/emb_bag_vocab.txt', mode='w') as fout:
    fout.write('[UNK]' + '\n')
    for item in list(res_bag):
        fout.write(item + '\n')
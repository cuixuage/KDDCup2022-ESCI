"""
    2022.03.21
    i. query分析
        统计Query、Ttitle在不同语言下的min、max、avg长度分布
    ii. title分析
        a. 产品词重复
"""
import numpy as np
from collections import OrderedDict
from scipy.special import factorial
import matplotlib.pyplot as plt

#######################################################1.train、eval query共现
def query_analysis():
    train_file = "/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2.csv"
    test_public_file = "/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/test_public-v0.2.csv"
    
    train_query_list = []
    test_public_query_list = []

    unique_query_dict = dict()
    unique_query_public_dict = dict()

    with open(train_file, mode='r') as fin:
        for line in fin:
            tokens = line.strip().split(',')
            query = tokens[1]
            locale = tokens[3]
            if locale == 'query_locale': continue    # 跳过csv header
            train_query_list.append(query)
            if query in unique_query_dict.keys():
                unique_query_dict[query].add(locale)
            else:
                locale_set = set()
                locale_set.add(locale)
                unique_query_dict[query] = locale_set

    with open(test_public_file, mode='r') as fin:
        for line in fin:
            tokens = line.strip().split(',')
            query = tokens[1]
            locale = tokens[3]
            if query == 'query_locale': continue    # 跳过csv header
            test_public_query_list.append(query)
            if query in unique_query_public_dict.keys():
                unique_query_public_dict[query].add(locale)
            else:
                locale_set = set()
                locale_set.add(locale)
                unique_query_public_dict[query] = locale_set
    
    return train_query_list, test_public_query_list, unique_query_dict, unique_query_public_dict


train_query_list, test_public_query_list, unique_query_dict, unique_query_public_dict = query_analysis()

#1. 计算train、test query共现情况
train_set_query = set(train_query_list)
test_set_public_query = set(test_public_query_list)
query_jiaoji = train_set_query & test_set_public_query
print(len(train_query_list), len(test_public_query_list), len(train_set_query), len(test_set_public_query))
print(len(query_jiaoji))
"""
1834745 394368    unique queries = 91176 19584     183w行训练集, 9.1w query;  39w行验证集, 1.9w query   ===>  训练、验证query的平均深度为20个
85    ==>  训练集 、 test_public榜单的query几乎没有交集；  仅有85条query是存在交集的。
"""


#######################################################2. 语言多样性
single = 0
double = 0
triple = 0
for key, val in unique_query_dict.items():
    tokens = list(val)
    if len(tokens) == 1:
        single += 1
    elif len(tokens) == 2:
        double += 1
    else:
        triple += 1
print(single, double, triple, len(unique_query_dict.keys()))
single = 0
double = 0
triple = 0
for key, val in unique_query_public_dict.items():
    tokens = list(val)
    if len(tokens) == 1:
        single += 1
    elif len(tokens) == 2:
        double += 1
    else:
        triple += 1
print(single, double, triple, len(unique_query_public_dict.keys()))
"""
90778 221 177 91176         =>  训练集合有9w unique query, 公开测试集有1.9w unique query
19530 13 41 19584     ==> query几乎不会跨语言出现, 仅有0.2% ~ 0.3% query出现在两个语言以上

官方公布:
不同语言的Query数量也是不同的, 但是train、vaild、test集合上Query数量级分布一致的。
https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search#dataset-(shopping-queries-data-set

"""

#######################################################3. unique queries分布
def query_distribution(unique_query_list, locale):
    line_num = len(unique_query_list)
    p_dict = dict()
    order_p_dict_distrib = []
    if locale == 'es' or locale == 'us':       #按照空格分词, 计算单词长度
        for query in unique_query_list:
            list_q_tokens = query.split(' ')
            if len(list_q_tokens) in p_dict:
                p_dict[len(list_q_tokens)] += 1
            else:
                p_dict[len(list_q_tokens)] = 1
    if locale == 'jp':       #按照character, 计算单词长度
        for query in unique_query_list:
            if len(query) in p_dict:
                p_dict[len(query)] += 1
            else:
                p_dict[len(query)] = 1

    order_p_dict = OrderedDict(sorted(p_dict.items()))
    for key, value in order_p_dict.items():
        order_p_dict_distrib.append(float(value) / line_num)
    # print(len(order_p_dict.keys()), len(order_p_dict_distrib))
    # print(order_p_dict_distrib)
    # print(order_p_dict.keys())
    return list(order_p_dict.keys()), order_p_dict_distrib

def get_plot(lower=1, upper=30, len_distrib_g=[], len_distrib_p=[], es_lens=[], es_dis=[], us_lens=[], us_dis=[], jp_lens=[], jp_dis=[]):
    lens = list(range(lower, upper + 1))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("query_len_distribution")
    plt.plot(lens, len_distrib_g, label = 'geometric_p=0.2')
    plt.plot(lens, len_distrib_p, label = 'poisson_miu=4')
    plt.plot(es_lens[:30], es_dis[:30], label = 'es')                     # 76是所有语言中的最长字符串, 从图片看, 30长度以上就基本趋于零
    plt.plot(us_lens[:30], us_dis[:30], label = 'us')
    plt.plot(jp_lens[:30], jp_dis[:30], label = 'jp')
    plt.legend()
    plt.show()
    plt.savefig('query_len_distribution.png')

def get_geometric_distribution(lower=1, upper=30, p=0.2):
    """
        字符长度的采样概率, 服从几何概率分布
        Return: [p1, p2, p3....., p10]  举例: P3=字符长度为3的概率
    """
    len_distrib = [p * (1-p)**(i - lower) for i in range(lower, upper + 1)] if p >= 0 else None
    len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
    return len_distrib

def get_poisson_distribution(lower=1, upper=30, miu=4):
    """
        字符长度的采样概率, 服从泊松概率分布
    """
    len_distrib = [np.power(miu, k) * np.exp(-miu) / factorial(k) for k in range(lower, upper + 1)] if miu >= 0 else None
    len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
    return len_distrib

unique_q_es = list()
unique_q_us = list()
unique_q_jp = list()
for q, v in unique_query_dict.items():            # Train集合
# for q, v in unique_query_public_dict.items():       # public test集合
    list_v = list(v)
    if len(list_v) == 1:
        locale = list_v[0]
        if locale == 'es':
            unique_q_es.append(q)
        elif locale == 'us':
            unique_q_us.append(q)   
        else:
            unique_q_jp.append(q)   # 按照character分词


es_lens, es_dis = query_distribution(unique_q_es, 'es')
us_lens, us_dis = query_distribution(unique_q_us, 'us')
jp_lens, jp_dis = query_distribution(unique_q_jp, 'jp')

len_distrib_g = get_geometric_distribution()
len_distrib_p = get_poisson_distribution()

print(len(es_lens), max(es_lens), np.mean(es_lens), np.median(es_lens))
print(len(us_lens), max(us_lens), np.mean(us_lens), np.median(us_lens))
print(len(jp_lens), max(jp_lens), np.mean(jp_lens), np.median(jp_lens)) 
"""
训练集: 
12 12 6.5 6.5
27 29 14.481481481481481 14.0
61 76 31.24590163934426 31.0    分词后有多少种长度, 长度最大值, 长度平均值, 长度中位数

1.西班牙语、英语按照空格分词, 最大长度不超过30;  虽然英语的中位数、平均值都超过西班牙语;       但是这种信息似乎我无法使用上。。。。。。。
2.日语我按照character分词计算
"""

get_plot(len_distrib_g=len_distrib_g, len_distrib_p=len_distrib_p,          #2022.02.15 几何分布、泊松分布
        es_lens=es_lens, es_dis=es_dis,                                     #2022.02.15 训练集合，西班牙语分布
        us_lens=us_lens,us_dis=us_dis,                                      #2022.02.15 训练集合，英语分布
        jp_lens=jp_lens,jp_dis=jp_dis)                                      #2022.02.15 训练集合，日语分布

"""
训练集:
西班牙语、英语按照空格分词,  日语我按照character分词计算

西班牙语、英语  =  泊松分布, miu=4
日语  =  泊松分布, miu=8
"""



#######################################################3. 训练集合，Query词下sku的正负例统计, 尽量1:1

def get_positive_num(val_list = []):
    count = 0
    for item in val_list:
        if int(item) > 0.5:
            count += 1
    return count

def get_need_balance_query(input_file=""):
    input_file = "/home/cuixuange/kddcup_2022/v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2.csv"
    query_dict = dict()
    with open(input_file, mode='r') as fin:
        for line in fin:
            tokens = line.strip().split(',')
            query = tokens[1]
            esci_label = tokens[-1]
            label = -1
            locale = tokens[3]
            if locale == 'query_locale': continue    # 跳过csv header

            if esci_label == 'exact' or esci_label == 'substitute':
                label = 1
            elif esci_label == 'complement' or esci_label == 'irrelevant':
                label = 0
            else:
                print('error')
                break
            if query in query_dict:
                query_dict[query].append(label)
            else:
                query_dict[query] = []
                query_dict[query].append(label)
   
    all_count = 0
    prob_cout = 0
    need_balance_query = []
    for key,val_list in query_dict.items():
        all_count += len(val_list)
        pos_num = get_positive_num(val_list)
        prob = pos_num / float(len(val_list))
        if len(val_list) >= 4 and (prob <= 0.2 or prob >= 0.8):   # 约50%的数据,需要重新平衡
            prob_cout += len(val_list)
            need_balance_query.append(key + '\t' + str(len(val_list)) + '\t' + str(prob))
    print('get_need_balance_query=', all_count, prob_cout, prob_cout/all_count)
    return need_balance_query

need_balance_query = get_need_balance_query()
print(len(need_balance_query))

"""
按照Query维度统计, 约81%的训练数据; 75855/9.1w既83%的query数据存在不平衡
get_need_balance_query= 1834744 1486543 0.8102182102789272
75855

训练集类别严重不平衡:
[cuixuange@nb-cuixuange-test task_2_multiclass_product_classification]$ cat train-v0.2.csv  | grep 'exact' | wc -l 
1196533
[cuixuange@nb-cuixuange-test task_2_multiclass_product_classification]$ cat train-v0.2.csv  | grep 'substitute' | wc -l 
401612
[cuixuange@nb-cuixuange-test task_2_multiclass_product_classification]$ cat train-v0.2.csv  | grep 'irrelevant' | wc -l 
183413
[cuixuange@nb-cuixuange-test task_2_multiclass_product_classification]$ cat train-v0.2.csv  | grep 'complement' | wc -l      平均每条query都无法分到一条
53369
"""
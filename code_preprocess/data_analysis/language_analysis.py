"""
    2022.05.02  语言F1Score分析
    将fold_5_vaild划分为三种语言, 分别计算F1-Score
"""
import numpy as np
from collections import OrderedDict
from scipy.special import factorial
import pandas as pd
import string

# valid_df = pd.read_csv("../data_process/flod_5_valid.csv", na_values="", keep_default_na=True)
# vaild_df_us = valid_df[valid_df['query_locale'] == 'us']
# vaild_df_es = valid_df[valid_df['query_locale'] == 'es']
# vaild_df_jp = valid_df[valid_df['query_locale'] == 'jp']
# print(len(vaild_df_us), len(vaild_df_es), len(vaild_df_jp))

# vaild_df_us.to_csv('../data_process/flod_5_valid-us.csv', encoding='utf8', index=False)
# vaild_df_es.to_csv('../data_process/flod_5_valid-es.csv', encoding='utf8', index=False)
# vaild_df_jp.to_csv('../data_process/flod_5_valid-jp.csv', encoding='utf8', index=False)

"""
/home/cuixuange/kddcup_2022/v0.2_train/output/0.7369    分成不同语言计算F1-Score
"""
# micro = 汇总所有语言的计算F1-Score=0.7368
# vaild_df_us=0.7576531307289011
# vaild_df_es=0.7206744134882698
# vaild_df_jp=0.690685142417244

############# 加入optm-es-us数据集合
# vaild_df_us=0.7568007286286446
# vaild_df_es=0.7190743814876297
# vaild_df_jp=0.6921478060046189

############# 加入google-trans数据集合
# /home/cuixuange/kddcup_2022/v0.2_train/output/0.7394/
# vaild_df_us=0.7549285578057068
# vaild_df_es=0.7236344726894538
# vaild_df_jp=0.6901462663587374

############# 加入google-trans + optm-es-us数据集合
# /home/cuixuange/kddcup_2022/v0.2_train/output/0.7361/
# vaild_df_us=0.7515851176042441
# vaild_df_es=0.7196143922878457
# vaild_df_jp=0.6876674364896074

############# 加入google-trans(jp-us + es-us) + optm-es-us数据集合




valid_df = pd.read_csv("../data_process/flod_5_train_with_google_trans_QT.csv", na_values="", keep_default_na=True)
# vaild_df_us = valid_df[valid_df['query_locale'] == 'us']
# vaild_df_es = valid_df[valid_df['query_locale'] == 'es']
# vaild_df_jp = valid_df[valid_df['query_locale'] == 'jp']
# print(len(vaild_df_us), len(vaild_df_es), len(vaild_df_jp))
list_locale = valid_df['query_locale'].tolist()
print(len(list_locale), len(set(list_locale)))
"""
    2022.07.14 word2vec模型
"""
import multiprocessing
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from multiprocesspandas import applyparallel
import string
import os
import sys
import re
from time import time
from collections import defaultdict
from gensim.models import KeyedVectors
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    os.environ['PYTHONHASHSEED'] = '12345'
    os.execv(sys.executable, [sys.executable] + sys.argv)

"""
    代码参考:
    https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook
    https://radimrehurek.com/gensim/models/word2vec.html
"""
# # 1.处理数据--转化为List
# t = time()
# pd = pd.read_csv("./input/sku_text_char3_gram-head1w.csv")
# sentences = pd['char3_grams'].to_list()
# sentences = [str(item).split() for item in sentences]
# print("init=", len(sentences), len(sentences[0]))
# # 1.处理数据--获取Top-frequence-vocab
# word_freq = defaultdict(int)
# for sent in sentences:
#     for i in sent:
#         word_freq[i] += 1
# print('vocab_len=', len(word_freq))
# min_count = 15
# all_count = 0
# with open('./output_10epoch/char3ngram_top_freq_vocab-head1w.txt', mode='w') as fout:
#     for k,v in word_freq.items():
#         if v >= min_count:
#             all_count += 1
#             fout.write(str(k) + '\t' + str(v) + '\n')
# print('vocab size=', all_count)     
# print('Time to get vocab: {} mins'.format(round((time() - t) / 60, 2)))


# 1.处理数据, dataloader格式
min_count = 15
t = time()
class getSentences:
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        df = pd.read_csv(self.filename)
        for index, row in df.iterrows():
            yield str(row['char3_grams']).split()
# sentences = getSentences('./input/sku_text_char3_gram-head1w.csv')      # 2022.07.15 for test
sentences = getSentences('./input/sku_text_char3_gram.csv')      # 2022.07.15 for test
print('Time to get vocab: {} mins'.format(round((time() - t) / 60, 2)))

# 2.训练模型
cores = multiprocessing.cpu_count() - 4
w2v_model = Word2Vec(min_count=min_count,
                     window=15,
                     vector_size=64,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores,
                     sg=1)

t = time()
w2v_model.build_vocab(sentences, progress_per=100000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=3, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

#  precomputing L2-norms of word weight vectors
w2v_model.init_sims(replace=True)
w2v_model.save('./output_10epoch/gensim_word2vec.model')
word_vectors = w2v_model.wv
word_vectors.save("./output_10epoch/word2vec.wordvectors")

# # 2.测试数据
wv = KeyedVectors.load("./output_10epoch/word2vec.wordvectors", mmap='r')
vector = wv['com']  # Get numpy vector of a word
print(vector)
print(type(vector), len(vector))

# 3.将embedding写入文件
vector_np = np.zeros((1,64))    # 第一个位置是0.0
for key, index in wv.key_to_index.items():
    vector_np = np.vstack((vector_np, wv[key]))
np.savetxt('./output_10epoch/vocab_append1_embedding.txt', vector_np, fmt='%.8e')
print("vector_emb=", type(vector_np), vector_np.shape)
# 3.将词表写入文件
with open('./output_10epoch/vocab_append1_key.txt', mode='w') as fout:
    fout.write('[nan]' + '\n')
    for key, index in wv.key_to_index.items():
        fout.write(key + '\n')
print("vocab_file=", len(wv.key_to_index), type(wv.key_to_index))
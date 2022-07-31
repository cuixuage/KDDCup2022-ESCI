"""
    统一将英语作为中间语言进行转化
    1. opus-mt-{src}-{trg}模型。 先统一转化为英语, 再从英语转化为es、jp
        i. 全部转化为英语, 看一下效果    (方法不够优雅, 因为 train、vaild、test需要全部做翻译)
        ii. 语言互相翻译后, 看一下效果   (仅翻译训练集合)
"""
import torch
from transformers import MarianTokenizer, MarianMTModel
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import pandas as pd
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TRANSFORMERS_CACHE'] = '/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/'
# os.environ['HF_DATASETS_CACHE'] = '/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/datasets/'
# os.environ['HF_MODULES_CACHE'] = '/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/models/'
# os.environ['HF_METRICS_CACHE'] = '/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/metrics/'
# export TRANSFORMERS_CACHE="/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/"
# export HF_DATASETS_CACHE="/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/datasets/"
# export HF_MODULES_CACHE="/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/models/"
# export HF_METRICS_CACHE="/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/metrics/"
logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

################################# 1.模型初始化
src = "es"  # source language
trg = "en"  # target language
model_name = f"opus-mt-{src}-{trg}"
model_path = "/home/kddcup_2022/huggingface_models/translation/" + model_name
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("torch_device:", torch_device)
model = MarianMTModel.from_pretrained(model_path).to(torch_device)
tokenizer = MarianTokenizer.from_pretrained(model_path)

################################# 2.准备数据
train_df = pd.read_csv("./flod_5_train-head1k.csv", na_values="", keep_default_na=True)
train_df = pd.read_csv("./flod_5_train.csv", na_values="", keep_default_na=True)
es_train_df = train_df[train_df["query_locale"] == "es"].copy()
# print(es_train_df)
query = es_train_df['query'].fillna('').astype(str).values.tolist()
title_str = es_train_df['product_title'].fillna('').astype(str).values.tolist()
assert(len(query) == len(title_str))
print("len=", len(query), len(title_str))


############################## 3.翻译Query
data = query

batch_tensor_list = []
translations_query = []
gen_list = []

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
line_count = 0
for src_text_list in chunks(data, 1): # copy paste chunks fn from run_eval.py, consider wrapping tqdm_notebook
    """
    Code Ref: https://github.com/huggingface/transformers/issues/5602
    """
    batch = tokenizer(src_text_list, return_tensors="pt", padding=True).to(torch_device)
    batch_tensor_list.append(batch)
    line_count += 1
    if line_count % 10000 == 0:
        print(line_count)

line_count = 0
for batch in batch_tensor_list:
    model.eval()
    gen = model.generate(**batch)
    gen_list.append(gen)
    line_count += 1
    if line_count % 1000 == 0:
        print("Q:", line_count)

line_count = 0
for gen in gen_list:
    result = tokenizer.batch_decode(gen, skip_special_tokens=True)
    translations_query.extend(result)
    line_count += 1
    if line_count % 10000 == 0:
        print(line_count)
print(data[-10:], translations_query[-10:])
print(data[:10], translations_query[:10])

############################## 3.翻译Title
data = title_str

batch_tensor_list = []
translations_title = []
gen_list = []

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
line_count = 0
for src_text_list in chunks(data, 1): # copy paste chunks fn from run_eval.py, consider wrapping tqdm_notebook
    """
    Code Ref: https://github.com/huggingface/transformers/issues/5602
    """
    batch = tokenizer(src_text_list, return_tensors="pt", padding=True).to(torch_device)
    batch_tensor_list.append(batch)
    line_count += 1
    if line_count % 10000 == 0:
        print(line_count)

line_count = 0
for batch in batch_tensor_list:
    model.eval()
    gen = model.generate(**batch)
    gen_list.append(gen)
    line_count += 1
    if line_count % 1000 == 0:
        print("T:", line_count)

line_count = 0
for gen in gen_list:
    result = tokenizer.batch_decode(gen, skip_special_tokens=True)
    translations_title.extend(result)
    line_count += 1
    if line_count % 10000 == 0:
        print(line_count)
print(data[-10:], translations_title[-10:])
print(data[:10], translations_title[:10])

assert(len(translations_query) == len(translations_title))

############################## 4.写入文件
print(es_train_df["product_title"])
es_train_df["trans_query"] = translations_query
es_train_df["trans_title"] = translations_title
es_train_df["trans_locale"] = "es-us"
es_train_df["query"] = es_train_df["trans_query"]
es_train_df["product_title"] = es_train_df["trans_title"]
es_train_df["query_locale"] = es_train_df["trans_locale"]
print(es_train_df["product_title"])
es_train_df = es_train_df.drop(['trans_query', 'trans_title', 'trans_locale'], axis = 1)
es_train_df.to_csv("flod_5_train_es-us.csv", encoding='utf8', index=False)



################## Code From HuggingFace
# sample_text = "ponchos de polar mujer"
# batch = tokenizer([sample_text], return_tensors="pt").to(torch_device)
# generated_ids = model.generate(**batch)
# result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(result)



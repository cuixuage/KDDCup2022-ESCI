import pandas as pd
from zipfile import ZipFile
import random
import tempfile
import logging, string, math, os, re, random
import torch
import numpy as np
import datasets
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score
import RobertaWithSampleWeight_EmbBag
import RobertaWithSampleWeight
import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
    set_seed,
)
import psutil
import torch.nn.functional as F
from shared.base_predictor import BasePredictor, PathType
logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()


class Task1Predictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.label_to_id = {'complement': 0, 'exact': 1, 'irrelevant': 2, 'substitute': 3}
        self.brand_vocab = None
        self.color_vocab = None
        self.per_device_eval_batch_size = 32
        self.num_labels = 4
        # self.model_name_or_path_1 = './models/task01_bs256_0.7538'                     # public=0.8941
        # self.model_name_or_path_2 = './models/task23_bs256_0.7538'                     # path1+path2=0.894
        ##### self.model_name_or_path_2 = './models/task01_bs256_after_cf_0.7548'        # path1+path2=0.8933
        ##### self.model_name_or_path_3 = './models/regression_for_task1_loss_0.1172'    # public=0.8996
        ##### self.model_name_or_path_3 = './models/regression_for_task1_ndcg_0.9519'                   # public=0.8999
        ##### self.model_name_or_path_3 = './models/regression_for_task1_ndcg_flod20_after_cf_0.96684'  # public=0.8997, path1+path2+current=0.9029  
        ##### self.model_name_or_path_3 = './models/regression_for_task1_ndcg_flod20_0.9522' # public=0.9001, path1+path2+current=0.902
        ##### self.model_name_or_path_3 = './models/regression_for_task1_ndcg_flod20_after_cf_0.9668'   # public=0.8983, path1+path2+current=0.9022
        ##### sself.model_name_or_path_c1 = './models/task01_bs256_0.7540'
        ##### sself.model_name_or_path_c2 = './models/task23_bs256_0.7538'
        ##### sself.model_name_or_path_c3 = './models/task01_bs256_flod_20_0.7740'        # c1+c2+c3, public=0.8949, +after_cf_0.96684, public=0.9029

        # # first enhance
        # self.model_name_or_path_c1 = './models/task01_bs256_0.7540'
        # self.model_name_or_path_c1 = './models/task23_bs256_0.7538'
        # self.model_name_or_path_c2 = './models/task01_bs256_flod20_ptbdbc_0.7741'
        # self.model_name_or_path_c3 = './models/task01_bs256_flod20_ptdbbc_0.7729'         # c1+c2+c3, public=0.8946, +after_cf_0.96684, public=0.903
        # self.model_name_or_path_c4 = './models/task01_bs256_flod20_embbag_frompretrained_wo_bc_emb_0.7743'  # 3enhance_embag, task23+c3+c4+r1+r2+r3, 3public=0.9029

        # self.model_name_or_path_r1 = './models/regression_for_task1_ndcg_flod20_0.9522'
        # self.model_name_or_path_r2 = './models/regression_for_task1_ndcg_flod20_after_cf_0.96684'       # 3=c1+c2+c3+r1+r2, public=0.9032
        # self.model_name_or_path_r3 = './models/regression_for_task1_fload20_after_cf_frompretrianed_ndcg_0.96689'  # enhance_embag, c1+c2+c3+r1+r2+r3, 1public=0.9035  ;enhance_embag, c1+c2+c3+c4+r1+r2+r3, 2public=0.903

        # # 修复分类融合bug, public=0.9038
        # self.model_name_or_path_c1 = './models/task01_bs256_0.7540'
        # self.model_name_or_path_c2 = './models/task01_bs256_flod20_ptbdbc_0.7741'
        # self.model_name_or_path_c3 = './models/task01_bs256_flod20_ptdbbc_0.7729'  
        # self.model_name_or_path_r1 = './models/regression_for_task1_ndcg_flod20_0.9522'
        # self.model_name_or_path_r2 = './models/regression_for_task1_ndcg_flod20_after_cf_0.96684'
        # self.model_name_or_path_r3 = './models/regression_for_task1_fload20_after_cf_frompretrianed_ndcg_0.96689' # 0.9035 + 修复分类融合bug

        # try final enhance
        self.model_name_or_path_c1 = './models/task01_bs256_0.7540'
        self.model_name_or_path_c2 = './models/task01_bs256_flod20_ptbdbc_0.7741'
        self.model_name_or_path_c3 = './models/task01_bs256_flod20_ptdbbc_0.7729' 
        # self.model_name_or_path_c4 = './models/task01_bs256_flod_20_0.7740'  
        self.model_name_or_path_r1 = './models/regression_for_task1_ndcg_flod20_0.9522'
        self.model_name_or_path_r2 = './models/regression_for_task1_ndcg_flod20_after_cf_0.96684'
        self.model_name_or_path_r3 = './models/regression_for_task1_fload20_after_cf_frompretrianed_ndcg_0.96689'
        # self.model_name_or_path_r4 = './models/regression_for_task1_fload20_after_cf_frompretrianed_ndcg_ptbdbc_0.9666'
        self.model_name_or_path_r5 = './models/regression_for_task1_fload20_after_cf_frompretrianed_ndcg_ptdbbc_0.96641'    # at 0.9038+c4+r4+r5, enhance ptbdbc+ptdbb

        self.accelerator = Accelerator()

    def prediction_setup(self):
        """To be implemented by the participants.

        Participants can add the steps needed to initialize their models,
        and/or any other setup related things here.
        """
        transformers.logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path_c1, use_fast=False, truncation_side='right')

        def get_vocab_from_file(file_path):
            idx = 0
            name_idx = dict()
            with open(file_path, mode='r') as fin:
                for line in fin:
                    item = line.strip()
                    if item != '[UNK]':
                        name_idx[item] = idx
                    idx += 1
            # print( len(name_idx.keys()) )
            return name_idx
        self.brand_vocab = get_vocab_from_file('./data/brand.txt')
        self.color_vocab = get_vocab_from_file('./data/color.txt')
        self.emb_bag_vocab = get_vocab_from_file('./data/word2vec.wordvectors.vocab_char3.txt')
        print("self.classification model=", self.model_name_or_path_c1, self.model_name_or_path_c2, self.model_name_or_path_c3)
        print("self.regression model=", self.model_name_or_path_r1, self.model_name_or_path_r2, self.model_name_or_path_r3)

    def get_eval_dataloader(self, raw_datasets = None):

        def preprocess_function(examples):
            cleaner = re.compile('<.*?>')    # 删除网页标签, <>内容会被删除
            examples['product_text'] = []
            for idx, _ in enumerate(examples["query"]):
                title = examples["product_title"][idx]
                title = title if title and len(title) > 0 and not title.isspace() else ''

                brand = examples["product_brand"][idx]
                brand = brand if brand and len(brand) > 0 and not brand.isspace() else ''

                color = examples["product_color_name"][idx]
                color = color if color and len(color) > 0 and not color.isspace() else ''

                bullet_point = examples['product_bullet_point'][idx]
                bullet_point = bullet_point if bullet_point and len(bullet_point) > 0 and not bullet_point.isspace() else ''

                desc = examples['product_description'][idx]
                desc = re.sub(cleaner, '', desc) if desc and len(desc) > 0 and not desc.isspace() else ''

                item = title + ' </s></s> ' + brand + ' </s></s> ' + color + ' </s></s> ' + bullet_point + ' </s></s> ' + desc
                examples['product_text'].append(item)

            examples['query'] = [
                line if line and len(line) > 0 and not line.isspace() else '' for line in examples["query"] 
            ]

            texts = (
                (examples["query"], examples["product_text"])
            )
            result = self.tokenizer(*texts, padding="max_length", max_length=256, truncation=True)  # 2022.05.03 加入翻译后的数据,有些Query长度翻译的很长,不能用only-second阶段方式

            if "product_brand" in examples:
                result["brand_idx"] = []
                for item in examples['product_brand']:
                    if not item: 
                        result['brand_idx'].append(0)       # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.brand_vocab:
                        result['brand_idx'].append(int(self.brand_vocab[item]))
                    else:
                        result['brand_idx'].append(0) # out of vocabulary
            if "product_color_name" in examples:
                result["color_idx"] = []
                for item in examples['product_color_name']:
                    if not item: 
                        result['color_idx'].append(0) # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.color_vocab:
                        result['color_idx'].append(int(self.color_vocab[item]))
                    else:
                        result['color_idx'].append(0)   # out of vocabulary   

            if "esci_label" in examples:
                result["labels"] = [self.label_to_id[l] for l in examples["esci_label"]]

            return result

        with self.accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=raw_datasets.column_names,
                desc="Running tokenizer on dataset",
            )
        
        eval_dataset = processed_datasets
        data_collator = default_data_collator
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size)
        return eval_dataloader

    def get_eval_dataloader_ptbdbc(self, raw_datasets = None):

        def preprocess_function(examples):
            cleaner = re.compile('<.*?>')    # 删除网页标签, <>内容会被删除
            examples['product_text'] = []
            for idx, _ in enumerate(examples["query"]):
                title = examples["product_title"][idx]
                title = title if title and len(title) > 0 and not title.isspace() else ''

                brand = examples["product_brand"][idx]
                brand = brand if brand and len(brand) > 0 and not brand.isspace() else ''

                color = examples["product_color_name"][idx]
                color = color if color and len(color) > 0 and not color.isspace() else ''

                bullet_point = examples['product_bullet_point'][idx]
                bullet_point = bullet_point if bullet_point and len(bullet_point) > 0 and not bullet_point.isspace() else ''

                desc = examples['product_description'][idx]
                desc = re.sub(cleaner, '', desc) if desc and len(desc) > 0 and not desc.isspace() else ''

                # item = title + ' </s></s> ' + brand + ' </s></s> ' + color + ' </s></s> ' + bullet_point + ' </s></s> ' + desc
                item = title + ' </s></s> ' + bullet_point + ' </s></s> ' + desc + ' </s></s> ' + brand + ' </s></s> ' + color
                examples['product_text'].append(item)

            examples['query'] = [
                line if line and len(line) > 0 and not line.isspace() else '' for line in examples["query"] 
            ]

            texts = (
                (examples["query"], examples["product_text"])
            )
            result = self.tokenizer(*texts, padding="max_length", max_length=256, truncation=True)  # 2022.05.03 加入翻译后的数据,有些Query长度翻译的很长,不能用only-second阶段方式

            if "product_brand" in examples:
                result["brand_idx"] = []
                for item in examples['product_brand']:
                    if not item: 
                        result['brand_idx'].append(0)       # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.brand_vocab:
                        result['brand_idx'].append(int(self.brand_vocab[item]))
                    else:
                        result['brand_idx'].append(0) # out of vocabulary
            if "product_color_name" in examples:
                result["color_idx"] = []
                for item in examples['product_color_name']:
                    if not item: 
                        result['color_idx'].append(0) # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.color_vocab:
                        result['color_idx'].append(int(self.color_vocab[item]))
                    else:
                        result['color_idx'].append(0)   # out of vocabulary   

            if "esci_label" in examples:
                result["labels"] = [self.label_to_id[l] for l in examples["esci_label"]]

            return result

        with self.accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=raw_datasets.column_names,
                desc="Running tokenizer on dataset",
            )
        
        eval_dataset = processed_datasets
        data_collator = default_data_collator
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size)
        return eval_dataloader
   
    def get_eval_dataloader_ptdbbc(self, raw_datasets = None):

        def preprocess_function(examples):
            cleaner = re.compile('<.*?>')    # 删除网页标签, <>内容会被删除
            examples['product_text'] = []
            for idx, _ in enumerate(examples["query"]):
                title = examples["product_title"][idx]
                title = title if title and len(title) > 0 and not title.isspace() else ''

                brand = examples["product_brand"][idx]
                brand = brand if brand and len(brand) > 0 and not brand.isspace() else ''

                color = examples["product_color_name"][idx]
                color = color if color and len(color) > 0 and not color.isspace() else ''

                bullet_point = examples['product_bullet_point'][idx]
                bullet_point = bullet_point if bullet_point and len(bullet_point) > 0 and not bullet_point.isspace() else ''

                desc = examples['product_description'][idx]
                desc = re.sub(cleaner, '', desc) if desc and len(desc) > 0 and not desc.isspace() else ''

                # item = title + ' </s></s> ' + brand + ' </s></s> ' + color + ' </s></s> ' + bullet_point + ' </s></s> ' + desc
                item = title + ' </s></s> ' + desc + ' </s></s> ' + bullet_point + ' </s></s> ' + brand + ' </s></s> ' + color
                examples['product_text'].append(item)

            examples['query'] = [
                line if line and len(line) > 0 and not line.isspace() else '' for line in examples["query"] 
            ]

            texts = (
                (examples["query"], examples["product_text"])
            )
            result = self.tokenizer(*texts, padding="max_length", max_length=256, truncation=True)  # 2022.05.03 加入翻译后的数据,有些Query长度翻译的很长,不能用only-second阶段方式

            if "product_brand" in examples:
                result["brand_idx"] = []
                for item in examples['product_brand']:
                    if not item: 
                        result['brand_idx'].append(0)       # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.brand_vocab:
                        result['brand_idx'].append(int(self.brand_vocab[item]))
                    else:
                        result['brand_idx'].append(0) # out of vocabulary
            if "product_color_name" in examples:
                result["color_idx"] = []
                for item in examples['product_color_name']:
                    if not item: 
                        result['color_idx'].append(0) # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.color_vocab:
                        result['color_idx'].append(int(self.color_vocab[item]))
                    else:
                        result['color_idx'].append(0)   # out of vocabulary   

            if "esci_label" in examples:
                result["labels"] = [self.label_to_id[l] for l in examples["esci_label"]]

            return result

        with self.accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=raw_datasets.column_names,
                desc="Running tokenizer on dataset",
            )
        
        eval_dataset = processed_datasets
        data_collator = default_data_collator
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size)
        return eval_dataloader

    def get_eval_dataloader_embbag(self, raw_datasets=None):
        def preprocess_function(examples):
            cleaner = re.compile('<.*?>')    # 删除网页标签, <>内容会被删除
            examples['product_text'] = []
            for idx, _ in enumerate(examples["query"]):
                title = examples["product_title"][idx]
                title = title if title and len(title) > 0 and not title.isspace() else ''

                brand = examples["product_brand"][idx]
                brand = brand if brand and len(brand) > 0 and not brand.isspace() else ''

                color = examples["product_color_name"][idx]
                color = color if color and len(color) > 0 and not color.isspace() else ''

                bullet_point = examples['product_bullet_point'][idx]
                bullet_point = bullet_point if bullet_point and len(bullet_point) > 0 and not bullet_point.isspace() else ''

                desc = examples['product_description'][idx]
                desc = re.sub(cleaner, '', desc) if desc and len(desc) > 0 and not desc.isspace() else ''

                item = title + ' </s></s> ' + brand + ' </s></s> ' + color + ' </s></s> ' + bullet_point + ' </s></s> ' + desc
                examples['product_text'].append(item)

            examples['query'] = [
                line if line and len(line) > 0 and not line.isspace() else '' for line in examples["query"] 
            ]

            texts = (
                (examples["query"], examples["product_text"])
            )
            result = self.tokenizer(*texts, padding="max_length", max_length=256, truncation=True)  # 2022.05.03 加入翻译后的数据,有些Query长度翻译的很长,不能用only-second阶段方式

            if "product_brand" in examples:
                result["brand_idx"] = []
                for item in examples['product_brand']:
                    if not item: 
                        result['brand_idx'].append(0)       # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.brand_vocab:
                        result['brand_idx'].append(int(self.brand_vocab[item]))
                    else:
                        result['brand_idx'].append(0) # out of vocabulary
            if "product_color_name" in examples:
                result["color_idx"] = []
                for item in examples['product_color_name']:
                    if not item: 
                        result['color_idx'].append(0) # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.color_vocab:
                        result['color_idx'].append(int(self.color_vocab[item]))
                    else:
                        result['color_idx'].append(0)   # out of vocabulary   

            if "esci_label" in examples:
                result["labels"] = [self.label_to_id[l] for l in examples["esci_label"]]

            # 2.chars-tokenizer
            def change_2_chars_idx(item_str):
                item_str = str(item_str).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                ids = []
                n = 3
                if len(item_str) <= n:
                    if item_str in self.emb_bag_vocab:
                        ids.append(self.emb_bag_vocab.get(item_str, 0))
                else:
                    for i in range(len(item_str) - n):
                        i_v = item_str[i:i+n].strip()
                        if i_v in self.emb_bag_vocab:
                            ids.append(self.emb_bag_vocab.get(i_v, 0))  
                ids = ids if len(ids) != 0 else [0]
                return ids

            def collate_fn_padd(batch=[]):
                """ batch of variable lengths"""
                batch = [torch.IntTensor(t) for t in batch]
                batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=int(0))
                # mask = (batch != 0)
                # return batch.numpy(), mask.numpy()
                return batch.numpy()
            
            def get_chars_fea(examples, result):
                query_char3_idx = []
                query_char3_lens = []
                title_bc_char3_idx = []
                title_bc_char3_lens = []
                desc_char3_idx = []
                desc_char3_lens = []
                for idx, _ in enumerate(examples["query"]):
                    # 2.1 获取Query侧 char-3 特征
                    query_str = examples["query"][idx]
                    ids = change_2_chars_idx(query_str)
                    query_char3_idx.append(ids)
                    query_char3_lens.append(len(ids))

                    # 2.2 获取Product侧 char-3 特征; 分为三组, 如果是1组的话, 会被desc主导
                    title_bc_str = str(examples["product_title"][idx]) + ' ' + str(examples["product_brand"][idx]) + ' ' + str(examples["product_color_name"][idx]) + ' ' + str(examples["product_bullet_point"][idx])
                    desc_str = examples["product_description"][idx]
                    ids = change_2_chars_idx(title_bc_str)
                    title_bc_char3_idx.append(ids)
                    title_bc_char3_lens.append(len(ids))
                    ids = change_2_chars_idx(desc_str)
                    desc_char3_idx.append(ids)
                    desc_char3_lens.append(len(ids))

                result['query_chars_input_ids'] = collate_fn_padd(query_char3_idx)
                result['query_chars_lens'] = query_char3_lens
                result['title_bc_chars_input_ids'] = collate_fn_padd(title_bc_char3_idx)
                result['title_bc_chars_lens'] = title_bc_char3_lens
                result['desc_chars_input_ids'] = collate_fn_padd(desc_char3_idx)
                result['desc_chars_lens'] = desc_char3_lens

                return result
        
            result = get_chars_fea(examples, result)

            # 3. country idx
            result['country_idx'] = []
            for locale in examples['query_locale']:
                if locale == 'us':  result['country_idx'].append(1)
                elif locale == 'es':  result['country_idx'].append(2)
                elif locale == 'jp':  result['country_idx'].append(3)
                elif locale == 'es-us':  result['country_idx'].append(1)   
                else: result['country_idx'].append(0)

            return result

        with self.accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                batch_size=self.per_device_eval_batch_size,
                remove_columns=raw_datasets.column_names,
                desc="Running tokenizer on dataset",
            )
        eval_dataset = processed_datasets
        data_collator = default_data_collator
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size)
        return eval_dataloader

    def get_eval_dataloader_embbag_ptbdbc(self, raw_datasets=None):
        def preprocess_function(examples):
            cleaner = re.compile('<.*?>')    # 删除网页标签, <>内容会被删除
            examples['product_text'] = []
            for idx, _ in enumerate(examples["query"]):
                title = examples["product_title"][idx]
                title = title if title and len(title) > 0 and not title.isspace() else ''

                brand = examples["product_brand"][idx]
                brand = brand if brand and len(brand) > 0 and not brand.isspace() else ''

                color = examples["product_color_name"][idx]
                color = color if color and len(color) > 0 and not color.isspace() else ''

                bullet_point = examples['product_bullet_point'][idx]
                bullet_point = bullet_point if bullet_point and len(bullet_point) > 0 and not bullet_point.isspace() else ''

                desc = examples['product_description'][idx]
                desc = re.sub(cleaner, '', desc) if desc and len(desc) > 0 and not desc.isspace() else ''

                # item = title + ' </s></s> ' + brand + ' </s></s> ' + color + ' </s></s> ' + bullet_point + ' </s></s> ' + desc
                item = title + ' </s></s> ' + bullet_point + ' </s></s> ' + desc + ' </s></s> ' + brand + ' </s></s> ' + color
                examples['product_text'].append(item)

            examples['query'] = [
                line if line and len(line) > 0 and not line.isspace() else '' for line in examples["query"] 
            ]

            texts = (
                (examples["query"], examples["product_text"])
            )
            result = self.tokenizer(*texts, padding="max_length", max_length=256, truncation=True)  # 2022.05.03 加入翻译后的数据,有些Query长度翻译的很长,不能用only-second阶段方式

            if "product_brand" in examples:
                result["brand_idx"] = []
                for item in examples['product_brand']:
                    if not item: 
                        result['brand_idx'].append(0)       # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.brand_vocab:
                        result['brand_idx'].append(int(self.brand_vocab[item]))
                    else:
                        result['brand_idx'].append(0) # out of vocabulary
            if "product_color_name" in examples:
                result["color_idx"] = []
                for item in examples['product_color_name']:
                    if not item: 
                        result['color_idx'].append(0) # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.color_vocab:
                        result['color_idx'].append(int(self.color_vocab[item]))
                    else:
                        result['color_idx'].append(0)   # out of vocabulary   

            if "esci_label" in examples:
                result["labels"] = [self.label_to_id[l] for l in examples["esci_label"]]

            # 2.chars-tokenizer
            def change_2_chars_idx(item_str):
                item_str = str(item_str).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                ids = []
                n = 3
                if len(item_str) <= n:
                    if item_str in self.emb_bag_vocab:
                        ids.append(self.emb_bag_vocab.get(item_str, 0))
                else:
                    for i in range(len(item_str) - n):
                        i_v = item_str[i:i+n].strip()
                        if i_v in self.emb_bag_vocab:
                            ids.append(self.emb_bag_vocab.get(i_v, 0))  
                ids = ids if len(ids) != 0 else [0]
                return ids

            def collate_fn_padd(batch=[]):
                """ batch of variable lengths"""
                batch = [torch.IntTensor(t) for t in batch]
                batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=int(0))
                # mask = (batch != 0)
                # return batch.numpy(), mask.numpy()
                return batch.numpy()
            
            def get_chars_fea(examples, result):
                query_char3_idx = []
                query_char3_lens = []
                title_bc_char3_idx = []
                title_bc_char3_lens = []
                desc_char3_idx = []
                desc_char3_lens = []
                for idx, _ in enumerate(examples["query"]):
                    # 2.1 获取Query侧 char-3 特征
                    query_str = examples["query"][idx]
                    ids = change_2_chars_idx(query_str)
                    query_char3_idx.append(ids)
                    query_char3_lens.append(len(ids))

                    # 2.2 获取Product侧 char-3 特征; 分为三组, 如果是1组的话, 会被desc主导
                    title_bc_str = str(examples["product_title"][idx]) + ' ' + str(examples["product_brand"][idx]) + ' ' + str(examples["product_color_name"][idx]) + ' ' + str(examples["product_bullet_point"][idx])
                    desc_str = examples["product_description"][idx]
                    ids = change_2_chars_idx(title_bc_str)
                    title_bc_char3_idx.append(ids)
                    title_bc_char3_lens.append(len(ids))
                    ids = change_2_chars_idx(desc_str)
                    desc_char3_idx.append(ids)
                    desc_char3_lens.append(len(ids))

                result['query_chars_input_ids'] = collate_fn_padd(query_char3_idx)
                result['query_chars_lens'] = query_char3_lens
                result['title_bc_chars_input_ids'] = collate_fn_padd(title_bc_char3_idx)
                result['title_bc_chars_lens'] = title_bc_char3_lens
                result['desc_chars_input_ids'] = collate_fn_padd(desc_char3_idx)
                result['desc_chars_lens'] = desc_char3_lens

                return result
        
            result = get_chars_fea(examples, result)

            # 3. country idx
            result['country_idx'] = []
            for locale in examples['query_locale']:
                if locale == 'us':  result['country_idx'].append(1)
                elif locale == 'es':  result['country_idx'].append(2)
                elif locale == 'jp':  result['country_idx'].append(3)
                elif locale == 'es-us':  result['country_idx'].append(1)   
                else: result['country_idx'].append(0)

            return result

        with self.accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                batch_size=self.per_device_eval_batch_size,
                remove_columns=raw_datasets.column_names,
                desc="Running tokenizer on dataset",
            )
        eval_dataset = processed_datasets
        data_collator = default_data_collator
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size)
        return eval_dataloader

    def get_eval_dataloader_embbag_ptdbbc(self, raw_datasets=None):
        def preprocess_function(examples):
            cleaner = re.compile('<.*?>')    # 删除网页标签, <>内容会被删除
            examples['product_text'] = []
            for idx, _ in enumerate(examples["query"]):
                title = examples["product_title"][idx]
                title = title if title and len(title) > 0 and not title.isspace() else ''

                brand = examples["product_brand"][idx]
                brand = brand if brand and len(brand) > 0 and not brand.isspace() else ''

                color = examples["product_color_name"][idx]
                color = color if color and len(color) > 0 and not color.isspace() else ''

                bullet_point = examples['product_bullet_point'][idx]
                bullet_point = bullet_point if bullet_point and len(bullet_point) > 0 and not bullet_point.isspace() else ''

                desc = examples['product_description'][idx]
                desc = re.sub(cleaner, '', desc) if desc and len(desc) > 0 and not desc.isspace() else ''

                # item = title + ' </s></s> ' + brand + ' </s></s> ' + color + ' </s></s> ' + bullet_point + ' </s></s> ' + desc
                item = title + ' </s></s> ' + desc + ' </s></s> ' + bullet_point + ' </s></s> ' + brand + ' </s></s> ' + color
                examples['product_text'].append(item)

            examples['query'] = [
                line if line and len(line) > 0 and not line.isspace() else '' for line in examples["query"] 
            ]

            texts = (
                (examples["query"], examples["product_text"])
            )
            result = self.tokenizer(*texts, padding="max_length", max_length=256, truncation=True)  # 2022.05.03 加入翻译后的数据,有些Query长度翻译的很长,不能用only-second阶段方式

            if "product_brand" in examples:
                result["brand_idx"] = []
                for item in examples['product_brand']:
                    if not item: 
                        result['brand_idx'].append(0)       # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.brand_vocab:
                        result['brand_idx'].append(int(self.brand_vocab[item]))
                    else:
                        result['brand_idx'].append(0) # out of vocabulary
            if "product_color_name" in examples:
                result["color_idx"] = []
                for item in examples['product_color_name']:
                    if not item: 
                        result['color_idx'].append(0) # out of vocabulary
                        continue
                    item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                    if item in self.color_vocab:
                        result['color_idx'].append(int(self.color_vocab[item]))
                    else:
                        result['color_idx'].append(0)   # out of vocabulary   

            if "esci_label" in examples:
                result["labels"] = [self.label_to_id[l] for l in examples["esci_label"]]

            # 2.chars-tokenizer
            def change_2_chars_idx(item_str):
                item_str = str(item_str).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                ids = []
                n = 3
                if len(item_str) <= n:
                    if item_str in self.emb_bag_vocab:
                        ids.append(self.emb_bag_vocab.get(item_str, 0))
                else:
                    for i in range(len(item_str) - n):
                        i_v = item_str[i:i+n].strip()
                        if i_v in self.emb_bag_vocab:
                            ids.append(self.emb_bag_vocab.get(i_v, 0))  
                ids = ids if len(ids) != 0 else [0]
                return ids

            def collate_fn_padd(batch=[]):
                """ batch of variable lengths"""
                batch = [torch.IntTensor(t) for t in batch]
                batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=int(0))
                # mask = (batch != 0)
                # return batch.numpy(), mask.numpy()
                return batch.numpy()
            
            def get_chars_fea(examples, result):
                query_char3_idx = []
                query_char3_lens = []
                title_bc_char3_idx = []
                title_bc_char3_lens = []
                desc_char3_idx = []
                desc_char3_lens = []
                for idx, _ in enumerate(examples["query"]):
                    # 2.1 获取Query侧 char-3 特征
                    query_str = examples["query"][idx]
                    ids = change_2_chars_idx(query_str)
                    query_char3_idx.append(ids)
                    query_char3_lens.append(len(ids))

                    # 2.2 获取Product侧 char-3 特征; 分为三组, 如果是1组的话, 会被desc主导
                    title_bc_str = str(examples["product_title"][idx]) + ' ' + str(examples["product_brand"][idx]) + ' ' + str(examples["product_color_name"][idx]) + ' ' + str(examples["product_bullet_point"][idx])
                    desc_str = examples["product_description"][idx]
                    ids = change_2_chars_idx(title_bc_str)
                    title_bc_char3_idx.append(ids)
                    title_bc_char3_lens.append(len(ids))
                    ids = change_2_chars_idx(desc_str)
                    desc_char3_idx.append(ids)
                    desc_char3_lens.append(len(ids))

                result['query_chars_input_ids'] = collate_fn_padd(query_char3_idx)
                result['query_chars_lens'] = query_char3_lens
                result['title_bc_chars_input_ids'] = collate_fn_padd(title_bc_char3_idx)
                result['title_bc_chars_lens'] = title_bc_char3_lens
                result['desc_chars_input_ids'] = collate_fn_padd(desc_char3_idx)
                result['desc_chars_lens'] = desc_char3_lens

                return result
        
            result = get_chars_fea(examples, result)

            # 3. country idx
            result['country_idx'] = []
            for locale in examples['query_locale']:
                if locale == 'us':  result['country_idx'].append(1)
                elif locale == 'es':  result['country_idx'].append(2)
                elif locale == 'jp':  result['country_idx'].append(3)
                elif locale == 'es-us':  result['country_idx'].append(1)   
                else: result['country_idx'].append(0)

            return result

        with self.accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                batch_size=self.per_device_eval_batch_size,
                remove_columns=raw_datasets.column_names,
                desc="Running tokenizer on dataset",
            )
        eval_dataset = processed_datasets
        data_collator = default_data_collator
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size)
        return eval_dataloader

    def predict_classification_embbag(self, model_name_or_path=None, eval_dataloader=None):
        # 1. model_1 Predict
        config_1 = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels)
        model_1 = RobertaWithSampleWeight_EmbBag.RobertaWithSampleWeight.from_pretrained(model_name_or_path, from_tf=False, config=config_1)
        model_1.config.label2id = self.label_to_id
        model_1.config.id2label = {id: label for label, id in config_1.label2id.items()}
        # model_1 = self.accelerator.prepare(model_1)
        model_1 = model_1.cuda()
        model_1.eval()
        task1_predictions_1 = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model_1(**batch)
                task1_pred = torch.softmax(outputs.logits, dim=-1)
                task1_predictions_1.append(self.accelerator.gather(task1_pred).cpu().numpy())
        task1_predictions_1 = np.concatenate(task1_predictions_1)
        del model_1
        torch.cuda.empty_cache()
        print("Done: predict_classification_embbag=", model_name_or_path)
        return task1_predictions_1

    def predict_regression_embbag(self, model_name_or_path=None, eval_dataloader=None):
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels)
        config.num_labels = 1
        model = RobertaWithSampleWeight_EmbBag.RobertaWithSampleWeight.from_pretrained(model_name_or_path, from_tf=False, config=config)
        # model = self.accelerator.prepare(model)
        model = model.cuda()
        model.eval()
        task1_predictions_3 = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
                task1_pred = F.sigmoid(outputs.logits)
                task1_predictions_3.append(self.accelerator.gather(task1_pred).cpu().numpy())
        task1_predictions_3 = np.array(np.concatenate(task1_predictions_3))
        del model
        torch.cuda.empty_cache()
        print("Done: predict_regression_embbag=", model_name_or_path)
        return task1_predictions_3

    def predict_regression(self, model_name_or_path=None, eval_dataloader=None):
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels)
        config.num_labels = 1
        model = RobertaWithSampleWeight.RobertaWithSampleWeight.from_pretrained(model_name_or_path, from_tf=False, config=config)
        # model = self.accelerator.prepare(model)
        model = model.cuda()
        model.eval()
        task1_predictions_3 = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
                task1_pred = F.sigmoid(outputs.logits)
                task1_predictions_3.append(self.accelerator.gather(task1_pred).cpu().numpy())
        task1_predictions_3 = np.array(np.concatenate(task1_predictions_3))
        del model
        torch.cuda.empty_cache()
        print("Done: predict_regression=", model_name_or_path)
        return task1_predictions_3

    def predict_classification(self, model_name_or_path=None, eval_dataloader=None):
        # 1. model_1 Predict
        config_1 = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels)
        model_1 = RobertaWithSampleWeight.RobertaWithSampleWeight.from_pretrained(model_name_or_path, from_tf=False, config=config_1)
        model_1.config.label2id = self.label_to_id
        model_1.config.id2label = {id: label for label, id in config_1.label2id.items()}
        # model_1 = self.accelerator.prepare(model_1)
        model_1 = model_1.cuda()
        model_1.eval()
        task1_predictions_1 = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model_1(**batch)
                task1_pred = torch.softmax(outputs.logits, dim=-1)
                task1_predictions_1.append(self.accelerator.gather(task1_pred).cpu().numpy())
        task1_predictions_1 = np.concatenate(task1_predictions_1)
        del model_1
        torch.cuda.empty_cache()
        print("Done: predict_classification=", model_name_or_path)
        return task1_predictions_1

    def predict(self,
                test_set_path: PathType,
                product_catalogue_path: PathType,
                predictions_output_path: PathType,
                register_progress = lambda x: print("Progress : ", x)):
    
        transformers.logging.set_verbosity_error()
        # 0.数据集Join, 得到原始dataframe
        test_query_df = pd.read_csv(test_set_path)
        product_df = pd.read_csv(product_catalogue_path)
        test_df = pd.merge(test_query_df, product_df, how='left', left_on=['query_locale','product_id'], right_on=['product_locale', 'product_id'])

        # 1.准备dataloader
        raw_datasets = Dataset.from_pandas(test_df)
        eval_dataloader = self.get_eval_dataloader(raw_datasets)
        eval_dataloader_ptbdbc = self.get_eval_dataloader_ptbdbc(raw_datasets)
        eval_dataloader_ptdbbc = self.get_eval_dataloader_ptdbbc(raw_datasets)
        eval_dataloader_bag = self.get_eval_dataloader_embbag(raw_datasets)
        # eval_dataloader_bag_ptbdbc = self.get_eval_dataloader_embbag_ptbdbc(raw_datasets)
        eval_dataloader_bag_ptdbbc = self.get_eval_dataloader_embbag_ptdbbc(raw_datasets)
        eval_dataloader,eval_dataloader_ptbdbc, eval_dataloader_ptdbbc, eval_dataloader_bag, eval_dataloader_bag_ptdbbc = self.accelerator.prepare(eval_dataloader,eval_dataloader_ptbdbc,
                            eval_dataloader_ptdbbc, eval_dataloader_bag, eval_dataloader_bag_ptdbbc)

        # 2.分类模型, 内部聚合计算平均分
        task1_predictions_c1 = self.predict_classification(self.model_name_or_path_c1, eval_dataloader)
        task1_predictions_c2 = self.predict_classification(self.model_name_or_path_c2, eval_dataloader_ptbdbc)
        task1_predictions_c3 = self.predict_classification(self.model_name_or_path_c3, eval_dataloader_ptdbbc)
        # task1_predictions_c4 = self.predict_classification(self.model_name_or_path_c4, eval_dataloader)
        task1_predictions = []
        predictions = []
        for index, item in enumerate(task1_predictions_c1):
            # https://stackoverflow.com/questions/18461623/average-values-in-two-numpy-arrays
            merge_item = np.mean( np.array([ item, task1_predictions_c2[index], task1_predictions_c3[index]]), axis=0 )
            label_index = np.argmax(merge_item)
            task1_predictions.append(merge_item)
            predictions.append(label_index)

        with open(predictions_output_path, mode='w') as writer:
            writer.write("index\tprediction\tprobs\n")
            for index, item in enumerate(predictions):
                probs = task1_predictions[index]
                value_str = ','.join([str(i) for i in probs.tolist()])
                writer.write(f"{index}\t{item}\t{value_str}\n")

        # 3.回归模型, 内部聚合计算平均分
        task1_predictions_r1 = self.predict_regression(self.model_name_or_path_r1, eval_dataloader)
        task1_predictions_r2 = self.predict_regression(self.model_name_or_path_r2, eval_dataloader)
        task1_predictions_r3 = self.predict_regression_embbag(self.model_name_or_path_r3, eval_dataloader_bag)
        # task1_predictions_r4 = self.predict_regression_embbag(self.model_name_or_path_r4, eval_dataloader_bag_ptbdbc)
        task1_predictions_r5 = self.predict_regression_embbag(self.model_name_or_path_r5, eval_dataloader_bag_ptdbbc)
        task1_predictions_3 = []
        for index, item in enumerate(task1_predictions_r1):
            # https://stackoverflow.com/questions/18461623/average-values-in-two-numpy-arrays
            merge_item = np.mean( np.array([ item, task1_predictions_r2[index], task1_predictions_r3[index], task1_predictions_r5[index]]), axis=0 )
            task1_predictions_3.append(float(merge_item[0]))
        print("Done: task1_predictions_3")

        ####################################################################################################
        ####################################################################################################
        ## 
        ## STEP 4 : Save Predictions to `predictions_output_path` as a valid CSV file
        ## 
        ####################################################################################################
        ####################################################################################################

        list_queryid = test_query_df['query_id'].tolist()
        list_productid = test_query_df['product_id'].tolist()

        list_label_key = []
        list_label_prob = []
        list_label_prob_merge = []    # 2022.07.20 发现以前的代码bug, 重新计算0.01*prob0 + 1.0*prob1 + 0.0*prob2 + 0.1*prob3 
        label_dict = {'0':0.01, '1':1.0, '2':0.0, '3':0.1}
        with open(predictions_output_path, mode='r') as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                index = tokens[0]
                probs = tokens[-1].split(',')
                if index == 'index': continue

                probs = [float(i) for i in probs]
                label_index = np.argmax(np.array(probs))
                label_prob = probs[label_index]
                label_index_key = label_dict[str(label_index)]

                list_label_key.append(label_index_key)
                list_label_prob.append(label_prob)
                list_label_prob_merge.append(0.01 * probs[0] + 1.0 * probs[1] + 0.0 * probs[2] + 0.1 * probs[3])

        print("line num debug=", len(list_queryid), len(list_productid), len(list_label_key), len(list_label_prob))
        assert(len(list_queryid) == len(list_productid) == len(list_label_key) == len(list_label_prob))

        result = dict()         # key=query_id  value=list(dict())
        for idx, _  in enumerate(list_queryid):
            query_id = list_queryid[idx]
            item_dict = dict()
            item_dict['product_id'] = list_productid[idx]
            item_dict['sort_key_1'] = list_label_key[idx]                       # 方式1, 先按照label, 再按照prob概率排序
            item_dict['sort_key_2'] = list_label_prob[idx]                      # 方式1, 先按照label, 再按照prob概率排序
            item_dict['sort_key_3'] = list_label_key[idx] * list_label_prob[idx]  # 方式2, 按照 label*prob的分数排序
            item_dict['sort_key_4'] = list_label_key[idx] * list_label_prob[idx] + float(task1_predictions_3[idx])  # 方式3, 按照 label*prob的分数排序 + 融合回归模型的分数
            item_dict['sort_key_5'] = float(list_label_prob_merge[idx]) + float(task1_predictions_3[idx])  # 方式3, 按照 label*prob的分数排序 + 融合回归模型的分数
            if query_id not in result:
                result[query_id] = list()
            result[query_id].append(item_dict)
                
        with open(predictions_output_path, mode='w') as fout:
            fout.write('product_id' + ',' + 'query_id' + '\n')
            for query_id, v_list_dict in result.items():
                # sort_v_list_dict = sorted(v_list_dict, key=lambda x : (x['sort_key_1'], x['sort_key_2']), reverse=True) # 方式1, 先按照label, 再按照prob概率排序
                # sort_v_list_dict = sorted(v_list_dict, key=lambda x : (x['sort_key_3']), reverse=True)     # 方式2, 按照 label*prob的分数排序
                # sort_v_list_dict = sorted(v_list_dict, key=lambda x : (x['sort_key_4']), reverse=True)     # 方式3, 按照 label*prob的分数排序 + 融合回归模型的分数  （bug实现）
                sort_v_list_dict = sorted(v_list_dict, key=lambda x : (x['sort_key_5']), reverse=True)     # 方式3, 按照 label*prob的分数排序 + 融合回归模型的分数  
                product_ids = []
                for item_dict in sort_v_list_dict:
                    product_ids.append(str(item_dict['product_id']))
                    fout.write(str(item_dict['product_id']) + ',' + str(query_id) + '\n')

        print("Writing Task-1 Predictions to : ", predictions_output_path)


if __name__ == "__main__":

    transformers.logging.set_verbosity_error()
    set_seed(12345)

    # Instantiate Predictor Class
    predictor = Task1Predictor()
    predictor.prediction_setup()
    test_set_path = "./data/task_1_query-product_ranking/test_public-v0.3.csv.zip"
    # test_set_path = "./data/task_1_query-product_ranking/test_public-v0.3-head100.csv.zip"
    product_catalogue_path = "./data/task_1_query-product_ranking/task1_product_catalogue-v0.3.csv.zip"
    
    # Generate a Random File to store predictions
    with tempfile.NamedTemporaryFile(suffix='.csv') as output_file:
        output_file_path = output_file.name
        # output_file_path = 'submission_task1_tmp.txt'
        print(output_file_path)

        # Make Predictions
        predictor.predict(
            test_set_path=test_set_path,
            product_catalogue_path=product_catalogue_path,
            predictions_output_path=output_file_path
        )
        
        ## TODO - Add Validations for Task-1 Sample Submissions

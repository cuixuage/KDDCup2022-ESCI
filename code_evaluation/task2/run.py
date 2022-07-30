import pandas as pd
import random
import tempfile
import logging, string, math, os, re, random
import torch
import numpy as np
import datasets
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score
from RobertaWithSampleWeight import RobertaWithSampleWeight
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
from shared.base_predictor import BasePredictor, PathType
logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()

class Task2Predictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.label_to_id = {'complement': 0, 'exact': 1, 'irrelevant': 2, 'substitute': 3}
        self.brand_vocab = None
        self.color_vocab = None
        self.per_device_eval_batch_size = 128
        self.accelerator = Accelerator()

    def prediction_setup(self):
        """To be implemented by the participants.

        Participants can add the steps needed to initialize their models,
        and/or any other setup related things here.
        """
        transformers.logging.set_verbosity_error()
        num_labels = 4
        ### model_name_or_path = './models/task01_bs256_0.7538'                    # public=0.8176
        ### model_name_or_path = './models/task01_bs256_after_cf_0.7560'           # public=0.8168
        ### model_name_or_path = './models/task01_bs256_after_cf_0.7548'           # public=0.8171
        ### model_name_or_path = './models/task01_bs256_flod_100_01_0.7763'        # public=0.815
        ### model_name_or_path = './models/task23_bs256_0.7538'                    # public=0.8174
        ### model_name_or_path = './models/task01_bs256_after_cf_flod_20_0.7825'   # public=0.8160
        ### model_name_or_path = './models/task01_bs256_flod_20_0.7712'            # public=0.8152
        ### model_name_or_path = './models/task01_bs256_flod_20_0.7740'            # public=0.8175
        ### model_name_or_path = './models/task01_bs256_flod_20_0.7748'            # public=0.8167
        model_name_or_path = './models/task01_bs256_0.7540'         # public=0.8177                    Used?
        ### model_name_or_path = './models/task01_bs256_0.7547'                    # public=0.8172
        ### model_name_or_path = './models/task01_bs256_flod_20_0.7752'            # public=0.8169
        ### model_name_or_path = './models/task01_bs256_flod20_ptbdbc_0.7739'      # public=0.8173
        ### model_name_or_path = './models/task01_bs256_flod20_ptdbbc_0.7729'      # public=0.8175     Used?
        ### model_name_or_path = './models/task01_bs256_flod20_ptbdbc_0.7741'      # public=0.8172     Used?
        ### model_name_or_path = './models/task01_bs256_flod20_ptdbbc_0.7738'      # public=0.8171
        ### model_name_or_path = './models/task01_bs256_flod20_ptdbbc_0.7742'      # public=0.8168
        model_name_or_path = './models/task01_bs256_flod20_5e6_0.7741'      # public=
        model_name_or_path = './models/task01_bs256_flod20_5e6_0.7752'      # public=

        print("model_name_or_path=", model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, truncation_side='right')
        self.model = RobertaWithSampleWeight.from_pretrained(model_name_or_path, from_tf=False, config=config)
        self.model.config.label2id = self.label_to_id
        self.model.config.id2label = {id: label for label, id in config.label2id.items()}

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

    def predict(self,
                test_set_path: PathType,
                product_catalogue_path: PathType,
                predictions_output_path: PathType,
                register_progress=lambda x: print("Progress : ", x)):
        """To be implemented by the participants.

        Participants need to consume the test set (the path of which is passed) 
        and write the final predictions as a CSV file to `predictions_output_path`.

        Args:
            test_set_path: Path to the Test Set for the specific task.
            predictions_output_path: Output Path to write the predictions as a CSV file. 
            register_progress: A helper callable to register progress. Accepts a value [0, 1].
        """
        transformers.logging.set_verbosity_error()
        # dataset merged
        test_query_df = pd.read_csv(test_set_path)
        product_df = pd.read_csv(product_catalogue_path)
        test_df = pd.merge(test_query_df, product_df, how='left', left_on=['query_locale','product_id'], right_on=['product_locale', 'product_id'])
        
        raw_datasets = Dataset.from_pandas(test_df)

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
                # item = title + ' </s></s> ' + bullet_point + ' </s></s> ' + desc + ' </s></s> ' + brand + ' </s></s> ' + color     # 2022.07.08 reverse-title 测试效果  ptbdbc
                # item = title + ' </s></s> ' + desc + ' </s></s> ' + bullet_point + ' </s></s> ' + brand + ' </s></s> ' + color     # 2022.07.09 reverse-title 测试效果  ptdbbc
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
        self.model, eval_dataloader = self.accelerator.prepare(self.model, eval_dataloader)


        def metric_compute(y_pred=[], y_true=[]):
            return {
                "recall": recall_score(y_true, y_pred, average='micro'),
                "accuracy": accuracy_score(y_true, y_pred), 
                "f1": f1_score(y_true, y_pred, average='micro') , 
            }

        self.model.eval()
        predictions = []
        task1_predictions = []
        labels = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                pred = outputs.logits.argmax(dim=-1)
                task1_pred = torch.softmax(outputs.logits, dim=-1)
                predictions.append(self.accelerator.gather(pred).cpu().numpy())
                task1_predictions.append(self.accelerator.gather(task1_pred).cpu().numpy())
    
        predictions = np.concatenate(predictions)
        task1_predictions = np.concatenate(task1_predictions)

        with open(predictions_output_path, mode='w') as writer:
            writer.write("index\tprediction\tprobs\n")
            for index, item in enumerate(predictions):
                probs = task1_predictions[index]
                value_str = ','.join([str(i) for i in probs.tolist()])
                writer.write(f"{index}\t{item}\t{value_str}\n")

        def change_id(predict):
            if int(predict) == 0:
                return 'complement'
            if int(predict) == 1:
                return 'exact'
            if int(predict) == 2:
                return 'irrelevant'
            if int(predict) == 3:
                return 'substitute'
        test_df = pd.read_csv(test_set_path, skipinitialspace=True, usecols=['example_id'])
        predict_df = pd.read_csv(predictions_output_path, sep='\t', skipinitialspace=True)
        predict_df['esci_label'] =  predict_df['prediction'].map(lambda x: change_id(x))
        submission_df = pd.concat([test_df, predict_df], axis=1)
        submission_df.example_id = submission_df.example_id.astype(int)
        submission_df.to_csv(predictions_output_path, columns = ['example_id', 'esci_label'], encoding='utf8', index=False)

        return


if __name__ == "__main__":

    transformers.logging.set_verbosity_error()
    set_seed(12345)

    # Instantiate Predictor Class
    predictor = Task2Predictor()
    predictor.prediction_setup()
    
    test_set_path = "./data/task_2_multiclass_product_classification/test_public-v0.3.csv.zip"
    product_catalogue_path = "./data/task_2_multiclass_product_classification/product_catalogue-v0.3.csv.zip"

    # Generate a Random File to store predictions
    with tempfile.NamedTemporaryFile(suffix='.csv') as output_file:
        output_file_path = output_file.name
        print(output_file_path)

        # Make Predictions
        predictor.predict(
            test_set_path=test_set_path,
            product_catalogue_path=product_catalogue_path,
            predictions_output_path=output_file_path
        )
        
        ####################################################################################################
        ####################################################################################################
        ## 
        ## Adding some simple validations to ensure that the generated file has the expected structure
        ## 
        ####################################################################################################
        ####################################################################################################
        predictions_df = pd.read_csv(output_file_path)
        test_df = pd.read_csv(test_set_path)

        # Check-#1 : Sample Submission has "example_id" and "esci_label" columns
        expected_columns = ["example_id", "esci_label"]
        assert set(expected_columns) <= set(predictions_df.columns.tolist()), \
            "Predictions file's column names do not match the expected column names : {}".format(
                expected_columns)

        # Check-#2 : Sample Submission contains predictions for all example_ids
        predicted_example_ids = sorted(predictions_df["example_id"].tolist())
        expected_example_ids = sorted(test_df["example_id"].tolist())
        assert expected_example_ids == predicted_example_ids, \
            "`example_id`s present in the Predictions file do not match the `example_id`s provided in the test set"

        # Check-#3 : Predicted `esci_label`s are valid
        VALID_OPTIONS = sorted(
            ["exact", "complement", "irrelevant", "substitute"])
        predicted_esci_labels = sorted(predictions_df["esci_label"].unique())
        assert predicted_esci_labels == VALID_OPTIONS, \
            "`esci_label`s present in the Predictions file do not match the expected ESCI Lables : {}".format(
                VALID_OPTIONS)

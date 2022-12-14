#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

from parameter import parse_args
from multi_task_model import MultiTaskPretrainedModel
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    XLMRobertaConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    SchedulerType,
    get_scheduler,
    XLMRobertaForMaskedLM,
    RobertaForMaskedLM,
)
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version
os.environ['TRANSFORMERS_CACHE'] = '/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/'
os.environ['HF_DATASETS_CACHE'] = '/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/datasets/'
os.environ['HF_MODULES_CACHE'] = '/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/models/'
os.environ['HF_METRICS_CACHE'] = '/home/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/metrics/'

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    # accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files_train = {}
        data_files_vaild = {}
        data_files_fq = {}
        if args.train_mlm_file is not None:
            data_files_train["train"] = args.train_mlm_file
        if args.validation_file is not None:
            data_files_vaild["validation"] = args.validation_file
        if args.train_fake_query_file is not None:
            data_files_fq["train"] = args.train_fake_query_file
        extension = args.train_mlm_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets_train = load_dataset(extension, data_files=data_files_train)
        raw_datasets_vaild = load_dataset(extension, data_files=data_files_vaild)
        raw_datasets_fq = load_dataset(extension, data_files=data_files_fq)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # if args.config_name:
    #     config = AutoConfig.from_pretrained(args.config_name)
    # elif args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(args.model_name_or_path)
    # else:
    #     config = CONFIG_MAPPING[args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # if args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    # elif args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported by this script."
    #         "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    #     )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    model = MultiTaskPretrainedModel.from_pretrained(args.model_name_or_path, config=config)
    # model = RobertaForMaskedLM.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    # )

    print("2022-4-21 len(tokenizer)==", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    remove_column_names_train = raw_datasets_train["train"].column_names
    remove_column_names_vaild = raw_datasets_vaild["validation"].column_names
    remove_column_names_fq = raw_datasets_fq["train"].column_names
    print("2022-4-21 remove-column-names:", remove_column_names_train, ",", remove_column_names_vaild, ",", remove_column_names_fq)

    if args.mlm_max_seq_length is None:
        mlm_max_seq_length = tokenizer.model_max_length
        if mlm_max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --mlm_max_seq_length xxx."
            )
            mlm_max_seq_length = 1024
    else:
        if args.mlm_max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The mlm_max_seq_length passed ({args.mlm_max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using mlm_max_seq_length={tokenizer.model_max_length}."
            )
        mlm_max_seq_length = min(args.mlm_max_seq_length, tokenizer.model_max_length)

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=mlm_max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        print("2022-04-21, group_texts MLM Pretrain")
        """
            1.MLM?????????, ???examples???????????????????????????
            2.MLM?????????, title???????????????, ??????padding;  ???????????????GroupBy????????????  ???????????????????????????????????????
        """
        padding = "max_length" if args.pad_to_max_length else False
        fq_label_to_id = None
        vaild_label_to_id = {"exact":1, "substitute":1, "complement":0, "irrelevant":0}
        def tokenize_function_train(examples):
            return tokenizer(examples["TEXT"], return_special_tokens_mask=True)
        
        def tokenize_function_vaild_mlm(examples):
            # Remove empty lines
            examples["product_title"] = [
                line for line in examples["product_title"] if line and len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples["product_title"],
                padding=padding,
                truncation=True,
                max_length=mlm_max_seq_length,
                return_special_tokens_mask=True,
            )

        def tokenize_function_vaild(examples):
            examples['product_title'] = [
                line if line and len(line) > 0 and not line.isspace() else ' </s> ' for line in examples["product_title"] 
            ]
            texts = (
                 (examples['query'], examples['product_title'])
            )
            result = tokenizer(*texts, padding=padding, max_length=args.mlm_max_seq_length, truncation=True)
            if "esci_label" in examples:
                result["labels"] = [vaild_label_to_id[l] for l in examples["esci_label"]]
            return result

        def tokenize_function_fq(examples):
            # Tokenize the texts
            texts = (
                 (examples['faked_query'], examples['title_str'])
            )
            result = tokenizer(*texts, padding=padding, max_length=args.mlm_max_seq_length, truncation=True)
            if "faked_query_label" in examples:
                if fq_label_to_id is not None:
                    result["labels"] = [fq_label_to_id[l] for l in examples["faked_query_label"]]
                else:
                    result["labels"] = examples["faked_query_label"]
            return result

        with accelerator.main_process_first():
            tokenized_datasets_train = raw_datasets_train.map(
                tokenize_function_train,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=remove_column_names_train,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on every text in dataset---Train",
            )
            tokenized_datasets_vaild = raw_datasets_vaild.map(
                tokenize_function_vaild_mlm,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=remove_column_names_vaild,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on every text in dataset---Vaild",
            )
            train_fq_dataset = raw_datasets_fq.map(
                tokenize_function_fq,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=remove_column_names_fq,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on every text in dataset---FakedQuery",
            )

            print("2022-04-21 header= ", tokenized_datasets_train, tokenized_datasets_vaild, train_fq_dataset)
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # mlm_max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= mlm_max_seq_length:
                total_length = (total_length // mlm_max_seq_length) * mlm_max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + mlm_max_seq_length] for i in range(0, total_length, mlm_max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with accelerator.main_process_first():
            tokenized_datasets_train = tokenized_datasets_train.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {mlm_max_seq_length}",
            )

    train_dataset = tokenized_datasets_train["train"]
    train_fq_dataset = train_fq_dataset["train"]
    eval_dataset = tokenized_datasets_vaild["validation"]

    # Conditional for small test subsets
    if len(train_fq_dataset) > 3:
        for index in random.sample(range(1000), 3):
            logger.info(f"Sample {index} of the train_dataset: {train_dataset[index]}.")
            logger.info(f"Sample {index} of the train_fq_dataset: {train_fq_dataset[index]}.")
            logger.info(f"Sample {index} of the eval_dataset: {eval_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    data_fq_collator = default_data_collator

    print("2022-4-25: mlm_max_seq_length=", mlm_max_seq_length)
    # DataLoaders creation:
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_lm_collator, batch_size=args.per_device_train_batch_size)
    train_fq_dataloader = DataLoader(train_fq_dataset, shuffle=True, collate_fn=data_fq_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_lm_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    min_steps_per_epoch = min([len(train_fq_dataloader), len(train_dataloader)])
    logger.info(f"2022-04-30: min_steps_per_epoch = {min_steps_per_epoch}")
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(min_steps_per_epoch / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, train_fq_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                    model, optimizer, train_dataloader, train_fq_dataloader, eval_dataloader, lr_scheduler)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mlm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset), len(train_fq_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            resume_step = None
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        if "epoch" in path:
            args.num_train_epochs -= int(path.replace("epoch_", ""))
        else:
            resume_step = int(path.replace("step_", ""))
            args.num_train_epochs -= resume_step // len(train_dataloader)
            resume_step = (args.num_train_epochs * len(train_dataloader)) - resume_step

    data_loaders = {"FakedQuery":train_fq_dataloader, "MLM":train_dataloader}
    data_iterators = {"FakedQuery":iter(train_fq_dataloader), "MLM":iter(train_dataloader)}
    min_perplexity = 100000.0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step in range(min_steps_per_epoch):
            for name in data_loaders.keys():    # 2022.04.29 multi-task pretraining.   # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L681
                data_iter = data_iterators[name]
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loaders[name])
                    data_iterators[name] = data_iter
                    batch = next(data_iter)
                outputs = model(**batch, task_name=name)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                # print(batch.keys(), name, loss)
                accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == min_steps_per_epoch - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
        
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= args.max_train_steps:
                    break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                # print("2022-04-25: eval-batch" , batch.keys())
                outputs = model(**batch, task_name='MLM')

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            exp_perplexity = math.exp(torch.mean(losses))
            perplexity = torch.mean(losses)             # 2022-04-25 ?????????????????????????????????exp??????, ??????????????????
        except OverflowError:
            exp_perplexity = float("inf")
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} exp_perplexity: {exp_perplexity}")

        if args.with_tracking:
            accelerator.log(
                {"perplexity": perplexity, "train_loss": total_loss, "epoch": epoch, "step": completed_steps},
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            # accelerator.save_state(output_dir)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

        if perplexity < min_perplexity and args.output_dir is not None:
            min_perplexity = min_perplexity
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()
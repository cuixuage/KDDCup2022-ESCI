"""
    代码来源:
    https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py
"""
from parameter import parse_args
import logging, string, math, os, re, random
from pathlib import Path
import torch
import numpy as np
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score
from RobertaWithSampleWeight import RobertaWithSampleWeight
from utils import EMA,get_vocab_from_file, FGM, util_loss_fct

import transformers
from accelerate import Accelerator
# from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version
os.environ['TRANSFORMERS_CACHE'] = '/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/.cache/huggingface/'
os.environ['HF_DATASETS_CACHE'] = '/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/.cache/huggingface/datasets/'
os.environ['HF_MODULES_CACHE'] = '/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/.cache/huggingface/models/'
os.environ['HF_METRICS_CACHE'] = '/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/.cache/huggingface/metrics/'



logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
def metric_compute(y_pred=[], y_true=[]):
    return {
        "recall": recall_score(y_true, y_pred, average='micro'),
        "accuracy": accuracy_score(y_true, y_pred), 
        "f1": f1_score(y_true, y_pred, average='micro') , 
    }
transformers.logging.set_verbosity_error()


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
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

    transformers.logging.set_verbosity_error()  # 2022-05-17

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
            # repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        print("extension=", extension, data_files)
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    is_regression = raw_datasets["train"].features["esci_label"].dtype in ["float32", "float64"]

    label_list = raw_datasets["train"].unique("esci_label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    num_labels = 2              # 2022.06.26 task3, 二分类任务


    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    # config.update({"hidden_dropout_prob": 0.0})         

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, truncation_side='right')
    model = RobertaWithSampleWeight.from_pretrained(args.model_name_or_path, from_tf=False, config=config)

    label_to_id = None
    brand_vocab = get_vocab_from_file('/home/cuixuange/kddcup_2022/data_process/extra_vocab/brand.txt')
    color_vocab = get_vocab_from_file('/home/cuixuange/kddcup_2022/data_process/extra_vocab/color.txt')
    # emb_bag_vocab = get_vocab_from_file('/home/cuixuange/kddcup_2022/data_process/extra_ngram/emb_bag_vocab.txt')
    emb_bag_vocab = get_vocab_from_file('/home/cuixuange/kddcup_2022/data_process/extra_ngram/word2vec.wordvectors.vocab_char3.txt')
    label_to_id =  {
        "complement": 0,
        "exact": 0,
        "irrelevant": 0,
        "substitute": 1
    }
    model.config.label2id = label_to_id
    model.config.id2label = {0: 'no_substitute', 1:'substitute'}            # 2022.06.26 添加2分类

    padding = "max_length" if args.pad_to_max_length else False
    print("2022-04-10, label_to_id=, config.num_labels=", label_to_id, args.push_to_hub, padding, config.num_labels)   # 2022.04.20 padding = max_length

    def preprocess_function(examples):
        sample_weights = [0.3] * len(examples['esci_label'])

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
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)  # 2022.05.03 加入翻译后的数据,有些Query长度翻译的很长,不能用only-second阶段方式

        if "product_brand" in examples:
            result["brand_idx"] = []
            for item in examples['product_brand']:
                if not item: 
                    result['brand_idx'].append(0)       # out of vocabulary
                    continue
                item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                if item in brand_vocab:
                    result['brand_idx'].append(int(brand_vocab[item]))
                else:
                    result['brand_idx'].append(0) # out of vocabulary
        if "product_color_name" in examples:
            result["color_idx"] = []
            for item in examples['product_color_name']:
                if not item: 
                    result['color_idx'].append(0) # out of vocabulary
                    continue
                item = str(item).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
                if item in color_vocab:
                    result['color_idx'].append(int(color_vocab[item]))
                else:
                    result['color_idx'].append(0)   # out of vocabulary   

        if "esci_label" in examples:
            result["labels"] = [label_to_id[l] for l in examples["esci_label"]]

        if "query_locale" in examples:
            for idx, locale in enumerate(examples['query_locale']):
                if locale == 'us' or locale == 'es' or locale == 'jp':
                    sample_weights[idx] = 1.0
        result["sample_weights"] = sample_weights

        # 2.chars-tokenizer
        def change_2_chars_idx(item_str):
            item_str = str(item_str).translate(str.maketrans('\n', ' ', string.punctuation)).lower().strip()
            ids = []
            n = 3
            if len(item_str) <= n:
                if item_str in emb_bag_vocab:
                    ids.append(emb_bag_vocab.get(item_str, 0))
            else:
                for i in range(len(item_str) - n):
                    i_v = item_str[i:i+n].strip()
                    if i_v in emb_bag_vocab:
                        ids.append(emb_bag_vocab.get(i_v, 0))  
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


    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            batch_size=args.per_device_train_batch_size,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        print('pad_to_max_length=', args.pad_to_max_length)
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    """
        variable-length, 即padding=batch_longest 减少计算量(双塔模型, Query长度很短)
        配置要求: Dataloader.batch_size, num_proc=1, shuffle=False;    tokenizer:padding=longest
        缺点:  分词耗时较久, 约1h, 当然最后训练时间相对会更快
    """
    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

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

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    ############################################################################################# 2022.05.10 准备训练
    ema = EMA(model, 0.999)
    ema.register()
    fgm = FGM(model)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    args.num_warmup_steps = int(0.02 * args.max_train_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    metric_a = load_metric("/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/datasets_metric/accuracy/accuracy.py")
    metric_p = load_metric("/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/datasets_metric/precision/precision.py")
    metric_r = load_metric("/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/datasets_metric/recall/recall.py")
    metric_f = load_metric("/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/datasets_metric/f1/f1.py")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    stand_eval_steps = int(0.3 * len(train_dataloader)) + 1    # 每0.3epoch保存一次模型
    print('stand_eval_steps=', stand_eval_steps)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    # for i in model.state_dict():
    #     print(i)

    max_eval_metric_f = 0.0
    fgm_epsilon = 0.5
    r_drop_lambda = 0.5
    fintune_loss_num = 3.0      # 3个loss
    # for epoch in range(args.num_train_epochs):
    #     # ###
    #     # if epoch >= 2: break
    #     # ###
    #     model.train()
    #     for step, batch in enumerate(train_dataloader):
    #         # 1. normal loss
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         logits = outputs.logits
    #         loss = loss / args.gradient_accumulation_steps
    #         accelerator.backward(loss / fintune_loss_num, retain_graph=True)
    #         # 2.fgm loss
    #         fgm.attack(fgm_epsilon, 'word_embeddings')
    #         outputs_adv = model(**batch)
    #         loss_adv = outputs_adv.loss
    #         logits_adv = outputs_adv.logits
    #         loss_adv = loss_adv / args.gradient_accumulation_steps
    #         accelerator.backward(loss_adv / fintune_loss_num, retain_graph=True) # 累加, fgm梯度 == ce_Loss + perturbation_loss
    #         fgm.restore('word_embeddings')                
    #         # 3.r-drop loss
    #         kl_loss = util_loss_fct(logits_adv, logits, batch['sample_weights'], None, loss_name='r-drop-kl')
    #         kl_loss = r_drop_lambda * kl_loss / args.gradient_accumulation_steps
    #         accelerator.backward(kl_loss / fintune_loss_num) # 累加, r-drop梯度 == ce_Loss + perturbation_loss + kl_loss

    #         if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
    #             optimizer.step()
    #             lr_scheduler.step()
    #             ema.update()    #保存平滑后的参数
    #             optimizer.zero_grad()
    #             progress_bar.update(1)
    #             completed_steps += 1

    #         if completed_steps >= args.max_train_steps:
    #             break

    #         if step % (953 * args.gradient_accumulation_steps) == 0:   # 0.1 epoch
    #             model.eval()
    #             ema.apply_shadow()
    #             for step, batch in enumerate(eval_dataloader):
    #                 with torch.no_grad():
    #                     outputs = model(**batch)
    #                     predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
    #                     metric_a.add_batch(
    #                         predictions=accelerator.gather(predictions),
    #                         references=accelerator.gather(batch["labels"]),
    #                     )
    #                     metric_p.add_batch(
    #                         predictions=accelerator.gather(predictions),
    #                         references=accelerator.gather(batch["labels"]),
    #                     )
    #                     metric_r.add_batch(
    #                         predictions=accelerator.gather(predictions),
    #                         references=accelerator.gather(batch["labels"]),
    #                     )
    #                     metric_f.add_batch(
    #                         predictions=accelerator.gather(predictions),
    #                         references=accelerator.gather(batch["labels"]),
    #                     )
    #             eval_metric_a = metric_a.compute()
    #             eval_metric_p = metric_p.compute(average='micro')
    #             eval_metric_r = metric_r.compute(average='micro')
    #             eval_metric_f = metric_f.compute(average='micro')           #2022.04.09  多个metric指标
    #             logger.info(f"epoch {epoch}: {eval_metric_a} {eval_metric_p} {eval_metric_r} {eval_metric_f}")

    #             if args.output_dir is not None and eval_metric_f['f1'] > max_eval_metric_f:  # 2021.11.10   保存最优指标的模型
    #                 max_eval_metric_f = eval_metric_f['f1']
    #                 accelerator.wait_for_everyone()
    #                 unwrapped_model = accelerator.unwrap_model(model)
    #                 unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    #                 if accelerator.is_main_process:
    #                     tokenizer.save_pretrained(args.output_dir)
    #                     if args.push_to_hub:
    #                         repo.push_to_hub(commit_message="End of training")

    #             ema.restore()
    #             model.train()           # 2022.07.12 非常关键，需要重新打开训练模式。不然效果会变差很多。e.g. dropout, layernorm

    #     model.eval()
    #     ema.apply_shadow()
    #     for step, batch in enumerate(eval_dataloader):
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #             predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
    #             metric_a.add_batch(
    #                 predictions=accelerator.gather(predictions),
    #                 references=accelerator.gather(batch["labels"]),
    #             )
    #             metric_p.add_batch(
    #                 predictions=accelerator.gather(predictions),
    #                 references=accelerator.gather(batch["labels"]),
    #             )
    #             metric_r.add_batch(
    #                 predictions=accelerator.gather(predictions),
    #                 references=accelerator.gather(batch["labels"]),
    #             )
    #             metric_f.add_batch(
    #                 predictions=accelerator.gather(predictions),
    #                 references=accelerator.gather(batch["labels"]),
    #             )
    #     eval_metric_a = metric_a.compute()
    #     eval_metric_p = metric_p.compute(average='micro')
    #     eval_metric_r = metric_r.compute(average='micro')
    #     eval_metric_f = metric_f.compute(average='micro')           #2022.04.09  多个metric指标
    #     logger.info(f"epoch {epoch}: {eval_metric_a} {eval_metric_p} {eval_metric_r} {eval_metric_f}")

    #     if args.output_dir is not None and eval_metric_f['f1'] > max_eval_metric_f:  # 2021.11.10   保存最优指标的模型
    #         max_eval_metric_f = eval_metric_f['f1']
    #         accelerator.wait_for_everyone()
    #         unwrapped_model = accelerator.unwrap_model(model)
    #         unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    #         if accelerator.is_main_process:
    #             tokenizer.save_pretrained(args.output_dir)
    #             if args.push_to_hub:
    #                 repo.push_to_hub(commit_message="End of training")

    #     ema.restore()



    # ################################################################################### 2022.04.09 predict
    if args.output_dir is not None:  # 2021.11.10   Predict阶段
        output_test_file = os.path.join(args.output_dir, "test_results.txt.head1k")
        with open(output_test_file, mode='w') as writer:
            predictions = []
            task1_predictions = []
            labels = []
            model.eval()
            count = 0
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)
                    pred = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    task1_pred = torch.softmax(outputs.logits, dim=-1)

                    predictions.append(accelerator.gather(pred).cpu().numpy())
                    task1_predictions.append(accelerator.gather(task1_pred).cpu().numpy())
                    labels.append(accelerator.gather(batch["labels"]).cpu().numpy())

            predictions = np.concatenate(predictions)
            task1_predictions = np.concatenate(task1_predictions)
            labels = np.concatenate(labels)
            metric_dict = metric_compute(predictions, labels)
            print(metric_dict)

            logger.info(f"***** Test results *****")
            writer.write("index\tprediction\tprobs\n")
            ### task2,task3 prediction
            for index, item in enumerate(predictions):
                probs = task1_predictions[index]
                value_str = ','.join([str(i) for i in probs.tolist()])
                writer.write(f"{index}\t{item}\t{value_str}\n")


if __name__ == "__main__":
    main()



# 0.00075229 0.36768094 0.01072851 0.6208382 ] <class 'numpy.ndarray'>
# [0.00144366 0.7555662  0.11918692 0.12380328] <class 'numpy.ndarray'>
# [0.02305878 0.16715544 0.72630703 0.08347876] <class 'numpy.ndarray'>
# [6.2353991e-04 9.9452388e-01 4.7583331e-04 4.3767104e-03] <class 'numpy.ndarray'>
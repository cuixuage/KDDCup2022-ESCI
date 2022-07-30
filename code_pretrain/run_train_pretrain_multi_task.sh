source /etc/profile && source ~/.bashrc
conda deactivate
conda activate huggingface
export TRANSFORMERS_CACHE="/home/cuixuange/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/"
export HF_DATASETS_CACHE="/home/cuixuange/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/datasets/"
export HF_MODULES_CACHE="/home/cuixuange/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/models/"
export HF_METRICS_CACHE="/home/cuixuange/kddcup_2022/v0.2_train_pretrain/.cache/huggingface/metrics/"

export DATA_DIR="/home/cuixuange/kddcup_2022/data_process/"
export OUTOUT_DIR="/home/cuixuange/kddcup_2022/v0.2_train_pretrain/output"
export CUDA_VISIBLE_DEVICES=3

python3 /home/cuixuange/kddcup_2022/v0.2_train_pretrain/run_mlm_no_trainer_multi_task_extra_embedding.py \
    --model_name_or_path '/home/cuixuange/kddcup_2022/huggingface_models/kddcup_2022/xlm-roberta-large' \
    --mlm_max_seq_length 256 \
    --mlm_probability 0.15 \
    --pad_to_max_length \
    --preprocessing_num_workers 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --seed 12345 \
    --train_mlm_file $DATA_DIR/Product_shuffled_220W.csv \
    --train_fake_query_file $DATA_DIR/Product_shuffled_220W_x4_NSP_in_pretrain-Append.csv\
    --validation_file $DATA_DIR/flod_5_valid.csv \
    --output_dir $OUTOUT_DIR/multi_task_mlm_fq_bc_loss_shuffled_large_model_xlmr



# roberta-large-mnli
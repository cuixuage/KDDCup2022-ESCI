source /etc/profile && source ~/.bashrc
conda deactivate
conda activate huggingface
export TRANSFORMERS_CACHE="/home/kddcup_2022/v0.2_train/.cache/huggingface/"
export HF_DATASETS_CACHE="/home/kddcup_2022/v0.2_train/.cache/huggingface/datasets/"
export HF_MODULES_CACHE="/home/kddcup_2022/v0.2_train/.cache/huggingface/models/"
export HF_METRICS_CACHE="/home/kddcup_2022/v0.2_train/.cache/huggingface/metrics/"

export DATA_DIR="/home/kddcup_2022/data_process/"
export OUTOUT_DIR="/home/kddcup_2022/v0.2_train/output/"
export CUDA_VISIBLE_DEVICES=3

python3 /home/kddcup_2022/v0.2_train/run_glue_no_trainer.py \
    --model_name_or_path '/home/kddcup_2022/v0.2_train_pretrain/output/multi_task_mlm_fq_bc_loss_shuffled_large_model_3_continue_final'  \
    --max_length 256 \
    --pad_to_max_length \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --output_dir $OUTOUT_DIR/task2_tmp_test \
    --seed 12345 \
    --train_file   $DATA_DIR/extra_confident_learning/flod_89_train-head1k.csv\
    --validation_file $DATA_DIR/extra_confident_learning/flod_89_valid-head1k.csv


# --train_file   $DATA_DIR/extra_single_language/us_finetune_train_wo_es-wi_ci.csv  \
# --validation_file $DATA_DIR/extra_single_language/us_finetune_valid.csv

# --train_file   $DATA_DIR/extra_data_with_task1_rebalance/task2/rebalanced_public_data.csv \
# --validation_file $DATA_DIR/extra_data_with_task1/task2/flod_5_valid.csv


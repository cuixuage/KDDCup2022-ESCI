source /etc/profile && source ~/.bashrc
conda deactivate
conda activate huggingface
export TRANSFORMERS_CACHE="/home/kddcup_2022/v0.2_train_for_task1/.cache/huggingface/"
export HF_DATASETS_CACHE="/home/kddcup_2022/v0.2_train_for_task1/.cache/huggingface/datasets/"
export HF_MODULES_CACHE="/home/kddcup_2022/v0.2_train_for_task1/.cache/huggingface/models/"
export HF_METRICS_CACHE="/home/kddcup_2022/v0.2_train_for_task1/.cache/huggingface/metrics/"

export DATA_DIR="/home/kddcup_2022/data_process/"
export OUTOUT_DIR="/home/kddcup_2022/v0.2_train_for_task1/output/"
export CUDA_VISIBLE_DEVICES=3

python3 /home/kddcup_2022/v0.2_train_for_task1/run_glue_no_trainer.py \
    --model_name_or_path '/home/kddcup_2022/v0.2_train_pretrain/output/multi_task_mlm_fq_bc_loss_shuffled_large_model_3_continue_final' \
    --max_length 256 \
    --pad_to_max_length \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --output_dir $OUTOUT_DIR/regression_for_task1_ndcg \
    --seed 12345789 \
    --train_file   $DATA_DIR/extra_confident_learning/flod_01_train-ci.csv \
    --validation_file $DATA_DIR/extra_data_with_task1/task2/flod_5_valid.csv




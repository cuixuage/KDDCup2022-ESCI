source /etc/profile && source ~/.bashrc
conda deactivate
conda activate huggingface
export TRANSFORMERS_CACHE="/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/.cache/huggingface/"
export HF_DATASETS_CACHE="/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/.cache/huggingface/datasets/"
export HF_MODULES_CACHE="/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/.cache/huggingface/models/"
export HF_METRICS_CACHE="/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/.cache/huggingface/metrics/"

export DATA_DIR="/home/cuixuange/kddcup_2022/data_process/"
export OUTOUT_DIR="/home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/output/"
export CUDA_VISIBLE_DEVICES=0

python3 /home/cuixuange/kddcup_2022/v0.2_train_for_task3_embbag/run_glue_no_trainer.py \
    --model_name_or_path '/home/cuixuange/kddcup_2022/v0.2_train_pretrain/output/multi_task_mlm_fq_bc_loss_shuffled_large_model_3_continue_final' \
    --max_length 256 \
    --pad_to_max_length \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --output_dir $OUTOUT_DIR/task3_2_classfication_embbag_frompretrained \
    --seed 12346 \
    --train_file   $DATA_DIR/extra_confident_learning/flod_20_01_train_wi_es-ci-preshuffled.csv \
    --validation_file $DATA_DIR/extra_confident_learning/flod_20_01_valid_wi_es-ci-preshuffled.csv


    # --train_file   $DATA_DIR/extra_confident_learning/flod_20_01_train_wi_es-ci-preshuffled.csv \
    # --validation_file $DATA_DIR/extra_confident_learning/flod_20_01_valid_wi_es-ci-preshuffled.csv
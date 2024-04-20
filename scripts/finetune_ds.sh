#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# set the paths
MODEL="path to the Qwen-VL-Chat checkpoint"
IMAGE_DIR="path to the object365 image directory"

# SC-Tune hyperparameters
caption_update_steps=200
bbox_update_steps=100
caption_lr=5e-7
bbox_lr=1e-6
kl_coef=0.01


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --train_data_path $IMAGE_DIR \
    --bf16 True \
    --fix_vit True \
    --num_train_epochs 1 \
    --caption_update_steps $caption_update_steps \
    --bbox_update_steps $bbox_update_steps \
    --caption_lr $caption_lr \
    --bbox_lr $bbox_lr \
    --kl_coef $kl_coef \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 10 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 384 \
    --gradient_checkpointing True \
    --deepspeed scripts/ds_config_zero2.json \
    --output_dir "outputs/"
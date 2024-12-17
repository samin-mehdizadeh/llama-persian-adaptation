#!/bin/bash
#SBATCH --job-name=pre-train
#SBATCH --account=
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:8         
#SBATCH --mem-per-cpu=
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --constraint=
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT_DIR= YOUR_ROOT_DIRECTORY
SRC_DIR= YOUR_SOURCE_CODE_DIRECTORY

export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export MASTER_PORT=12345
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


lr=2e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8

pretrained_model= ORIGINAL_MODEL
farsi_tokenizer_path= MERGED_TOKENIZER_DIRECTORY
dataset_dir= PATH_TO_PRETRAIN_DATA
data_cache= PATH_TO_DATA_CACHE_DIRECTORY
cache_dir= PATH_TO_CACHE_DIRECTORY
output_dir= PATH_TO_FINAL_MODEL
deepspeed_config_file= PATH_TO_DEEP_SPEED_CONFIG

cd $SRC_DIR
set -x

torchrun --nnodes 1 --nproc_per_node 8 run_clm_pt_with_peft.py \
    --llama \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${farsi_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --cache_dir ${cache_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy steps\
    --eval_steps 2000\
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --overwrite_output_dir False \
    --freeze_transformer False \
    --use_lora
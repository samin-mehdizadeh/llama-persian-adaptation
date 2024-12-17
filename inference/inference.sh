#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --gres=gpu:1
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --time=
#SBATCH --constraint=
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT_DIR= YOUR_ROOT_DIRECTORY
SRC_DIR= YOUR_SOURCE_CODE_DIRECTORY

base_model= YOUR_PRETRAINED_MODEL
lora_model= SFT_LORA_MODEL
data_dir= TEST_FILE_DIRECTORY
predictions_dir= OUTPUT_PREDICTION_DIRECTORY
cache_dir= HUGGINGFACE_CACHE_DIRECTORY
cd $SRC_DIR

python3 inference.py --base_model ${base_model} \
                         --data_dir ${data_dir} --lora_model ${lora_model}\
                         --predictions_dir ${predictions_dir} --cache_dir ${cache_dir} --with_prompt 
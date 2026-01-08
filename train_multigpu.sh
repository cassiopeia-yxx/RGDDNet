#!/bin/bash

# ====================================
# 多GPU训练配置（直接修改下面的参数）
# ====================================

config="Options/LOL_Deraining.yml"

gpu_ids="0,1"

master_port=4321

gpu_count=$(echo $gpu_ids | tr -cd ',' | wc -c)
gpu_count=$((gpu_count + 1))

# pytorch2.x
CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$gpu_count --master_port=$master_port basicsr/train.py --opt $config --launcher pytorch

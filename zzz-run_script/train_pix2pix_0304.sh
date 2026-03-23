#!/bin/bash

# 1. 硬件设置
export CUDA_VISIBLE_DEVICES="4,5"
NUM_GPUS=1

# 2. 路径设置
DATAROOT="/data2/xujr/idr_data/dataset_processed/idr0003-breker-plasticity/screenA"
CHECKPOINTS_DIR="/data2/xujr/output_model"
NAME="pix2pix_experiment_0304_01"

# ================= 修改区域 =================
DIR_A="channel_0" # 请替换为实际的目录名称，如 channel_0 等
DIR_B="channel_1" # 请替换为实际的目录名称，如 channel_1 等
# ===========================================

# 3. 关键模型设置
# 使用 pix2pix 模型
MODEL="pix2pix"
# 如果数据是分开的对应图片，请使用 aligned_separated
DATASET_MODE="aligned_separated"

NET_G="unet_256"

# Norm Layer: pix2pix 默认用 batch，但在 batch_size=1 时建议用 instance
# NORM_LAYER="batch" 
NORM_LAYER="instance"

INPUT_NC=3
OUTPUT_NC=3

BATCH_SIZE_PER_GPU=8
LR="0.0002"
N_EPOCHS=50
N_EPOCHS_DECAY=50
MASTER_PORT=29501

echo "-----------------------------------------------------"
echo "启动 Pix2Pix 训练..."
echo "模型: $MODEL | NetG: $NET_G | Dataset: $DATASET_MODE"
echo "-----------------------------------------------------"

/data2/xujr/conda-envs/cut/bin/torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --model "$MODEL" \
    --dataset_mode "$DATASET_MODE" \
    --input_nc $INPUT_NC \
    --output_nc $OUTPUT_NC \
    --netG "$NET_G" \
    --norm "$NORM_LAYER" \
    --direction AtoB \
    --lambda_L1 100.0 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --lr "$LR" \
    --n_epochs $N_EPOCHS \
    --n_epochs_decay $N_EPOCHS_DECAY \
    --preprocess none \
    --no_dropout \
    --dir_A "$DIR_A" \
    --dir_B "$DIR_B"
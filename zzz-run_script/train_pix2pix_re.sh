#!/bin/bash

# 1. 硬件设置
export CUDA_VISIBLE_DEVICES="3"
NUM_GPUS=1

# 2. 路径设置
DATAROOT="/data2/xujr/msi_if_npy"
CHECKPOINTS_DIR="/data2/xujr"
NAME="if_msi_pix2pix_v1_BtoA" # 1. 修改模型名称，避免覆盖

# 3. 关键模型设置
# 使用 pix2pix 模型
MODEL="pix2pix"
# 使用我们刚写的配对 Dataset
DATASET_MODE="aligned_npy"

NET_G="unet_256"

# Norm Layer: pix2pix 默认用 batch，但在 batch_size=1 时建议用 instance
NORM_LAYER="instance" 

# 恢复 INPUT_NC 和 OUTPUT_NC, 因为 direction BtoA 会自动处理
INPUT_NC=3
OUTPUT_NC=1

BATCH_SIZE_PER_GPU=1
LR="0.0002"
N_EPOCHS=100
N_EPOCHS_DECAY=100
MASTER_PORT=29502

echo "-----------------------------------------------------"
echo "启动 Pix2Pix 反向训练 (B->A)..."
echo "模型: $MODEL | NetG: $NET_G | Dataset: $DATASET_MODE"
echo "-----------------------------------------------------"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --model "$MODEL" \
    --dataset_mode "$DATASET_MODE" \
    --input_nc $INPUT_NC \
    --output_nc $OUTPUT_NC \
    --netG "$NET_G" \
    --norm "$NORM_LAYER" \
    --direction BtoA \
    --lambda_L1 100.0 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --lr "$LR" \
    --n_epochs $N_EPOCHS \
    --n_epochs_decay $N_EPOCHS_DECAY \
    --preprocess none \
    --no_dropout
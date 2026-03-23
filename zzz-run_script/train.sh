#!/bin/bash

# 1. 硬件设置
export CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUS=2
NORM_LAYER="instance"

# 2. 路径设置
DATAROOT="/data2/xujr/msi_if_npy"
CHECKPOINTS_DIR="/data2/xujr"

# 改个名字，以免覆盖之前的实验
NAME="if_msi_cyclegan_rescue_unet_v1"

DATASET_MODE="unaligned_npy"

# 3. 模型参数 (IF -> MSI)
INPUT_NC=1
OUTPUT_NC=3

NET_G="unet_256"

BATCH_SIZE_PER_GPU=1
LR="0.0002"

N_EPOCHS=200
N_EPOCHS_DECAY=50

# 5. 损失函数权重
LAMBDA_A=20.0
LAMBDA_B=20.0

# 因为通道数不同 身份损失依然只能为 0
LAMBDA_IDENTITY=0.0

# 6. 其他
PREPROCESS="none"
MASTER_PORT=29505

# 7. 保存设置
SAVE_EPOCH_FREQ=10

# calculate total epochs for logging
TOTAL_EPOCHS=$((N_EPOCHS + N_EPOCHS_DECAY))


echo "-----------------------------------------------------"
echo "正在启动 CycleGAN 训练..."
echo "架构: $NET_G (UNet)"
echo "Cycle权重: $LAMBDA_A | $LAMBDA_B"
echo "实验名称: $NAME"
echo "总训练轮数 (n_epochs + n_epochs_decay): $TOTAL_EPOCHS"
echo "每 $SAVE_EPOCH_FREQ 个 epoch 保存一次模型"
echo "-----------------------------------------------------"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --model cycle_gan \
    --dataset_mode "$DATASET_MODE" \
    --input_nc $INPUT_NC \
    --output_nc $OUTPUT_NC \
    --netG "$NET_G" \
    --norm "$NORM_LAYER" \
    --batch_size $BATCH_SIZE_PER_GPU \
    --lr "$LR" \
    --n_epochs $N_EPOCHS \
    --n_epochs_decay $N_EPOCHS_DECAY \
    --save_epoch_freq $SAVE_EPOCH_FREQ \
    --lambda_A $LAMBDA_A \
    --lambda_B $LAMBDA_B \
    --lambda_identity $LAMBDA_IDENTITY \
    --preprocess "$PREPROCESS" \
    --no_dropout
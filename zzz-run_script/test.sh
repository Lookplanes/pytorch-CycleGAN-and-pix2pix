#!/bin/bash

# =========================================================
#                 推理 (Test) 脚本
# =========================================================

# 1. 硬件与路径
# ---------------------------------------------------------
# 推理只需要 1 张卡
export CUDA_VISIBLE_DEVICES="0"

# 数据集路径
DATAROOT="/data2/xujr/msi_if_npy_full"

# 训练结果所在的【上级】目录
CHECKPOINTS_DIR="/data2/xujr"

# 实验名称 
# NAME="if_msi_cyclegan_ddp_v1"
# NAME="if_msi_pix2pix_v1_BtoA"
NAME="if_msi_cyclegan_rescue_unet_v1"

# 2. 模型参数 
# ---------------------------------------------------------
MODEL="cycle_gan"
# MODEL="pix2pix"

# 使用我们自定义的 npy dataset
# DATASET_MODE="unaligned_npy"
DATASET_MODE="aligned_npy"

INPUT_NC=1
OUTPUT_NC=3

# DIRECTION="BtoA"


# NET_G="resnet_9blocks"
# NORM_LAYER="instance"
NET_G="unet_256"
NORM_LAYER="instance"


PREPROCESS="none"

# 测试多少张图片 (默认 50，设为 -1 则测试所有图片)
NUM_TEST=1000

# 设置extra参数
EXTRA_ARGS=""
if [ "$DIRECTION" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --direction $DIRECTION"
fi

# =========================================================
#                  启动命令
# =========================================================

echo "-----------------------------------------------------"
echo "开始推理..."
echo "加载模型: $CHECKPOINTS_DIR/$NAME"
echo "读取数据: $DATAROOT/testA"
echo "-----------------------------------------------------"

python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --model "$MODEL" \
    --dataset_mode "$DATASET_MODE" \
    --input_nc $INPUT_NC \
    --output_nc $OUTPUT_NC \
    --netG "$NET_G" \
    --norm "$NORM_LAYER" \
    --preprocess "$PREPROCESS" \
    --no_dropout \
    --num_test $NUM_TEST \
    --eval \
    $EXTRA_ARGS
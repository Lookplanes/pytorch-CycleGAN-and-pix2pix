#!/bin/bash

# =========================================================
#                 推理 (Test) 脚本
# =========================================================

# 1. 硬件与路径
# ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES="4"

# 数据集路径 
# （推理阶段，需要有 $DATAROOT/$DIR_A 和 $DATAROOT/$DIR_B，
# 由于 test 代码默认会去找 testA 或您自定义的目录，
# 请确保目录下有对应的频道目录）
DATAROOT="/data2/xujr/idr_data/dataset_processed/idr0003_test_samples/screenA"

# 训练结果所在的【上级】目录
CHECKPOINTS_DIR="/data2/xujr/output_model"

# 实验名称 必须要和训练时保持一致，才能找到模型
# NAME="pix2pix_experiment_0304_01"
NAME="pix2pix_idr0003_c1Toc0"

# ================= 修改区域 =================
# 指定您要测试的通道名。它会去 $DATAROOT 找这个目录加载图片。
# DIR_A="channel_0"
# DIR_B="channel_1"
DIR_A="channel_1" 
DIR_B="channel_0"
# ===========================================

# 2. 模型参数 
# ---------------------------------------------------------
MODEL="pix2pix"

# 和训练时保持一致
DATASET_MODE="aligned_separated"

INPUT_NC=3
OUTPUT_NC=3

NET_G="unet_256"
NORM_LAYER="instance"

PREPROCESS="none"

# 测试多少张图片 (默认 50，设为极大的数或根据所需指定)
NUM_TEST=1000

# =========================================================
#                  启动命令
# =========================================================

echo "-----------------------------------------------------"
echo "开始推理..."
echo "加载模型: $CHECKPOINTS_DIR/$NAME"
echo "读取数据: $DATAROOT/$DIR_A"
echo "-----------------------------------------------------"

/data2/xujr/conda-envs/cut/bin/python test.py \
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
    --dir_A "$DIR_A" \
    --dir_B "$DIR_B"

    $EXTRA_ARGS
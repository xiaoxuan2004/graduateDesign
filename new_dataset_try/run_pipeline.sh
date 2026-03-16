#!/bin/bash
# SToFM完整流程运行脚本
# 1. 数据预处理
# 2. 提取embeddings

set -e

BASE_DIR="/data/xiaox/projects/20250801_aging"
SCRIPT_DIR="${BASE_DIR}/scripts/graduate_design"

echo "=========================================="
echo "SToFM 完整流程"
echo "=========================================="

# 步骤1: 数据预处理
echo ""
echo "步骤1: 数据预处理"
echo "------------------------------------------"
python3 ${SCRIPT_DIR}/preprocess.py

# 步骤2: 提取embeddings
echo ""
echo "步骤2: 提取embeddings"
echo "------------------------------------------"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoint/ckpt"
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "错误: Checkpoint目录不存在: ${CHECKPOINT_DIR}"
    echo "请先解压checkpoint文件:"
    echo "cd ${SCRIPT_DIR}/checkpoint && unzip checkpoit.zip"
    exit 1
fi

python3 ${SCRIPT_DIR}/get_embeddings.py \
  --cell_encoder_path ${CHECKPOINT_DIR}/cell_encoder \
  --config_path ${CHECKPOINT_DIR}/config.json \
  --model_path ${CHECKPOINT_DIR}/se2transformer.pth \
  --batch_size 4 \
  --gpu 0

echo ""
echo "=========================================="
echo "完成!"
echo "=========================================="

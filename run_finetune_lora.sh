#!/bin/bash
# SToFM LoRA 微调：与 get_embeddings 使用相同的数据与 checkpoint 路径
# 数据：2_stofm_processed_dlpfc/sample_list.txt
# 输出：lora_finetune_out/（LoRA 权重、train_log、可选 stofm_merged.pt）
# 使用前需已跑过 get_embeddings，各切片目录下存在 ce_emb.npy

set -e
BASE_DIR="/data/xiaox/projects/20250801_aging"
SCRIPT_DIR="${BASE_DIR}/scripts/graduate_design"
STOFM_DIR="${BASE_DIR}/scripts/SToFM/SToFM"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoint/ckpt"
PROCESSED_DIR="${SCRIPT_DIR}/2_stofm_processed_dlpfc"
SAMPLE_LIST="${PROCESSED_DIR}/sample_list.txt"
OUTPUT_DIR="${SCRIPT_DIR}/lora_finetune_out"

export CUDA_VISIBLE_DEVICES=0

if [ ! -f "${CHECKPOINT_DIR}/config.json" ] || [ ! -f "${CHECKPOINT_DIR}/se2transformer.pth" ]; then
    echo "错误: checkpoint 不存在，请确认以下文件存在："
    echo "  ${CHECKPOINT_DIR}/config.json"
    echo "  ${CHECKPOINT_DIR}/se2transformer.pth"
    exit 1
fi
if [ ! -f "${SAMPLE_LIST}" ]; then
    echo "错误: 样本列表不存在: ${SAMPLE_LIST}"
    echo "请先完成预处理并生成 sample_list.txt（或参考 get_embeddings.py 的生成逻辑）"
    exit 1
fi

echo "=========================================="
echo "SToFM LoRA 微调（路径与 get_embeddings 一致）"
echo "=========================================="
echo "  config_path: ${CHECKPOINT_DIR}/config.json"
echo "  model_path:  ${CHECKPOINT_DIR}/se2transformer.pth"
echo "  data_path:   ${SAMPLE_LIST}"
echo "  output_dir:  ${OUTPUT_DIR}"
echo "=========================================="

cd "${STOFM_DIR}"
python finetune_lora.py \
  --config_path   "${CHECKPOINT_DIR}/config.json" \
  --model_path    "${CHECKPOINT_DIR}/se2transformer.pth" \
  --data_path     "${SAMPLE_LIST}" \
  --output_dir    "${OUTPUT_DIR}" \
  --lora_r        8 \
  --epochs        10 \
  --batch_size    4 \
  --save_merged

echo ""
echo "完成. 输出目录: ${OUTPUT_DIR}"
echo "  - lora_weights.pt   LoRA 参数"
echo "  - train_log.txt     训练日志"
echo "  - stofm_merged.pt    合并后的 backbone（可用于 get_embeddings 替换 se2transformer.pth）"

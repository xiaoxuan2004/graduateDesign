#!/bin/bash
# 使用 LoRA 微调后的 backbone（stofm_merged.pt）生成新嵌入文件 stofm_emb_lora.npy
# 输出写入各切片目录 2_stofm_processed_dlpfc/<slide>/stofm_emb_lora.npy（不覆盖原 stofm_emb.npy）
# 运行前需已执行 run_finetune_lora.sh 并得到 lora_finetune_out/stofm_merged.pt

set -e
BASE_DIR="/data/xiaox/projects/20250801_aging"
SCRIPT_DIR="${BASE_DIR}/scripts/graduate_design"
STOFM_DIR="${BASE_DIR}/scripts/SToFM/SToFM"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoint/ckpt"
MERGED_MODEL="${SCRIPT_DIR}/lora_finetune_out/stofm_merged.pt"
SAMPLE_LIST="${SCRIPT_DIR}/2_stofm_processed_dlpfc/sample_list.txt"

export CUDA_VISIBLE_DEVICES=0

if [ ! -f "${MERGED_MODEL}" ]; then
    echo "错误: 未找到微调后的模型，请先运行: bash run_finetune_lora.sh"
    echo "  期望文件: ${MERGED_MODEL}"
    exit 1
fi
if [ ! -f "${SAMPLE_LIST}" ]; then
    echo "错误: 样本列表不存在: ${SAMPLE_LIST}"
    exit 1
fi

echo "=========================================="
echo "使用 LoRA 微调模型生成 SToFM 嵌入"
echo "=========================================="
echo "  model_path: ${MERGED_MODEL}"
echo "  data_path:  ${SAMPLE_LIST}"
echo "  输出: 各切片目录下 stofm_emb_lora.npy（新增，不覆盖原 stofm_emb.npy）"
echo "=========================================="

cd "${STOFM_DIR}"
python get_embeddings.py \
  --cell_encoder_path "${CHECKPOINT_DIR}/cell_encoder" \
  --config_path     "${CHECKPOINT_DIR}/config.json" \
  --model_path      "${MERGED_MODEL}" \
  --data_path       "${SAMPLE_LIST}" \
  --output_filename stofm_emb_lora.npy \
  --batch_size      4 \
  --seed            42

echo ""
echo "完成. 各切片已生成 stofm_emb_lora.npy，在 STAIG 中指定该文件名加载即可（见 LORA_FINETUNE_README 五.1）。"

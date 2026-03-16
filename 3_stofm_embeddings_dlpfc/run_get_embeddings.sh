#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd /data/xiaox/projects/20250801_aging/scripts/SToFM/SToFM && python get_embeddings.py \
    --cell_encoder_path /data/xiaox/projects/20250801_aging/scripts/graduate_design/checkpoint/ckpt/cell_encoder \
    --config_path /data/xiaox/projects/20250801_aging/scripts/graduate_design/checkpoint/ckpt/config.json \
    --model_path /data/xiaox/projects/20250801_aging/scripts/graduate_design/checkpoint/ckpt/se2transformer.pth \
    --data_path /data/xiaox/projects/20250801_aging/scripts/graduate_design/2_stofm_processed_dlpfc/sample_list.txt \
    --output_filename stofm_emb.npy \
    --batch_size 4 \
    --seed 42
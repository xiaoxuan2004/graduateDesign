# -*- coding: utf-8 -*-
"""
SToFM LoRA 微调脚本：在 DLPFC（或其它已预处理为 SToFM 图格式）数据上，
用 LoRA 微调预训练 SToFM，使表征更适配当前任务。

使用方式见 LORA_FINETUNE_README.md。
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.se2transformer import SToFMConfig, SToFMForMaskedLM
from model.extraction import load_data, SToFM_Collator
from model.lora import inject_lora_into_stofm, merge_lora_into_linear


def parse_args():
    p = argparse.ArgumentParser(description="SToFM LoRA fine-tuning on DLPFC")
    p.add_argument("--config_path", type=str, required=True, help="SToFM config.json 路径")
    p.add_argument("--model_path", type=str, required=True, help="预训练 se2transformer.pth 路径")
    p.add_argument("--data_path", type=str, required=True,
                   help="数据路径：单目录、data.h5ad，或包含多目录的 .txt 列表")
    p.add_argument("--emb_path", type=str, default=None,
                   help="cell embedding 路径，如 ce_emb.npy；默认 data_path/ce_emb.npy")
    p.add_argument("--output_dir", type=str, default="./lora_finetune_out",
                   help="保存 LoRA 权重与日志的目录")
    p.add_argument("--lora_r", type=int, default=8, help="LoRA 秩")
    p.add_argument("--lora_alpha", type=float, default=None,
                   help="LoRA 缩放因子，默认等于 lora_r")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--split_num", type=int, default=1000)
    p.add_argument("--leiden_res", type=float, default=1.0)
    p.add_argument("--leiden_alpha", type=float, default=0.2)
    p.add_argument("--mask_rate", type=float, default=0.12)
    p.add_argument("--mask_pair_rate", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_merged", action="store_true",
                   help="训练结束后将 LoRA 合并回 Linear 并保存完整模型")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径：支持单目录、.txt 列表
    if args.data_path.endswith(".txt"):
        data_roots = [line.strip() for line in open(args.data_path) if line.strip()]
    else:
        data_roots = [args.data_path]

    all_graphs = []
    for data_root in data_roots:
        data_root = data_root.rstrip("/")
        data_path = f"{data_root}/data.h5ad"
        emb_path = args.emb_path or f"{data_root}/ce_emb.npy"
        model_input_path = f"{data_root}/hf.dataset"
        if not os.path.exists(data_path):
            print(f"跳过（不存在）: {data_path}")
            continue
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"需要先生成 cell embedding: {emb_path}\n"
                "请先运行 get_embeddings.py 或使用 CellBERT 等得到 ce_emb.npy"
            )
        info = {
            "data_root": data_root,
            "data_path": data_path,
            "emb_path": emb_path,
            "model_input_path": model_input_path,
        }
        graphs = load_data(
            **info,
            new_emb=False,
            device=-1,
            filter=False,
            split_num=args.split_num,
            leiden_res=args.leiden_res,
            alpha=args.leiden_alpha,
        )
        all_graphs.extend(graphs)
        print(f"加载 {data_root}: {len(graphs)} 个子图")

    if not all_graphs:
        raise RuntimeError("没有加载到任何图，请检查 data_path 与 emb_path")

    collator = SToFM_Collator(
        mask=True,
        mask_pair=True,
        mask_rate=args.mask_rate,
        mask_pair_rate=args.mask_pair_rate,
    )
    dataloader = DataLoader(
        all_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    config = SToFMConfig.from_pretrained(args.config_path)
    model = SToFMForMaskedLM(config)
    state = torch.load(args.model_path, map_location="cpu")
    # 预训练权重可能是 SToFMModel 的 state_dict（无 model. 前缀），或带 "model." 前缀
    model_sd = model.model.state_dict()
    state_keys = set(state.keys())
    if not any(k in model_sd for k in state_keys):
        state = {k.replace("model.", ""): v for k, v in state.items() if "model." in k}
    model.model.load_state_dict(state, strict=False)
    model = model.to(device)

    n_lora = inject_lora_into_stofm(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    print(f"已向 SToFM 注入 LoRA，共替换 {n_lora} 个 Linear 层")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数数量: {sum(p.numel() for p in trainable)}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    log_file = os.path.join(args.output_dir, "train_log.txt")
    with open(log_file, "w") as f:
        f.write("epoch\tloss\tpair_loss\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_pair = 0.0
        n_batch = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(
                token_embeddings=batch["token_embeddings"],
                attn_bias=batch["attn_bias"],
                token_types=batch["token_types"],
                labels=batch.get("labels"),
                pair_labels=batch.get("pair_labels"),
                mask_token=-100.0,
                pair_mask_token=-100.0,
            )
            loss = 0.0
            if out.get("loss") is not None:
                loss = loss + out["loss"]
            if out.get("pair_loss") is not None:
                loss = loss + out["pair_loss"]
                total_pair += out["pair_loss"].item()
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
            total_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batch, 1)
        avg_pair = total_pair / max(n_batch, 1)
        with open(log_file, "a") as f:
            f.write(f"{epoch}\t{avg_loss:.6f}\t{avg_pair:.6f}\n")
        print(f"Epoch {epoch} loss={avg_loss:.4f} pair_loss={avg_pair:.4f}")

    # 保存 LoRA 参数（仅 A/B 与配置，便于后续加载）
    lora_state = {}
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = p.detach().cpu()
    torch.save(
        {
            "lora_state_dict": lora_state,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        },
        os.path.join(args.output_dir, "lora_weights.pt"),
    )
    print(f"LoRA 权重已保存到 {args.output_dir}/lora_weights.pt")

    if args.save_merged:
        merge_lora_into_linear(model.model.encoder)
        torch.save(model.model.state_dict(), os.path.join(args.output_dir, "stofm_merged.pt"))
        print(f"已合并 LoRA 并保存完整 backbone 到 {args.output_dir}/stofm_merged.pt")


if __name__ == "__main__":
    main()

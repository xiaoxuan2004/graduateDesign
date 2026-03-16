# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) for SToFM 微调.

LoRA 在冻结预训练权重的条件下，只训练注入的低秩矩阵 A、B，使
  W' = W + B @ A  (A: r×in_features, B: out_features×r, r << min(in, out))
从而用极少参数量适配下游任务（如 DLPFC 空间转录组）。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    对单个 nn.Linear 的 LoRA 包装：output = linear(x) + (x @ A.T) @ B.T。
    A 用 Kaiming 初始化，B 用零初始化，使初始时 LoRA 不改变原输出。
    """

    def __init__(
        self,
        original: nn.Linear,
        r: int = 8,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_features = original.in_features
        out_features = original.out_features
        self.r = min(r, in_features, out_features)
        self.alpha = alpha if alpha is not None else float(r)
        self.scaling = self.alpha / self.r

        self.original = original  # 不训练，仅前向时使用
        for p in self.original.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(self.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.r))
        self.dropout = nn.Dropout(p=dropout)

        self._init_lora()

    def _init_lora(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        # 保证 LoRA 参数与输入同设备（避免 inject 后 to(device) 未正确传播）
        a = self.lora_A.to(x.device)
        b = self.lora_B.to(x.device)
        lora_out = self.dropout(x) @ a.T @ b.T
        return out + self.scaling * lora_out


def _replace_linear_with_lora(
    module: nn.Module,
    name: str,
    r: int,
    alpha: Optional[float],
    dropout: float,
    target_names: Optional[List[str]],
) -> int:
    """递归地将指定名称的 Linear 替换为 LoRALayer，返回替换数量。"""
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Linear):
            if target_names is None or any(t in full_name for t in target_names):
                lora = LoRALayer(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora)
                replaced += 1
        else:
            replaced += _replace_linear_with_lora(
                child, full_name, r, alpha, dropout, target_names
            )
    return replaced


def inject_lora_into_stofm(
    model: nn.Module,
    r: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    target_module_names: Optional[List[str]] = None,
) -> int:
    """
    向 SToFM 的 Transformer 中注入 LoRA。

    Parameters
    ----------
    model : nn.Module
        SToFMModel 或 SToFMForMaskedLM（会对其 .model 或 .encoder 注入）
    r : int
        LoRA 秩，越大可表达能力越强但参数越多，常用 4~16
    alpha : float, optional
        LoRA 缩放因子，实际缩放为 alpha/r；None 时用 r
    dropout : float
        LoRA 分支上的 dropout
    target_module_names : list of str, optional
        要注入的模块名关键词，如 ["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]；
        None 时使用默认列表（encoder 内所有 attention 与 FFN 的 Linear）

    Returns
    -------
    int
        被替换的 Linear 层数量
    """
    if target_module_names is None:
        target_module_names = [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2",
        ]

    # 若为 SToFMForMaskedLM，对其 .model.encoder 注入
    root = model
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        root = model.model.encoder
    elif hasattr(model, "encoder"):
        root = model.encoder

    for p in root.parameters():
        p.requires_grad = False

    n = _replace_linear_with_lora(
        root, "", r, alpha, dropout, target_module_names
    )
    return n


def merge_lora_into_linear(module: nn.Module) -> None:
    """
    将 LoRA 权重合并回原始 Linear（用于导出单模型、推理加速）。
    原地修改：LoRALayer 被替换回 nn.Linear，且 weight = original.weight + scaling * (B @ A)。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALayer):
            orig = child.original
            dev = orig.weight.data.device
            a = child.lora_A.to(dev)
            b = child.lora_B.to(dev)
            merged_w = orig.weight.data + child.scaling * (b @ a)
            merged_linear = nn.Linear(
                orig.in_features, orig.out_features, bias=orig.bias is not None
            )
            merged_linear.weight.data = merged_w.to(dev)
            if orig.bias is not None:
                merged_linear.bias.data = orig.bias.data.to(dev)
            setattr(module, name, merged_linear)
        else:
            merge_lora_into_linear(child)

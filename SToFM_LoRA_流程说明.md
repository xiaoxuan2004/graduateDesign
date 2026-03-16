# SToFM LoRA 微调与嵌入生成流程说明

本文档说明「LoRA 微调 → 用微调模型生成新嵌入 → 在 STAIG 中使用」的完整流程及如何自检是否完成。

---

## 一、流程概览

```
1. run_finetune_lora.sh
   → 用 DLPFC 图数据微调 SToFM（只训 LoRA 参数）
   → 输出: lora_finetune_out/lora_weights.pt, stofm_merged.pt

2. run_get_embeddings_lora.sh
   → 用 stofm_merged.pt 对每个切片前向，得到 256 维细胞嵌入
   → 输出: 2_stofm_processed_dlpfc/<slide>/stofm_emb_lora.npy（不覆盖 stofm_emb.npy）

3. STAIG notebook
   → 指定加载 stofm_emb_lora.npy，放入 data.obsm['stofm_emb']
   → Run B / Run C 即使用微调后的 SToFM 嵌入
```

---

## 二、run_get_embeddings_lora.sh 具体做了什么

脚本会调用 SToFM 仓库里的 `get_embeddings.py`，与原来「冻结模型」提嵌入的流程相同，唯一区别是 **`--model_path` 改为微调后的 backbone**：

| 步骤 | 说明 |
|------|------|
| 1. 读 sample_list.txt | 得到 5 个切片目录（151507, 151508, 151673, 151675, 151676） |
| 2. 对每个切片 | 若没有 ce_emb.npy 则用 Cell encoder 生成；再用 **stofm_merged.pt** 的 SToFM 对图前向 |
| 3. 子图前向 | 每个切片被拆成多个子图（如 151673 为 6 个子图），逐 batch 过 SToFM，取 `last_hidden_state` |
| 4. 写回 | 按 `indices` 把各子图的节点嵌入拼回全切片顺序，保存为 **`<切片目录>/stofm_emb_lora.npy`**（由 `--output_filename stofm_emb_lora.npy` 指定） |

因此：**原 `stofm_emb.npy`（未微调）保留**，微调后的嵌入单独存在 **`stofm_emb_lora.npy`**。

---

## 三、如何检查是否完成

在终端执行：

```bash
# 查看各切片下是否同时存在两个 npy
ls -la /data/xiaox/projects/20250801_aging/scripts/graduate_design/2_stofm_processed_dlpfc/*/stofm_emb*.npy
```

预期类似：

- 每个切片目录下有两个文件：`stofm_emb.npy`（2 月日期，未微调）、`stofm_emb_lora.npy`（本次运行日期，微调）。
- `stofm_emb_lora.npy` 体积与同切片的 `stofm_emb.npy` 相近（细胞数 × 256 × 4 字节）。

再用 Python 快速检查形状与数值：

```python
import numpy as np
path = "/data/xiaox/projects/20250801_aging/scripts/graduate_design/2_stofm_processed_dlpfc/151673/stofm_emb_lora.npy"
x = np.load(path)
print("shape:", x.shape)   # 应为 (3611, 256) 与 151673 细胞数一致
print("dtype:", x.dtype)  # float32
print("min/max:", x.min(), x.max())
```

---

## 四、日志里可忽略的提示

- **anndata FutureWarning**：库的弃用提示，不影响结果。
- **cudf / numba_cuda**：RAPIDS 检测 GPU 时的兼容性警告，未用 RAPIDS 时可忽略。
- **BertModel pooler 未从 checkpoint 加载**：get_embeddings 里用的是自定义 Pooler（加载了 cell_proj.bin），BERT 自带的 pooler 被替换，该提示可忽略。

只要最后打印了「完成. 各切片已生成 stofm_emb_lora.npy」且上述 `ls` 与 Python 检查通过，即说明**已按预期完成**。

---

## 五、在 STAIG 中使用微调嵌入

在 STAIG 的 notebook（如 ARI0.68-Spatial_clustering-151673-img-stofm）中，将加载 SToFM 嵌入的那段改为使用新文件名即可：

```python
stofm_dir = "/data/xiaox/projects/20250801_aging/scripts/graduate_design/2_stofm_processed_dlpfc"
stofm_emb_file = "stofm_emb_lora.npy"   # 使用微调后的嵌入
emb_path = os.path.join(stofm_dir, args.slide, stofm_emb_file)
stofm_emb = np.load(emb_path)
data.obsm['stofm_emb'] = StandardScaler().fit_transform(stofm_emb)
```

如需对比，可把 `stofm_emb_file` 改为 `"stofm_emb.npy"` 即切换回未微调嵌入。

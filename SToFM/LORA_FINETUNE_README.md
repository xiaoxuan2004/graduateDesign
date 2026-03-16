# SToFM LoRA 微调说明

你当前是**直接使用预训练好的 SToFM**，把 DLPFC 的 cell 特征（如 CellBERT 的 `ce_emb.npy`）输入进去得到 `stofm_emb.npy`。若希望 SToFM 更贴合 DLPFC 或当前任务，可以在**不重训整个模型**的前提下，用 **LoRA（Low-Rank Adaptation）** 只微调少量参数，在 DLPFC 数据上做一次轻量微调。

本文档说明：**什么是 LoRA**、**在 SToFM 里怎么用**、**如何跑通 DLPFC 上的 LoRA 微调**。

---

## 一、为什么需要微调？LoRA 是什么？

### 1.1 为什么微调？

- 预训练 SToFM 是在大规模空间转录组数据（如 SToCorpus-88M）上训练的，学到的是**通用**的空间+表达表征。
- 你的数据是 **DLPFC**（或其它特定技术/组织），分布可能和预训练集有差异；直接拿预训练模型推理，有时在**聚类、识别区域**等任务上不是最优。
- **微调**：在 DLPFC 上再训练几步，让模型在“尽量保留预训练知识”的前提下，**适配当前数据**，通常能提升下游 ARI/NMI 等。

### 1.2 全量微调 vs LoRA

- **全量微调**：把所有参数都打开训练。缺点：显存大、容易过拟合、需要较多数据。
- **LoRA（Low-Rank Adaptation）**：  
  - 保持**预训练权重冻结**（不更新）；  
  - 只在部分层（如注意力里的 `q_proj, k_proj, v_proj, out_proj` 和 FFN 的 `fc1, fc2`）上**加一对低秩矩阵 A、B**，只训练 A 和 B。  
  - 前向时：`output = 原线性层(x) + (x @ A.T) @ B.T`，其中 A 形状 `(r, in_features)`，B 形状 `(out_features, r)`，`r` 为秩（如 8），远小于 in/out 维度，所以**参数量很小**。  
  - 优点：显存占用低、训练快、不易忘掉预训练知识，适合你这种“没有微调过、数据量有限”的场景。

### 1.3 在本仓库里 LoRA 加在哪？

- 加在 **SToFM 的 Transformer Encoder** 里：
  - 每个 **MultiheadAttention** 的 `q_proj, k_proj, v_proj, out_proj`（4 个 `nn.Linear`）；
  - 每个 **TransformerEncoderLayer** 里 FFN 的 `fc1, fc2`（2 个 `nn.Linear`）。
- 这些层被替换成“原 Linear + LoRA 分支”，原 Linear 冻结，只训练 LoRA 的 A、B。

---

## 二、整体流程（你需要在做什么）

1. **数据**：已经按 SToFM 的流程预处理好的 DLPFC（每个切片一个目录，里面有 `data.h5ad`、`ce_emb.npy`、`hf.dataset` 等）。
2. **预训练模型**：官方或你已有的 `config.json` + `se2transformer.pth`（SToFM 的 backbone，即 SToFMModel 的权重）。
3. **微调**：运行 `finetune_lora.py`，在 DLPFC 图上做**掩码节点 + 掩码边**的自监督损失（与 SToFM 预训练目标一致），只更新 LoRA 参数。
4. **用微调后的模型提特征**：  
   - 要么用“**原 backbone + 加载 LoRA 权重**”的脚本（见下）重新跑一遍 `get_embeddings` 得到新的 `stofm_emb.npy`；  
   - 要么先 `merge_lora_into_linear` 得到合并后的 backbone，再按原来 `get_embeddings.py` 的方式加载合并后的 `stofm_merged.pt` 做推理。

---

## 三、环境与依赖

- 与当前 SToFM 一致：PyTorch、transformers、scanpy、你已有的 `model/`（含 `se2transformer`、`extraction`、`lora`）等。
- 无需额外安装“LoRA 库”；本仓库在 `model/lora.py` 里实现了轻量 LoRA 层与注入逻辑。

---

## 四、如何跑 LoRA 微调（DLPFC 示例）

### 4.1 准备数据与路径（与 `scripts/graduate_design` 嵌入流程一致）

嵌入流程使用：`graduate_design/get_embeddings.py` 生成 `run_get_embeddings.sh`，用 `2_stofm_processed_dlpfc/sample_list.txt` 和 `checkpoint/ckpt` 跑 SToFM 得到各切片下的 `stofm_emb.npy`。LoRA 微调使用**同一套路径**：

- **预处理数据目录**：`scripts/graduate_design/2_stofm_processed_dlpfc/`
  - 每个切片一个子目录，如 `151507/`、`151673/`，内含：
    - `data.h5ad`
    - `ce_emb.npy`（先跑 get_embeddings 时由 Cell encoder 生成，与冻结嵌入流程一致）
    - `hf.dataset/`
- **样本列表**：`2_stofm_processed_dlpfc/sample_list.txt`，每行一个切片目录的绝对路径。
- **Checkpoint**：`scripts/graduate_design/checkpoint/ckpt/`（`config.json`、`se2transformer.pth`、`cell_encoder/`）。

### 4.2 运行命令

**方式一：在 `graduate_design` 下用封装脚本（推荐，与嵌入流程路径一致）**

```bash
cd /data/xiaox/projects/20250801_aging/scripts/graduate_design
bash run_finetune_lora.sh
```

脚本会调用 SToFM 的 `finetune_lora.py`，并自动使用与 `run_get_embeddings.sh` 相同的 config、model、数据列表和输出目录（见 4.2.1）。

**方式二：在 SToFM 目录下手动指定参数**

在 **SToFM 仓库根目录**（即 `SToFM/` 或包含 `model/` 的那一层）下执行，路径与 `graduate_design` 嵌入流程对齐：

```bash
cd /data/xiaox/projects/20250801_aging/scripts/SToFM/SToFM

export CUDA_VISIBLE_DEVICES=0
python finetune_lora.py \
  --config_path   /data/xiaox/projects/20250801_aging/scripts/graduate_design/checkpoint/ckpt/config.json \
  --model_path   /data/xiaox/projects/20250801_aging/scripts/graduate_design/checkpoint/ckpt/se2transformer.pth \
  --data_path    /data/xiaox/projects/20250801_aging/scripts/graduate_design/2_stofm_processed_dlpfc/sample_list.txt \
  --output_dir   /data/xiaox/projects/20250801_aging/scripts/graduate_design/lora_finetune_out \
  --lora_r       8 \
  --epochs       10 \
  --batch_size   4 \
  --save_merged
```

- 单切片时可将 `--data_path` 改为单个目录，如 `.../2_stofm_processed_dlpfc/151673`。
- 多切片时使用 `sample_list.txt`（与 `run_get_embeddings.sh` 中的 `--data_path` 一致）。

**4.2.1 `run_finetune_lora.sh` 使用的路径（与 `get_embeddings` 一致）**

| 参数 | 路径 |
|------|------|
| `--config_path` | `scripts/graduate_design/checkpoint/ckpt/config.json` |
| `--model_path` | `scripts/graduate_design/checkpoint/ckpt/se2transformer.pth` |
| `--data_path` | `scripts/graduate_design/2_stofm_processed_dlpfc/sample_list.txt` |
| `--output_dir` | `scripts/graduate_design/lora_finetune_out` |

- `--config_path`：SToFM 的 `config.json`（与 `get_embeddings.py` 用同一份即可）。  
- `--model_path`：预训练 backbone 权重 `se2transformer.pth`。  
- `--data_path`：  
  - 单个切片：写该切片目录，如 `.../2_stofm_processed_dlpfc/151673`；  
  - 多个切片：写一个 `.txt` 文件路径，里面每行一个切片目录。  
- `--emb_path`：若不写，默认用 `{data_path}/ce_emb.npy`；若你的 cell 嵌入不在默认路径，用此参数指定。  
- `--output_dir`：输出目录，下面会生成 `lora_weights.pt`、`train_log.txt`，以及可选 `stofm_merged.pt`。  
- `--lora_r`：LoRA 秩，建议 4～16，默认 8。  
- `--lora_alpha`：LoRA 缩放，实际缩放为 `alpha/r`；不写则等于 `lora_r`。  
- `--save_merged`：若加上，训练结束会把 LoRA 合并回 Linear，并保存完整 backbone 为 `stofm_merged.pt`，便于之后直接当“一个普通 SToFM”用。

更多参数（如 `--split_num`、`--leiden_res`、`--mask_rate`、`--mask_pair_rate`）与当前 `load_data` / `SToFM_Collator` 行为一致，可按需调整。

### 4.3 输出说明

- **`lora_weights.pt`**：仅含 LoRA 的 `lora_A`、`lora_B` 及 `lora_r`、`lora_alpha`，方便后续“加载到已加载预训练权重的 SToFM 上”再推理。  
- **`train_log.txt`**：每行 `epoch \t loss \t pair_loss`，用于看收敛。  
- **`stofm_merged.pt`**（加 `--save_merged` 时）：LoRA 已合并进 encoder 的 Linear，即“微调后的 SToFM backbone”，可直接被 `get_embeddings.py` 类逻辑加载使用。

---

## 五、微调后用 LoRA 模型做推理（取 stofm_emb）

两种方式二选一即可。

### 方式 A：使用合并后的 backbone（推荐，最简单）

1. 训练时加上 `--save_merged`，得到 `stofm_merged.pt`。  
   - 若用 `graduate_design/run_finetune_lora.sh`，合并后的文件在 `scripts/graduate_design/lora_finetune_out/stofm_merged.pt`。
2. 在现有 `get_embeddings.py`（或生成 `run_get_embeddings.sh` 时）里，把 `--model_path` 改为指向合并后的权重，例如：

   ```bash
   # 使用微调后的 backbone 重新生成 embedding 时
   --model_path /data/xiaox/projects/20250801_aging/scripts/graduate_design/lora_finetune_out/stofm_merged.pt
   ```

   或在 SToFM 的 `get_embeddings.py` 中加载时使用该路径的 `stofm_merged.pt` 到 `model`（SToFMModel），再对 DLPFC 跑一遍，得到新的 `stofm_emb.npy`。

### 方式 B：保留“预训练 + LoRA”分开的形式

1. 先加载预训练 SToFM（与现在一样）。  
2. 调用 `inject_lora_into_stofm(model, r=..., alpha=...)` 注入 LoRA 结构。  
3. 再加载 `lora_weights.pt` 里的 `lora_state_dict` 到 `model`（只覆盖 LoRA 相关参数）。  
4. 用该 model 做前向，得到 embedding。  

这样无需合并，但推理时每次都要“原层 + LoRA 分支”，略多一点计算；脚本可参考 `finetune_lora.py` 里的注入与保存逻辑自行写一个 `get_embeddings_lora.py`。

---

## 五.1 微调后生成新嵌入并用于 STAIG（graduate_design 流程）

按下面两步即可把 LoRA 微调后的表征用到 STAIG 空间聚类。

**步骤 1：用微调后的 backbone 生成新嵌入文件（不覆盖原文件）**

在 `scripts/graduate_design` 下执行：

```bash
cd /data/xiaox/projects/20250801_aging/scripts/graduate_design
bash run_get_embeddings_lora.sh
```

- 会读取 `lora_finetune_out/stofm_merged.pt` 和 `2_stofm_processed_dlpfc/sample_list.txt`。
- 在每个切片目录 `2_stofm_processed_dlpfc/<slide>/` 下**新增** `stofm_emb_lora.npy`，不覆盖原有的 `stofm_emb.npy`。

**步骤 2：在 STAIG 中加载微调后的嵌入**

STAIG 的 notebook（如 `ARI0.68-Spatial_clustering-151673-img-stofm.ipynb`）里，把加载 SToFM 嵌入的文件名改为 `stofm_emb_lora.npy` 即可使用微调结果，例如：

```python
stofm_dir = "/data/xiaox/projects/20250801_aging/scripts/graduate_design/2_stofm_processed_dlpfc"
stofm_emb_file = "stofm_emb_lora.npy"   # 微调后的嵌入；用 "stofm_emb.npy" 则为未微调
emb_path = os.path.join(stofm_dir, args.slide, stofm_emb_file)
stofm_emb = np.load(emb_path)
data.obsm['stofm_emb'] = StandardScaler().fit_transform(stofm_emb)
```

这样原 `stofm_emb.npy`（未微调）与 `stofm_emb_lora.npy`（微调）并存，通过切换 `stofm_emb_file` 即可对比两种嵌入在 Run B（SToFM veto）、Run C（SNF 伪标签）等中的效果。

---

## 六、LoRA 超参数简要说明

| 参数 | 含义 | 建议 |
|------|------|------|
| `lora_r` | 秩，决定 A/B 的“宽度” | 4～16，数据少可用 4 或 8 |
| `lora_alpha` | 缩放，实际为 alpha/r | 常取等于 r，或略大（如 16）做轻微放大 |
| `lora_dropout` | LoRA 分支上的 dropout | 0.05～0.1 |
| `lr` | 学习率 | 1e-4 左右，若过拟合可再小 |
| `epochs` | 微调轮数 | 先 5～10，看 `train_log.txt` 再调 |

---

## 七、常见问题

1. **没有 `ce_emb.npy`**  
   需要先用 CellBERT（或你当前流程）对 DLPFC 的每个切片生成 cell embedding，保存为 `ce_emb.npy`，再跑 LoRA 微调；与当前“预训练 SToFM + DLPFC 特征”的流程一致。

2. **显存不够**  
   减小 `--batch_size`（如 2）、或减小 `--lora_r`（如 4）、或减小 `--split_num` 等，让每个 batch 的图更小/更少。

3. **loss 不降或震荡**  
   降低学习率（如 5e-5）、适当减少 `epochs`、或略增 `lora_dropout`。

4. **想只对部分层做 LoRA**  
   在 `model/lora.py` 的 `inject_lora_into_stofm` 里，通过 `target_module_names` 只保留需要的模块名（如只留 `["q_proj","v_proj"]`）即可。

---

## 八、小结

- 你当前是：**预训练 SToFM + DLPFC 的 cell 特征 → 直接得到 stofm_emb**。  
- 若要**更好适配 DLPFC**：在相同数据上做 **LoRA 微调**，只训少量低秩矩阵，再用于提取 `stofm_emb`（或合并后当新 backbone 用）。  
- 步骤：准备好 `data.h5ad`、`ce_emb.npy` 等 → 运行 `finetune_lora.py` → 用合并后的模型或“预训练+LoRA”重新跑一遍 embedding → 在下游（如 STAIG 聚类）用新的 `stofm_emb.npy` 即可。

如有报错，请把命令、`train_log.txt` 前几行以及完整报错贴出来，便于排查。

# 快速开始指南

## 完整流程（3步）

### 步骤1: 数据预处理

```bash
cd /data/xiaox/projects/20250801_aging/scripts/graduate_design
python3 preprocess.py
```

这将处理所有10个样本，输出到 `2_stofm_processed/` 目录。

### 步骤2: 提取Embeddings

Checkpoint文件已经解压到 `checkpoint/ckpt/` 目录，可以直接运行：

```bash
python3 get_embeddings.py \
  --batch_size 4 \
  --gpu 0
```

或者直接运行生成的脚本：
```bash
bash 3_stofm_embeddings/run_get_embeddings.sh
```

**注意**: 如果checkpoint路径不同，可以手动指定：
```bash
python3 get_embeddings.py \
  --cell_encoder_path checkpoint/ckpt/cell_encoder \
  --config_path checkpoint/ckpt/config.json \
  --model_path checkpoint/ckpt/se2transformer.pth \
  --batch_size 4 \
  --gpu 0
```

## 输出文件

预处理后，每个样本目录包含：
- `data.h5ad`: 处理后的AnnData对象（包含空间坐标）
- `hf.dataset/`: HuggingFace格式的tokenized数据

提取embeddings后，每个样本目录包含：
- `stofm_emb.npy`: 256维的细胞embeddings (shape: [n_cells, 256])
- `ce_emb.npy`: Cell encoder的embeddings（中间结果）

## 使用Embeddings进行下游分析

```python
import numpy as np
import scanpy as sc

### 快速使用分析脚本（推荐）

```bash
# 使用提供的完整分析脚本
python scripts/graduate_design/downstream_analysis.py
```

### 基本Python使用

```python
import numpy as np
import scanpy as sc

# 加载embeddings和AnnData
embeddings = np.load("2_stofm_processed/C-18M-1/stofm_emb.npy")
adata = sc.read_h5ad("2_stofm_processed/C-18M-1/data.h5ad")

# 添加embeddings到AnnData
adata.obsm['X_stofm'] = embeddings

# 聚类分析
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_stofm')
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_stofm')

# UMAP降维和可视化
sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden_stofm', save='_stofm_clusters.pdf')

# 空间可视化（如果有空间坐标）
if 'spatial' in adata.obsm:
    sc.pl.embedding(adata, basis='spatial', color='leiden_stofm')
```

### 常见下游任务

详细的使用指南和示例代码请参考：
- **📚 完整文档**: `scripts/graduate_design/DOWNSTREAM_TASKS.md`
- **🐍 Python示例**: `scripts/graduate_design/downstream_analysis.py`
- **📊 R示例**: `scripts/graduate_design/downstream_analysis.R`

主要任务包括：
1. **无监督聚类**: 识别细胞亚群（Leiden, KMeans等）
2. **降维可视化**: UMAP, t-SNE, PCA可视化
3. **细胞类型注释**: 基于参考数据集或标记基因的分类
4. **空间模式分析**: 分析细胞在组织中的空间分布
5. **差异分析**: 识别不同条件/组间的差异
6. **批次校正**: 整合多个样本/批次的数据

## 常见问题

**Q: 预处理需要多长时间？**
A: 取决于数据大小，每个样本通常需要几分钟到十几分钟。

**Q: 需要多少GPU内存？**
A: 建议至少8GB GPU内存。如果不足，减小`--batch_size`参数。

**Q: 可以只处理部分样本吗？**
A: 可以，修改`preprocess.py`中的`sample_names`列表即可。

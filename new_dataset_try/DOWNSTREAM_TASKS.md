# SToFM Embeddings 下游任务使用指南

本指南介绍如何使用生成的 `stofm_emb.npy` 文件进行各种下游任务分析。

## 文件格式

- **文件路径**: `2_stofm_processed/{sample_name}/stofm_emb.npy`
- **格式**: NumPy 数组 (`.npy`)
- **形状**: `(n_cells, 256)` - 256维的细胞embeddings
- **数据类型**: `float32`

## 快速开始

### 1. 加载 Embeddings

```python
import numpy as np
import scanpy as sc

# 加载embeddings
embeddings = np.load("2_stofm_processed/C-18M-1/stofm_emb.npy")
print(f"Shape: {embeddings.shape}")  # (n_cells, 256)

# 如果有对应的h5ad文件，可以结合使用
adata = sc.read_h5ad("2_stofm_processed/C-18M-1/data.h5ad")
adata.obsm['X_stofm'] = embeddings  # 将embeddings添加到AnnData
```

### 2. 使用分析脚本

我们提供了一个完整的分析脚本 `downstream_analysis.py`：

```bash
# 激活环境
conda activate stofm

# 运行分析脚本
python scripts/graduate_design/downstream_analysis.py
```

或者自定义使用：

```python
from scripts.graduate_design.downstream_analysis import SToFMEmbeddingAnalyzer

# 创建分析器
analyzer = SToFMEmbeddingAnalyzer(
    emb_path="2_stofm_processed/C-18M-1/stofm_emb.npy",
    data_h5ad_path="2_stofm_processed/C-18M-1/data.h5ad",
    sample_name="C-18M-1"
)

# 进行聚类
cluster_col = analyzer.clustering(method='leiden', resolution=0.5)

# 降维可视化
reduction_key, coords = analyzer.dimensionality_reduction(method='umap')

# 可视化结果
analyzer.visualize_clusters(cluster_col=cluster_col, reduction_key=reduction_key)
```

## 常见下游任务

### 任务1: 无监督聚类分析

**目标**: 识别细胞亚群

```python
import scanpy as sc
import numpy as np

# 加载数据
embeddings = np.load("2_stofm_processed/C-18M-1/stofm_emb.npy")
adata = sc.read_h5ad("2_stofm_processed/C-18M-1/data.h5ad")

# 使用SToFM embeddings
adata.obsm['X_stofm'] = embeddings

# 方法1: 使用scanpy的Leiden聚类
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_stofm')
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_stofm')

# 方法2: 使用KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)
adata.obs['kmeans_stofm'] = clusters.astype(str)
```

### 任务2: 降维可视化

**目标**: 在2D/3D空间中可视化细胞分布

```python
# UMAP可视化（推荐）
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_stofm')
sc.tl.umap(adata, n_components=2)
sc.pl.umap(adata, color='leiden_stofm', save='_stofm_clusters.pdf')

# t-SNE可视化
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_coords = tsne.fit_transform(embeddings)
adata.obsm['X_tsne'] = tsne_coords
sc.pl.embedding(adata, basis='tsne', color='leiden_stofm')
```

### 任务3: 细胞类型注释

**目标**: 基于参考数据集或已知标记基因预测细胞类型

#### 3.1 使用参考数据集（Transfer Learning）

```python
# 加载参考数据集的embeddings和标签
ref_embeddings = np.load("reference_data/stofm_emb.npy")
ref_labels = pd.read_csv("reference_data/cell_types.csv")['cell_type'].values

# 训练分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# 方法1: KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ref_embeddings, ref_labels)
predictions = knn.predict(embeddings)
adata.obs['predicted_cell_type'] = predictions

# 方法2: Random Forest（更准确但更慢）
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(ref_embeddings, ref_labels)
predictions = rf.predict(embeddings)
probabilities = rf.predict_proba(embeddings)  # 获取置信度
adata.obs['predicted_cell_type'] = predictions
adata.obsm['cell_type_probs'] = probabilities
```

#### 3.2 使用标记基因（Marker-based）

```python
# 如果有已知的标记基因，可以结合基因表达数据
marker_genes = {
    'Neuron': ['Tubb3', 'Map2'],
    'Astrocyte': ['Gfap', 'Aqp4'],
    'Microglia': ['Cd68', 'Aif1']
}

# 计算每个细胞类型的得分
for cell_type, markers in marker_genes.items():
    # 使用基因表达数据计算标记基因得分
    adata.obs[f'{cell_type}_score'] = adata[:, markers].X.mean(axis=1)
    
    # 或者使用embeddings + 基因表达的融合方法
    # ...
```

### 任务4: 空间模式分析

**目标**: 分析细胞在组织中的空间分布模式

```python
import matplotlib.pyplot as plt

# 确保有空间坐标
if 'spatial' in adata.obsm:
    spatial_coords = adata.obsm['spatial']
    
    # 可视化聚类在空间中的分布
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                         c=adata.obs['leiden_stofm'].cat.codes,
                         cmap='tab20', s=1, alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Spatial Distribution of Clusters')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')
    plt.savefig('spatial_clusters.png', dpi=300)
    
    # 计算空间自相关（Moran's I）
    from libpysal.weights import Queen
    from esda.moran import Moran
    
    # 构建空间权重矩阵
    w = Queen.from_dataframe(adata.obs[['x_coord', 'y_coord']])
    # 计算空间自相关...
```

### 任务5: 差异分析

**目标**: 识别不同条件/组间的差异

```python
# 使用embeddings进行差异分析
from scipy.spatial.distance import euclidean

# 方法1: 基于embedding的差异
group1_emb = embeddings[adata.obs['group'] == 'Control']
group2_emb = embeddings[adata.obs['group'] == 'Treatment']

# 计算组间差异
group1_mean = group1_emb.mean(axis=0)
group2_mean = group2_emb.mean(axis=0)
difference = np.abs(group1_mean - group2_mean)

# 找出差异最大的维度
top_diff_dims = np.argsort(difference)[-10:]

# 方法2: 结合基因表达的差异分析
# 使用传统的DE分析方法，但可以考虑使用embeddings作为协变量
```

### 任务6: 批次校正与整合

**目标**: 整合多个样本/批次的数据

```python
# 加载多个样本的embeddings
samples = ['C-18M-1', 'C-18M-2', 'C-26M-1']
all_embeddings = []
sample_labels = []

for sample in samples:
    emb = np.load(f"2_stofm_processed/{sample}/stofm_emb.npy")
    all_embeddings.append(emb)
    sample_labels.extend([sample] * len(emb))

all_embeddings = np.vstack(all_embeddings)

# 方法1: 使用Harmony进行批次校正
import harmonypy as hm
ho = hm.run_harmony(all_embeddings, pd.DataFrame({'batch': sample_labels}))
corrected_embeddings = ho.Z_corr.T

# 方法2: 使用scanpy的scvi-tools
# 将embeddings添加到AnnData后使用scanpy的整合方法
```

## 高级应用

### 1. 构建分类/回归头进行预测

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CellTypeClassifier(nn.Module):
    def __init__(self, input_dim=256, n_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

# 准备数据
embeddings = torch.tensor(embeddings, dtype=torch.float32)
labels = torch.tensor(adata.obs['cell_type'].cat.codes, dtype=torch.long)

# 训练模型
model = CellTypeClassifier(n_classes=len(adata.obs['cell_type'].unique()))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环...
```

### 2. 相似性搜索

```python
from sklearn.metrics.pairwise import cosine_similarity

# 找到与特定细胞最相似的细胞
query_cell_idx = 0
query_embedding = embeddings[query_cell_idx:query_cell_idx+1]

similarities = cosine_similarity(query_embedding, embeddings)[0]
most_similar_indices = np.argsort(similarities)[-10:][::-1]

print(f"与细胞 {query_cell_idx} 最相似的10个细胞:")
for idx in most_similar_indices:
    print(f"  细胞 {idx}: 相似度 {similarities[idx]:.3f}")
```

### 3. 异常检测

```python
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

# 使用Isolation Forest检测异常细胞
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(embeddings)

# 标记异常细胞
adata.obs['is_outlier'] = outlier_labels == -1
```

## 与现有工作流整合

### 在Scanpy工作流中使用

```python
import scanpy as sc

# 标准scanpy工作流，但使用SToFM embeddings
adata = sc.read_h5ad("data.h5ad")
embeddings = np.load("stofm_emb.npy")
adata.obsm['X_stofm'] = embeddings

# 使用embeddings替代PCA
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_stofm')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# 后续分析保持不变
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=10)
```

### 在Seurat工作流中使用（通过Python）

```python
# 将embeddings转换为可以导入R/Seurat的格式
import pandas as pd

# 保存为CSV（如果数量不大）
emb_df = pd.DataFrame(embeddings, 
                     columns=[f'SToFM_{i}' for i in range(embeddings.shape[1])],
                     index=adata.obs_names)
emb_df.to_csv('stofm_embeddings.csv')

# 在R中使用:
# embeddings <- read.csv('stofm_embeddings.csv', row.names=1)
# seurat_obj[['stofm']] <- CreateDimReducObject(embeddings = as.matrix(embeddings))
# seurat_obj <- RunUMAP(seurat_obj, reduction='stofm', dims=1:256)
```

## 性能优化建议

1. **内存优化**: 对于大数据集，考虑分批处理或使用内存映射
   ```python
   embeddings = np.load("stofm_emb.npy", mmap_mode='r')  # 只读内存映射
   ```

2. **加速计算**: 使用GPU加速（如果可用）
   ```python
   import cupy as cp
   embeddings_gpu = cp.asarray(embeddings)
   # 在GPU上进行计算...
   ```

3. **并行处理**: 对多个样本进行批量处理
   ```python
   from multiprocessing import Pool
   # 并行处理多个样本...
   ```

## 常见问题

**Q: embeddings可以用于哪些下游任务？**
A: 几乎任何基于特征向量的任务，包括聚类、分类、降维、相似性分析等。

**Q: 如何选择合适的分辨率/聚类数？**
A: 可以使用轮廓系数、elbow method或基于生物学先验知识选择。

**Q: 如何评估embeddings的质量？**
A: 可以通过聚类质量（ARI, NMI）、分类准确率、或与已知标记基因的一致性来评估。

**Q: 可以结合基因表达数据使用吗？**
A: 可以，可以将embeddings作为额外的特征维度，或者使用多模态融合方法。

## 参考文献

- SToFM论文: [arXiv:2507.11588](https://arxiv.org/abs/2507.11588)
- Scanpy文档: https://scanpy.readthedocs.io/
- Scikit-learn文档: https://scikit-learn.org/

## 示例脚本

完整的示例脚本见: `scripts/graduate_design/downstream_analysis.py`

运行示例:
```bash
python scripts/graduate_design/downstream_analysis.py
```

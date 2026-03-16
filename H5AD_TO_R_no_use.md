# 将分析后的 AnnData 用于 R/Seurat 分析

## 保存的 AnnData 文件包含什么？

当运行 `downstream_analysis.py` 并调用 `save_results()` 时，保存的 `{sample_name}_analyzed.h5ad` 文件包含：

### 1. **原始基因表达数据** (`adata.X`)
- ✅ **完整的基因表达矩阵** - 形状: `(n_cells, n_genes)`
- ✅ **可用于R分析** - 这是原始的ST数据，可以直接用于Seurat分析
- ✅ **基因信息** (`adata.var`) - 基因名称、Ensembl ID等

### 2. **空间坐标** (`adata.obsm['spatial']`)
- ✅ **空间位置信息** - 形状: `(n_cells, 2)` 或 `(n_cells, 3)`
- ✅ **可用于空间分析** - 在R中可用于空间可视化、空间聚类等

### 3. **SToFM Embeddings** (`adata.obsm['X_stofm']`)
- ✅ **256维特征向量** - 形状: `(n_cells, 256)`
- ✅ **可用于降维和聚类** - 在R中可以作为额外的特征使用

### 4. **分析结果** (如果已执行)
- ✅ **聚类结果** (`adata.obs['leiden_clusters']` 等)
- ✅ **降维结果** (`adata.obsm['X_umap']`, `adata.obsm['X_tsne']` 等)
- ✅ **PCA结果** (`adata.obsm['X_pca_stofm']` 等)

### 5. **元数据** (`adata.obs`)
- ✅ **细胞注释信息** - 包括 `n_counts`, `filter_pass`, `x_coord`, `y_coord` 等
- ✅ **样本信息** - 如果有的话

## 在 R 中使用保存的 AnnData

### 方法1: 使用 reticulate（推荐）

```r
library(reticulate)
library(Seurat)

# 配置Python环境（如果需要）
# use_python("/path/to/python")
# use_condaenv("stofm")  # 如果使用conda环境

# 加载scanpy
sc <- import("scanpy")

# 读取h5ad文件
h5ad_path <- "3_stofm_embeddings/results/C-18M-1_analyzed.h5ad"
adata <- sc$read_h5ad(h5ad_path)

# 提取数据
# 1. 基因表达矩阵
counts_matrix <- t(py_to_r(adata$X))  # 转置，Seurat需要 (genes x cells)
rownames(counts_matrix) <- py_to_r(adata$var_names)
colnames(counts_matrix) <- py_to_r(adata$obs_names)

# 2. 空间坐标
spatial_coords <- py_to_r(adata$obsm[["spatial"]])
rownames(spatial_coords) <- py_to_r(adata$obs_names)
colnames(spatial_coords) <- c("x", "y")

# 3. SToFM embeddings（可选）
stofm_emb <- py_to_r(adata$obsm[["X_stofm"]])
rownames(stofm_emb) <- py_to_r(adata$obs_names)

# 4. 元数据
metadata <- py_to_r(adata$obs)

# 创建Seurat对象
seurat_obj <- CreateSeuratObject(
  counts = counts_matrix,
  meta.data = metadata
)

# 添加空间坐标
seurat_obj[["spatial"]] <- CreateDimReducObject(
  embeddings = spatial_coords,
  key = "spatial_",
  assay = DefaultAssay(seurat_obj)
)

# 添加SToFM embeddings（可选）
seurat_obj[["stofm"]] <- CreateDimReducObject(
  embeddings = t(stofm_emb),  # Seurat需要 (features x cells)
  key = "SToFM_",
  assay = DefaultAssay(seurat_obj)
)

# 现在可以使用Seurat进行标准分析
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- FindVariableFeatures(seurat_obj)
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj)
seurat_obj <- RunUMAP(seurat_obj, dims = 1:30)
seurat_obj <- FindNeighbors(seurat_obj)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)

# 空间可视化
SpatialFeaturePlot(seurat_obj, features = "nFeature_RNA")
```

### 方法2: 转换为其他格式（如果reticulate不可用）

在Python中先转换：

```python
import scanpy as sc
import pandas as pd
import numpy as np

# 读取h5ad
adata = sc.read_h5ad("3_stofm_embeddings/results/C-18M-1_analyzed.h5ad")

# 1. 保存表达矩阵为CSV或MTX格式
# 方法A: CSV（适合小数据集）
pd.DataFrame(
    adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
    index=adata.obs_names,
    columns=adata.var_names
).to_csv("expression_matrix.csv")

# 方法B: MTX格式（适合大数据集，Seurat可以直接读取）
from scipy import sparse
from scipy.io import mmwrite
mmwrite("matrix.mtx", adata.X)
pd.Series(adata.var_names).to_csv("genes.tsv", index=False, header=False)
pd.Series(adata.obs_names).to_csv("barcodes.tsv", index=False, header=False)

# 2. 保存空间坐标
pd.DataFrame(
    adata.obsm['spatial'],
    index=adata.obs_names,
    columns=['x', 'y']
).to_csv("spatial_coords.csv")

# 3. 保存SToFM embeddings
pd.DataFrame(
    adata.obsm['X_stofm'],
    index=adata.obs_names,
    columns=[f'SToFM_{i}' for i in range(adata.obsm['X_stofm'].shape[1])]
).to_csv("stofm_embeddings.csv")

# 4. 保存元数据
adata.obs.to_csv("metadata.csv")
```

然后在R中加载：

```r
library(Seurat)

# 读取表达矩阵（MTX格式）
seurat_obj <- Read10X(data.dir = ".")
seurat_obj <- CreateSeuratObject(counts = seurat_obj)

# 或读取CSV格式
# counts <- read.csv("expression_matrix.csv", row.names = 1)
# seurat_obj <- CreateSeuratObject(counts = counts)

# 读取空间坐标
spatial_coords <- read.csv("spatial_coords.csv", row.names = 1)
seurat_obj[["spatial"]] <- CreateDimReducObject(
  embeddings = as.matrix(spatial_coords),
  key = "spatial_",
  assay = DefaultAssay(seurat_obj)
)

# 读取SToFM embeddings
stofm_emb <- read.csv("stofm_embeddings.csv", row.names = 1)
seurat_obj[["stofm"]] <- CreateDimReducObject(
  embeddings = t(as.matrix(stofm_emb)),
  key = "SToFM_",
  assay = DefaultAssay(seurat_obj)
)

# 读取元数据
metadata <- read.csv("metadata.csv", row.names = 1)
seurat_obj <- AddMetaData(seurat_obj, metadata = metadata)
```

### 方法3: 使用 zellkonverter 包（Bioconductor）

```r
library(zellkonverter)
library(SingleCellExperiment)

# 直接读取h5ad文件
sce <- readH5AD("3_stofm_embeddings/results/C-18M-1_analyzed.h5ad")

# 转换为Seurat对象
seurat_obj <- as.Seurat(sce, data = NULL)

# 提取空间坐标
spatial_coords <- reducedDim(sce, "spatial")
seurat_obj[["spatial"]] <- CreateDimReducObject(
  embeddings = t(spatial_coords),
  key = "spatial_",
  assay = DefaultAssay(seurat_obj)
)
```

## 注意事项

1. **数据完整性**: 保存的AnnData包含完整的原始ST数据，可以完全用于R分析
2. **空间坐标**: 确保空间坐标在 `adata.obsm['spatial']` 中，这是标准位置
3. **基因表达**: `adata.X` 是原始表达矩阵，可能包含稀疏矩阵格式
4. **SToFM embeddings**: 作为额外特征，可以用于增强分析，但不是必需的
5. **细胞顺序**: 确保在转换时保持细胞顺序一致

## 验证数据完整性

在Python中检查：

```python
import scanpy as sc

adata = sc.read_h5ad("3_stofm_embeddings/results/C-18M-1_analyzed.h5ad")

print("数据形状:", adata.shape)
print("基因表达矩阵:", adata.X.shape if adata.X is not None else "None")
print("空间坐标:", adata.obsm['spatial'].shape if 'spatial' in adata.obsm else "None")
print("SToFM embeddings:", adata.obsm['X_stofm'].shape if 'X_stofm' in adata.obsm else "None")
print("元数据列:", list(adata.obs.columns))
```

## 总结

✅ **是的，保存的AnnData就是SToFM处理后的完整数据**
- 包含原始基因表达数据
- 包含空间坐标
- 包含SToFM embeddings
- 包含分析结果（如果已执行）

✅ **可以用于R/Seurat分析**
- 使用reticulate直接读取
- 或转换为其他格式后读取
- 所有标准ST分析功能都可用

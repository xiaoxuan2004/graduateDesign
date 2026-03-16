# SToFM Embeddings 下游任务分析示例 (R版本)
# 演示如何在R/Seurat中使用 stofm_emb.npy 进行下游任务

library(Seurat)
library(Matrix)
library(reticulate)  # 用于加载Python的numpy文件
library(ggplot2)
library(dplyr)

# ==============================================================================
# 1. 加载 Embeddings
# ==============================================================================

load_stofm_embeddings <- function(emb_path) {
  # 方法1: 使用reticulate直接加载
  if (require(reticulate)) {
    np <- import("numpy")
    embeddings <- np$load(emb_path)
    return(embeddings)
  } else {
    # 方法2: 转换为CSV后加载（需要在Python中先转换）
    stop("请安装reticulate包，或使用Python将.npy转换为CSV格式")
  }
}

# 使用示例
emb_path <- "/data/xiaox/projects/20250801_aging/2_stofm_processed/C-18M-1/stofm_emb.npy"
embeddings <- load_stofm_embeddings(emb_path)

# ==============================================================================
# 2. 整合到 Seurat 对象
# ==============================================================================

# 加载Seurat对象
seurat_obj <- readRDS("path/to/seurat_object.rds")  # 替换为实际路径

# 确保embeddings的细胞顺序与Seurat对象一致
if (nrow(embeddings) == ncol(seurat_obj)) {
  # 将embeddings添加为DimReduc对象
  embeddings <- t(embeddings)  # Seurat需要 (features x cells)
  colnames(embeddings) <- colnames(seurat_obj)
  rownames(embeddings) <- paste0("SToFM_", 1:nrow(embeddings))
  
  # 创建DimReduc对象
  seurat_obj[["stofm"]] <- CreateDimReducObject(
    embeddings = embeddings,
    key = "SToFM_",
    assay = DefaultAssay(seurat_obj)
  )
  
  # 或者添加为metadata（用于其他用途）
  # seurat_obj@meta.data <- cbind(seurat_obj@meta.data, 
  #                                as.data.frame(t(embeddings)))
} else {
  warning("Embeddings的细胞数量与Seurat对象不匹配")
}

# ==============================================================================
# 3. 基于 Embeddings 的降维和可视化
# ==============================================================================

# 使用SToFM embeddings进行UMAP
seurat_obj <- RunUMAP(
  seurat_obj,
  reduction = "stofm",
  dims = 1:nrow(embeddings),
  reduction.name = "umap.stofm",
  reduction.key = "UMAPSToFM_"
)

# 可视化
p1 <- DimPlot(seurat_obj, reduction = "umap.stofm", group.by = "orig.ident")
print(p1)

# ==============================================================================
# 4. 聚类分析
# ==============================================================================

# 基于SToFM embeddings构建邻居图
seurat_obj <- FindNeighbors(
  seurat_obj,
  reduction = "stofm",
  dims = 1:nrow(embeddings),
  graph.name = "stofm_nn"
)

# Leiden聚类
seurat_obj <- FindClusters(
  seurat_obj,
  graph.name = "stofm_nn",
  resolution = 0.5,
  algorithm = 4,  # Leiden algorithm
  method = "igraph"
)

# 可视化聚类结果
p2 <- DimPlot(seurat_obj, reduction = "umap.stofm", 
              label = TRUE, label.size = 4)
print(p2)

# ==============================================================================
# 5. 空间可视化
# ==============================================================================

# 如果有空间坐标信息
if ("spatial" %in% names(seurat_obj@images)) {
  # 空间可视化
  p3 <- SpatialFeaturePlot(
    seurat_obj,
    features = "seurat_clusters",
    images = "spatial",
    pt.size.factor = 1.6
  )
  print(p3)
  
  # 或者使用自定义的空间坐标
  if (exists("spatial_coords")) {
    # 假设spatial_coords是从h5ad文件中提取的空间坐标
    coords_df <- data.frame(
      x = spatial_coords[, 1],
      y = spatial_coords[, 2],
      cluster = seurat_obj$seurat_clusters
    )
    
    ggplot(coords_df, aes(x = x, y = y, color = cluster)) +
      geom_point(size = 0.5, alpha = 0.6) +
      coord_fixed() +
      theme_minimal() +
      labs(title = "Spatial Distribution of Clusters")
  }
}

# ==============================================================================
# 6. 细胞类型注释（使用参考数据集）
# ==============================================================================

# 方法1: 使用Seurat的FindTransferAnchors（需要参考数据集）
# reference_seurat <- readRDS("reference_seurat.rds")
# anchors <- FindTransferAnchors(
#   reference = reference_seurat,
#   query = seurat_obj,
#   normalization.method = "SCT",
#   reference.reduction = "pca",
#   dims = 1:50
# )
# seurat_obj <- MapQuery(
#   anchorset = anchors,
#   query = seurat_obj,
#   reference = reference_seurat,
#   refdata = list(celltype = "cell_type")
# )

# 方法2: 使用SToFM embeddings进行KNN分类
if (require(class)) {
  # 假设有参考数据集的embeddings和标签
  # ref_embeddings <- load_stofm_embeddings("reference/stofm_emb.npy")
  # ref_labels <- read.csv("reference/cell_types.csv")$cell_type
  
  # 转换embeddings格式
  query_emb <- t(embeddings)
  
  # KNN分类
  # predictions <- knn(
  #   train = ref_embeddings,
  #   test = query_emb,
  #   cl = ref_labels,
  #   k = 5
  # )
  # seurat_obj$predicted_cell_type <- predictions
}

# ==============================================================================
# 7. 差异表达分析（结合基因表达和embeddings）
# ==============================================================================

# 基于SToFM聚类进行差异表达分析
DefaultAssay(seurat_obj) <- "RNA"  # 或 "SCT"
markers <- FindAllMarkers(
  seurat_obj,
  only.pos = TRUE,
  min.pct = 0.25,
  logfc.threshold = 0.25
)

# 查看top标记基因
top_markers <- markers %>%
  group_by(cluster) %>%
  slice_max(n = 10, order_by = avg_log2FC)

# ==============================================================================
# 8. 整合多个样本的Embeddings
# ==============================================================================

integrate_multiple_samples <- function(sample_paths, sample_names) {
  # 加载所有样本的embeddings
  all_embeddings <- list()
  for (i in seq_along(sample_paths)) {
    emb <- load_stofm_embeddings(sample_paths[i])
    all_embeddings[[sample_names[i]]] <- emb
  }
  
  # 合并（假设所有embeddings维度相同）
  combined_emb <- do.call(rbind, all_embeddings)
  
  # 批次校正（可选，使用Harmony或其他方法）
  if (require(harmony)) {
    # 创建批次标签
    batch_labels <- rep(sample_names, sapply(all_embeddings, nrow))
    
    # Harmony校正
    # corrected_emb <- HarmonyMatrix(
    #   data_mat = t(combined_emb),
    #   meta_data = data.frame(batch = batch_labels),
    #   vars_use = "batch"
    # )
  }
  
  return(combined_emb)
}

# 使用示例
# sample_paths <- c(
#   "2_stofm_processed/C-18M-1/stofm_emb.npy",
#   "2_stofm_processed/C-18M-2/stofm_emb.npy"
# )
# sample_names <- c("C-18M-1", "C-18M-2")
# integrated_emb <- integrate_multiple_samples(sample_paths, sample_names)

# ==============================================================================
# 9. 辅助函数：从h5ad文件提取信息
# ==============================================================================

# 如果需要从h5ad文件提取空间坐标等信息
if (require(reticulate)) {
  extract_spatial_from_h5ad <- function(h5ad_path) {
    sc <- import("scanpy")
    adata <- sc$read_h5ad(h5ad_path)
    
    if ("spatial" %in% names(adata$obsm)) {
      spatial_coords <- adata$obsm[["spatial"]]
      return(spatial_coords)
    } else {
      warning("未找到空间坐标信息")
      return(NULL)
    }
  }
  
  # 使用示例
  # h5ad_path <- "2_stofm_processed/C-18M-1/data.h5ad"
  # spatial_coords <- extract_spatial_from_h5ad(h5ad_path)
}

# ==============================================================================
# 10. 保存结果
# ==============================================================================

# 保存包含SToFM embeddings的Seurat对象
saveRDS(seurat_obj, "seurat_with_stofm_embeddings.rds")

# 保存聚类结果
cluster_results <- data.frame(
  cell_id = colnames(seurat_obj),
  cluster = seurat_obj$seurat_clusters,
  umap_1 = Embeddings(seurat_obj, "umap.stofm")[, 1],
  umap_2 = Embeddings(seurat_obj, "umap.stofm")[, 2]
)
write.csv(cluster_results, "stofm_clustering_results.csv", row.names = FALSE)

# ==============================================================================
# 完整的分析流程示例
# ==============================================================================

complete_analysis_pipeline <- function(sample_name, emb_path, h5ad_path = NULL) {
  cat("开始分析样本:", sample_name, "\n")
  
  # 1. 加载embeddings
  cat("1. 加载embeddings...\n")
  embeddings <- load_stofm_embeddings(emb_path)
  
  # 2. 加载或创建Seurat对象
  if (!is.null(h5ad_path)) {
    cat("2. 从h5ad加载数据...\n")
    # 使用reticulate从h5ad加载，或使用其他方法
    # seurat_obj <- convert_from_h5ad(h5ad_path)
  } else {
    cat("2. 创建Seurat对象...\n")
    # 创建最小的Seurat对象
    # seurat_obj <- CreateSeuratObject(counts = matrix(0, nrow = 1, ncol = nrow(embeddings)))
  }
  
  # 3. 添加embeddings
  cat("3. 添加embeddings...\n")
  # ... (参考上面的代码)
  
  # 4. 降维和聚类
  cat("4. 降维和聚类...\n")
  # ... (参考上面的代码)
  
  # 5. 可视化
  cat("5. 生成可视化...\n")
  # ... (参考上面的代码)
  
  cat("分析完成!\n")
  return(seurat_obj)
}

# 使用示例
# result <- complete_analysis_pipeline(
#   sample_name = "C-18M-1",
#   emb_path = "2_stofm_processed/C-18M-1/stofm_emb.npy",
#   h5ad_path = "2_stofm_processed/C-18M-1/data.h5ad"
# )

# ==============================================================================
# 注意事项
# ==============================================================================

# 1. 确保reticulate配置正确，能够访问Python环境
#    reticulate::use_python("/path/to/python")
#    reticulate::py_config()

# 2. 如果无法使用reticulate，可以在Python中将.npy转换为CSV：
#    import numpy as np
#    import pandas as pd
#    emb = np.load("stofm_emb.npy")
#    pd.DataFrame(emb).to_csv("stofm_emb.csv", index=False)

# 3. 对于大数据集，考虑使用内存映射或分批处理

# 4. 确保embeddings的细胞顺序与Seurat对象一致

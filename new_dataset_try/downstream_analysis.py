"""
SToFM Embeddings 下游任务分析示例
演示如何使用 stofm_emb.npy 进行各种下游任务
"""
import numpy as np
import scanpy as sc
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置scanpy参数
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

class SToFMEmbeddingAnalyzer:
    """SToFM Embeddings 分析器"""
    
    def __init__(self, emb_path, data_h5ad_path=None, sample_name=None):
        """
        初始化分析器
        
        Parameters
        ----------
        emb_path : str or Path
            stofm_emb.npy 文件路径
        data_h5ad_path : str or Path, optional
            对应的 data.h5ad 文件路径（用于获取元数据和空间坐标）
        sample_name : str, optional
            样本名称
        """
        self.emb_path = Path(emb_path)
        self.data_h5ad_path = Path(data_h5ad_path) if data_h5ad_path else None
        self.sample_name = sample_name or self.emb_path.parent.name
        
        # 加载embeddings
        print(f"加载 embeddings: {self.emb_path}")
        self.embeddings = np.load(self.emb_path)
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  细胞数量: {self.embeddings.shape[0]}")
        print(f"  特征维度: {self.embeddings.shape[1]}")
        
        # 加载AnnData（如果提供）
        if self.data_h5ad_path and self.data_h5ad_path.exists():
            print(f"\n加载 AnnData: {self.data_h5ad_path}")
            self.adata = sc.read_h5ad(self.data_h5ad_path)
            print(f"  AnnData shape: {self.adata.shape}")
            
            # 确保embeddings和细胞数量匹配
            if self.embeddings.shape[0] != self.adata.n_obs:
                print(f"  警告: Embeddings数量({self.embeddings.shape[0]})与AnnData细胞数量({self.adata.n_obs})不匹配")
                # 截取或对齐
                min_cells = min(self.embeddings.shape[0], self.adata.n_obs)
                self.embeddings = self.embeddings[:min_cells]
                self.adata = self.adata[:min_cells].copy()
                print(f"  已对齐到 {min_cells} 个细胞")
            
            # 将embeddings添加到AnnData
            self.adata.obsm['X_stofm'] = self.embeddings
        else:
            self.adata = None
            print("  未提供AnnData文件，将仅使用embeddings进行分析")
    
    def clustering(self, n_clusters=None, method='leiden', resolution=0.5, 
                   use_embeddings=True, **kwargs):
        """
        使用embeddings进行聚类
        
        Parameters
        ----------
        n_clusters : int, optional
            KMeans的聚类数量（当method='kmeans'时）
        method : str
            聚类方法: 'leiden', 'kmeans', 'louvain'
        resolution : float
            Leiden/Louvain聚类的分辨率
        use_embeddings : bool
            是否使用SToFM embeddings（否则使用PCA降维后的特征）
        """
        print(f"\n进行聚类分析 (方法: {method})...")
        
        if self.adata is None:
            # 如果没有AnnData，创建临时对象
            self.adata = sc.AnnData(X=self.embeddings)
        
        # 使用embeddings或降维后的特征
        if use_embeddings:
            # 直接使用SToFM embeddings
            self.adata.obsm['X_stofm'] = self.embeddings
            
            # 方法1: 直接从embeddings构建邻居图（推荐）
            # 创建一个临时的AnnData对象用于计算PCA（scanpy的neighbors需要PCA）
            temp_adata = sc.AnnData(X=self.embeddings)
            sc.tl.pca(temp_adata, n_comps=min(50, self.embeddings.shape[1]-1), use_highly_variable=False)
            # 将PCA结果复制到主对象
            self.adata.obsm['X_pca_stofm'] = temp_adata.obsm['X_pca']
            # 使用SToFM embeddings的PCA构建邻居图
            sc.pp.neighbors(self.adata, n_neighbors=10, n_pcs=min(50, self.embeddings.shape[1]-1), 
                          use_rep='X_pca_stofm')
        else:
            # 使用原始数据的PCA降维
            if 'X_pca' not in self.adata.obsm:
                sc.tl.pca(self.adata, n_comps=50, use_highly_variable=True)
            # 构建邻居图
            sc.pp.neighbors(self.adata, n_neighbors=10, n_pcs=50, use_rep='X_pca')
        
        if method == 'kmeans':
            if n_clusters is None:
                # 自动确定聚类数（使用elbow method或固定值）
                n_clusters = 10
            print(f"  KMeans聚类 (k={n_clusters})...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            self.adata.obs['kmeans_clusters'] = pd.Categorical(cluster_labels.astype(str))
            cluster_col = 'kmeans_clusters'
        
        elif method == 'leiden':
            print(f"  Leiden聚类 (resolution={resolution})...")
            sc.tl.leiden(self.adata, resolution=resolution, key_added='leiden_clusters')
            cluster_col = 'leiden_clusters'
        
        elif method == 'louvain':
            print(f"  Louvain聚类 (resolution={resolution})...")
            sc.tl.louvain(self.adata, resolution=resolution, key_added='louvain_clusters')
            cluster_col = 'louvain_clusters'
        
        else:
            raise ValueError(f"未知的聚类方法: {method}")
        
        n_clusters_found = len(self.adata.obs[cluster_col].unique())
        print(f"  发现 {n_clusters_found} 个聚类")
        print(f"  聚类标签已保存到: adata.obs['{cluster_col}']")
        
        return cluster_col
    
    def dimensionality_reduction(self, method='umap', n_components=2, 
                                 use_embeddings=True, **kwargs):
        """
        降维可视化
        
        Parameters
        ----------
        method : str
            降维方法: 'umap', 'tsne', 'pca'
        n_components : int
            降维后的维度数
        use_embeddings : bool
            是否使用SToFM embeddings作为输入
        """
        print(f"\n进行降维可视化 (方法: {method}, 维度: {n_components})...")
        
        if self.adata is None:
            self.adata = sc.AnnData(X=self.embeddings)
        
        if use_embeddings:
            # 确保embeddings已添加到obsm
            self.adata.obsm['X_stofm'] = self.embeddings
            input_data = self.embeddings
            
            # 为embeddings计算PCA（用于UMAP的邻居图构建）
            if 'X_pca_stofm' not in self.adata.obsm:
                temp_adata = sc.AnnData(X=self.embeddings)
                sc.tl.pca(temp_adata, n_comps=min(50, self.embeddings.shape[1]-1), use_highly_variable=False)
                self.adata.obsm['X_pca_stofm'] = temp_adata.obsm['X_pca']
            
            rep_key = 'X_pca_stofm'  # 使用PCA结果构建邻居图
        else:
            # 使用原始数据的PCA
            if 'X_pca' not in self.adata.obsm:
                sc.tl.pca(self.adata, n_comps=50, use_highly_variable=True)
            input_data = self.adata.obsm['X_pca']
            rep_key = 'X_pca'
        
        if method == 'umap':
            # 需要先构建邻居图
            if 'neighbors' not in self.adata.uns:
                sc.pp.neighbors(self.adata, n_neighbors=10, n_pcs=min(50, input_data.shape[1]), use_rep=rep_key)
            sc.tl.umap(self.adata, n_components=n_components)
            coords = self.adata.obsm['X_umap']
            key = 'X_umap'
        
        elif method == 'tsne':
            print("  计算t-SNE（可能需要一些时间）...")
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, n_iter=1000)
            coords = tsne.fit_transform(input_data)
            key = 'X_tsne'
            self.adata.obsm[key] = coords
        
        elif method == 'pca':
            if use_embeddings:
                pca = PCA(n_components=n_components, random_state=42)
                coords = pca.fit_transform(input_data)
                key = 'X_pca_vis'
                self.adata.obsm[key] = coords
            else:
                sc.tl.pca(self.adata, n_comps=n_components)
                coords = self.adata.obsm['X_pca']
                key = 'X_pca'
        
        else:
            raise ValueError(f"未知的降维方法: {method}")
        
        print(f"  降维完成，结果保存在: adata.obsm['{key}']")
        return key, coords
    
    def visualize_clusters(self, cluster_col, reduction_key='X_umap', 
                          spatial=True, figsize=(12, 5), save_path=None):
        """
        可视化聚类结果
        
        Parameters
        ----------
        cluster_col : str
            聚类列名（在adata.obs中）
        reduction_key : str
            降维结果key（在adata.obsm中）
        spatial : bool
            如果可能，是否显示空间坐标图
        figsize : tuple
            图片大小
        save_path : str, optional
            保存路径
        """
        if self.adata is None:
            print("需要AnnData对象才能可视化")
            return
        
        # 确保有降维结果
        if reduction_key not in self.adata.obsm:
            print(f"未找到降维结果 '{reduction_key}'，先进行降维...")
            reduction_key, _ = self.dimensionality_reduction(method='umap')
        
        n_plots = 2 if spatial and 'spatial' in self.adata.obsm else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        # UMAP/t-SNE图
        coords = self.adata.obsm[reduction_key]
        scatter = axes[0].scatter(coords[:, 0], coords[:, 1], 
                                 c=pd.Categorical(self.adata.obs[cluster_col]).codes,
                                 cmap='tab20', s=1, alpha=0.6)
        axes[0].set_title(f'{self.sample_name} - {cluster_col}')
        axes[0].set_xlabel(f'{reduction_key} 1')
        axes[0].set_ylabel(f'{reduction_key} 2')
        plt.colorbar(scatter, ax=axes[0])
        
        # 空间坐标图
        if spatial and 'spatial' in self.adata.obsm:
            spatial_coords = self.adata.obsm['spatial']
            scatter2 = axes[1].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                                      c=pd.Categorical(self.adata.obs[cluster_col]).codes,
                                      cmap='tab20', s=1, alpha=0.6)
            axes[1].set_title(f'{self.sample_name} - Spatial')
            axes[1].set_xlabel('X coordinate')
            axes[1].set_ylabel('Y coordinate')
            axes[1].set_aspect('equal')
            plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  图片已保存: {save_path}")
        
        plt.show()
    
    def cell_type_annotation(self, reference_embeddings=None, reference_labels=None,
                           method='knn', n_neighbors=5, train_size=0.7):
        """
        细胞类型注释（如果有参考数据）
        
        Parameters
        ----------
        reference_embeddings : array-like, optional
            参考数据集的embeddings
        reference_labels : array-like, optional
            参考数据集的细胞类型标签
        method : str
            分类方法: 'knn', 'rf' (RandomForest), 'svm'
        n_neighbors : int
            KNN的邻居数
        train_size : float
            训练集比例（如果有labeled数据）
        """
        print(f"\n进行细胞类型注释 (方法: {method})...")
        
        if reference_embeddings is None or reference_labels is None:
            # 如果有AnnData且包含细胞类型信息
            if self.adata is not None and 'cell_type' in self.adata.obs:
                print("  使用AnnData中的细胞类型信息...")
                # 使用部分数据作为训练集，剩余作为测试集
                train_idx, test_idx = train_test_split(
                    np.arange(len(self.embeddings)), 
                    train_size=train_size, 
                    random_state=42,
                    stratify=self.adata.obs['cell_type'] if 'cell_type' in self.adata.obs else None
                )
                
                X_train = self.embeddings[train_idx]
                y_train = self.adata.obs['cell_type'].values[train_idx]
                X_test = self.embeddings[test_idx]
                y_test = self.adata.obs['cell_type'].values[test_idx] if 'cell_type' in self.adata.obs else None
            else:
                print("  警告: 没有提供参考数据，无法进行注释")
                return None
        else:
            X_train = reference_embeddings
            y_train = reference_labels
            X_test = self.embeddings
            y_test = None
        
        # 训练分类器
        if method == 'knn':
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif method == 'rf':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif method == 'svm':
            classifier = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            raise ValueError(f"未知的分类方法: {method}")
        
        print(f"  训练{classifier.__class__.__name__}分类器...")
        classifier.fit(X_train, y_train)
        
        # 预测
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test) if hasattr(classifier, 'predict_proba') else None
        
        # 保存结果
        if self.adata is not None:
            if y_test is not None:
                self.adata.obs.loc[test_idx, 'predicted_cell_type'] = predictions
            else:
                self.adata.obs['predicted_cell_type'] = predictions
            print(f"  预测结果已保存到: adata.obs['predicted_cell_type']")
        
        # 评估（如果有ground truth）
        if y_test is not None:
            accuracy = (predictions == y_test).mean()
            ari = adjusted_rand_score(y_test, predictions)
            nmi = normalized_mutual_info_score(y_test, predictions)
            print(f"  准确率: {accuracy:.3f}")
            print(f"  ARI: {ari:.3f}")
            print(f"  NMI: {nmi:.3f}")
        
        return predictions, probabilities
    
    def save_results(self, output_dir=None, prefix=None):
        """
        保存分析结果
        
        Parameters
        ----------
        output_dir : str or Path, optional
            输出目录
        prefix : str, optional
            文件前缀
        """
        if output_dir is None:
            output_dir = Path(self.emb_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        prefix = prefix or self.sample_name
        
        # 保存AnnData（如果存在）
        if self.adata is not None:
            output_path = output_dir / f"{prefix}_analyzed.h5ad"
            
            # 确保所有重要信息都被保存
            # 这个AnnData包含：
            # 1. 原始基因表达数据 (adata.X) - 可用于R分析
            # 2. 空间坐标 (adata.obsm['spatial']) - 可用于空间分析
            # 3. SToFM embeddings (adata.obsm['X_stofm']) - 256维特征
            # 4. 分析结果 (聚类、降维等) - 如果已执行分析
            # 5. 其他元数据 (adata.obs) - 细胞注释信息
            
            self.adata.write(output_path)
            print(f"AnnData已保存: {output_path}")
            print(f"  包含内容:")
            print(f"    - 基因表达数据: {self.adata.shape[0]} 细胞 × {self.adata.shape[1]} 基因")
            if 'spatial' in self.adata.obsm:
                print(f"    - 空间坐标: {self.adata.obsm['spatial'].shape}")
            if 'X_stofm' in self.adata.obsm:
                print(f"    - SToFM embeddings: {self.adata.obsm['X_stofm'].shape}")
            if any(col.startswith(('leiden', 'louvain', 'kmeans')) for col in self.adata.obs.columns):
                cluster_cols = [col for col in self.adata.obs.columns 
                              if col.startswith(('leiden', 'louvain', 'kmeans'))]
                print(f"    - 聚类结果: {', '.join(cluster_cols)}")
            print(f"  可用于R/Seurat分析（使用reticulate或转换为其他格式）")
        
        # 保存聚类结果（如果有）
        if self.adata is not None and any(col.startswith(('leiden', 'louvain', 'kmeans')) 
                                         for col in self.adata.obs.columns):
            cluster_cols = [col for col in self.adata.obs.columns 
                          if col.startswith(('leiden', 'louvain', 'kmeans'))]
            cluster_df = self.adata.obs[cluster_cols]
            output_path = output_dir / f"{prefix}_clusters.csv"
            cluster_df.to_csv(output_path)
            print(f"聚类结果已保存: {output_path}")


def main():
    """示例：如何使用SToFMEmbeddingAnalyzer"""
    
    # 示例1: 基本使用
    BASE_DIR = Path("/data/xiaox/projects/20250801_aging")
    sample_name = "C-18M-1"
    emb_path = BASE_DIR / "2_stofm_processed" / sample_name / "stofm_emb.npy"
    data_path = BASE_DIR / "2_stofm_processed" / sample_name / "data.h5ad"
    
    if not emb_path.exists():
        print(f"错误: 找不到文件 {emb_path}")
        print("\n请先运行 get_embeddings.py 生成 embeddings")
        return
    
    # 创建分析器
    analyzer = SToFMEmbeddingAnalyzer(
        emb_path=emb_path,
        data_h5ad_path=data_path,
        sample_name=sample_name
    )
    
    # 1. 聚类分析
    cluster_col = analyzer.clustering(method='leiden', resolution=0.5)
    
    # 2. 降维可视化
    reduction_key, coords = analyzer.dimensionality_reduction(method='umap', use_embeddings=True)
    
    # 3. 可视化聚类结果
    output_dir = BASE_DIR / "3_stofm_embeddings" / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    analyzer.visualize_clusters(
        cluster_col=cluster_col,
        reduction_key=reduction_key,
        spatial=True,
        save_path=output_dir / f"{sample_name}_clusters.png"
    )
    
    # 4. 如果有细胞类型注释，可以进行注释
    # analyzer.cell_type_annotation(method='knn')
    
    # 5. 保存结果
    analyzer.save_results(output_dir=BASE_DIR / "3_stofm_embeddings" / "results")
    
    print("\n✓ 分析完成！")


if __name__ == "__main__":
    main()

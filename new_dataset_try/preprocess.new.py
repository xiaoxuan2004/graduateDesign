"""
预处理脚本：将小鼠spaceranger数据转换为SToFM模型输入格式
处理所有10个样本，生成hf.dataset和data.h5ad文件
"""
import scanpy as sc
import pandas as pd
import numpy as np
import pickle as pkl
import os
import sys
from pathlib import Path

# 添加SToFM路径以便导入
stofm_path = Path(__file__).parent.parent / "SToFM" / "SToFM"
sys.path.insert(0, str(stofm_path))

from geneformer import TranscriptomeTokenizer
from geneformer.tokenizer import tokenize_cell

# 定义路径
BASE_DIR = Path("/data/xiaox/projects/20250801_aging")
SPACERANGER_DIR = BASE_DIR / "1_spaceranger_count"
OUTPUT_DIR = BASE_DIR / "2_stofm_processed"
MOUSE2HUMAN_PKL = stofm_path / "preprocessing" / "mouseid2humanid.pkl"

# 确保输出目录存在
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 获取所有样本名称
sample_dirs = [d for d in SPACERANGER_DIR.iterdir() 
               if d.is_dir() and not d.name.startswith('.') and d.name != 'rsync.sh' and d.name != 'samples.txt']
sample_names = sorted([d.name for d in sample_dirs])
print(f"找到 {len(sample_names)} 个样本: {sample_names}")


class SToFMTranscriptomeTokenizer(TranscriptomeTokenizer):
    """继承Geneformer的tokenizer，用于SToFM"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize_anndata(self, data):
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }
        else:
            file_cell_metadata = None

        # 定义编码蛋白和miRNA基因的位置
        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in data.var["ensembl_id"]]
        )[0]
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in data.var["ensembl_id"][coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = data.var["ensembl_id"][coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        # 定义通过过滤的细胞位置
        try:
            data.obs["filter_pass"]
            var_exists = True
        except AttributeError:
            var_exists = False

        if var_exists is True:
            filter_pass_loc = np.where(
                [True if i == 1 else False for i in data.obs["filter_pass"]]
            )[0]
        else:
            print(f"数据没有'filter_pass'列，将对所有细胞进行tokenization")
            filter_pass_loc = np.array([i for i in range(data.shape[0])])

        # tokenize细胞
        tokenized_cells = []
        subview = data[filter_pass_loc, coding_miRNA_loc]

        # 归一化：按细胞总计数归一化，乘以10000，再按基因归一化因子归一化
        subview_norm_array = (
            subview.X.toarray().T
            / subview.obs.n_counts.to_numpy()
            * 10_000
            / norm_factor_vector[:, None]
        )
        
        # tokenize每个细胞的基因表达向量
        tokenized_cells += [
            tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
            for i in range(subview_norm_array.shape[1])
        ]

        # 添加自定义属性
        if self.custom_attr_name_dict is not None:
            for k in file_cell_metadata.keys():
                file_cell_metadata[k] += subview.obs[k].tolist()

        return tokenized_cells, file_cell_metadata


def process_sample(sample_name):
    """处理单个样本"""
    print(f"\n{'='*60}")
    print(f"处理样本: {sample_name}")
    print(f"{'='*60}")
    
    # 输入路径
    sample_input_dir = SPACERANGER_DIR / sample_name
    visium_path = sample_input_dir / "outs"
    
    # 输出路径
    sample_output_dir = OUTPUT_DIR / sample_name
    sample_output_dir.mkdir(exist_ok=True, parents=True)
    
    h5ad_path = sample_output_dir / "data.h5ad"
    hf_dataset_path = sample_output_dir / "hf.dataset"
    
    # 检查是否已经处理过
    if h5ad_path.exists() and hf_dataset_path.exists():
        print(f"样本 {sample_name} 已处理，跳过...")
        return sample_output_dir
    
    # 1. 读取spaceranger数据
    print(f"读取数据: {visium_path}")
    
    # 尝试多种路径：标准visium -> binned_outputs
    adata = None
    data_path = None
    
    # 首先尝试标准 visium 路径
    if (visium_path / "filtered_feature_bc_matrix.h5").exists():
        try:
            print(f"  尝试标准 visium 路径: {visium_path}")
            adata = sc.read_visium(str(visium_path))
            data_path = visium_path
        except Exception as e:
            print(f"  标准路径读取失败: {e}")
    
    # 如果标准路径失败，尝试 binned_outputs
    if adata is None:
        binned_dirs = [
            visium_path / "binned_outputs" / "square_016um",
            visium_path / "binned_outputs" / "square_008um",
            visium_path / "binned_outputs" / "square_002um",
        ]
        
        for binned_path in binned_dirs:
            if binned_path.exists():
                try:
                    print(f"  尝试 binned 路径: {binned_path}")
                    # 读取 h5 文件
                    h5_file = binned_path / "filtered_feature_bc_matrix.h5"
                    if h5_file.exists():
                        adata = sc.read_10x_h5(str(h5_file))
                        adata.var_names_make_unique()
                        
                        # 读取空间信息 (parquet 格式)
                        spatial_file = binned_path / "spatial" / "tissue_positions.parquet"
                        if spatial_file.exists():
                            import pyarrow.parquet as pq
                            spatial_df = pq.read_table(str(spatial_file)).to_pandas()
                            # 设置索引为 barcode
                            if 'barcode' in spatial_df.columns:
                                spatial_df.set_index('barcode', inplace=True)
                            
                            # 匹配 barcodes 并添加空间信息
                            common_barcodes = adata.obs_names.intersection(spatial_df.index)
                            if len(common_barcodes) > 0:
                                adata = adata[common_barcodes]
                                
                                # 添加空间坐标 (使用像素坐标)
                                if 'pxl_col_in_fullres' in spatial_df.columns and 'pxl_row_in_fullres' in spatial_df.columns:
                                    spatial_coords = spatial_df.loc[adata.obs_names, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
                                elif 'array_col' in spatial_df.columns and 'array_row' in spatial_df.columns:
                                    # 如果没有像素坐标，使用 array 坐标
                                    spatial_coords = spatial_df.loc[adata.obs_names, ['array_col', 'array_row']].values
                                else:
                                    # 尝试使用前两列作为坐标
                                    spatial_coords = spatial_df.loc[adata.obs_names, spatial_df.columns[:2]].values
                                
                                adata.obsm['spatial'] = spatial_coords
                                
                                # 添加 in_tissue 信息到 obs
                                if 'in_tissue' in spatial_df.columns:
                                    adata.obs['in_tissue'] = spatial_df.loc[adata.obs_names, 'in_tissue'].values
                                
                                data_path = binned_path
                                print(f"  ✓ 成功读取 binned 数据")
                                break
                            else:
                                print(f"  ✗ barcode 不匹配")
                        else:
                            print(f"  ✗ 未找到空间信息文件: {spatial_file}")
                except Exception as e:
                    print(f"  ✗ binned 路径 {binned_path} 读取失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    if adata is None:
        print(f"  错误: 无法从任何路径读取数据")
        return None
    
    print(f"  数据形状: {adata.shape}")
    print(f"  基因数: {adata.n_vars}, 细胞数: {adata.n_obs}")
    if 'spatial' in adata.obsm:
        print(f"  空间信息: 已加载 ({adata.obsm['spatial'].shape})")
    
    # 2. 确保使用原始计数
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
        print("  使用layers['counts']作为原始计数")
    else:
        print("  使用adata.X作为计数数据")
    
    # 3. 加载小鼠到人类的基因映射
    print("加载基因映射表...")
    try:
        mouseid2humanid = pkl.load(open(MOUSE2HUMAN_PKL, "rb"))
        print(f"  加载了 {len(mouseid2humanid)} 个基因映射")
    except Exception as e:
        print(f"  错误: 无法加载基因映射文件 - {e}")
        return None
    
    # 4. 检查基因ID格式并映射
    print("映射基因ID...")
    # spaceranger通常使用gene_symbols作为var_names，需要找到gene_ids
    if 'gene_ids' in adata.var.columns:
        gene_id_col = 'gene_ids'
    elif 'gene_id' in adata.var.columns:
        gene_id_col = 'gene_id'
    else:
        # 如果没有gene_ids列，尝试使用var_names
        print("  警告: 未找到gene_ids列，尝试使用var_names")
        gene_id_col = None
    
    # 创建ensembl_id列
    if gene_id_col:
        # 使用gene_ids列进行映射
        adata.var['ensembl_id'] = [
            mouseid2humanid.get(gene_id, gene_id) 
            for gene_id in adata.var[gene_id_col]
        ]
    else:
        # 尝试直接使用var_names映射
        adata.var['ensembl_id'] = [
            mouseid2humanid.get(gene_name, gene_name) 
            for gene_name in adata.var_names
        ]
    
    # 统计映射结果
    if gene_id_col:
        original_ids = adata.var[gene_id_col].values
    else:
        original_ids = adata.var_names.values
    mapped_count = sum(1 for orig, mapped in zip(original_ids, adata.var['ensembl_id']) 
                      if mapped != orig or mapped in mouseid2humanid.values())
    print(f"  成功映射基因数: {mapped_count}/{len(adata.var)}")
    
    # 5. 添加必要的obs属性
    print("添加细胞属性...")
    # 计算每个细胞的总UMI数
    from scipy.sparse import issparse
    if issparse(adata.X):
        adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    else:
        adata.obs['n_counts'] = adata.X.sum(axis=1)
    
    # 添加filter_pass列（所有细胞都通过）
    adata.obs['filter_pass'] = 1
    
    # 确保空间坐标存在
    if 'spatial' in adata.obsm:
        print("  保留空间坐标信息")
        adata.obs['x_coord'] = adata.obsm['spatial'][:, 0]
        adata.obs['y_coord'] = adata.obsm['spatial'][:, 1]
    else:
        print("  警告: 未找到空间坐标信息")
    
    # 6. 初始化tokenizer并tokenize
    print("初始化tokenizer...")
    try:
        tk = SToFMTranscriptomeTokenizer({}, nproc=4)
        print("  Tokenizing数据...")
        tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)
        print(f"  成功tokenize {len(tokenized_cells)} 个细胞")
    except Exception as e:
        print(f"  错误: Tokenization失败 - {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 7. 创建并保存dataset
    print("创建dataset...")
    try:
        tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)
        print(f"  保存dataset到: {hf_dataset_path}")
        tokenized_dataset.save_to_disk(str(hf_dataset_path))
    except Exception as e:
        print(f"  错误: 保存dataset失败 - {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 8. 保存h5ad文件
    print(f"保存h5ad文件到: {h5ad_path}")
    try:
        adata.write(str(h5ad_path))
        print("  保存成功")
    except Exception as e:
        print(f"  错误: 保存h5ad失败 - {e}")
        return None
    
    print(f"✓ 样本 {sample_name} 处理完成!")
    return sample_output_dir


def main():
    """主函数：处理所有样本"""
    print("="*60)
    print("SToFM 数据预处理")
    print("="*60)
    print(f"输入目录: {SPACERANGER_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"基因映射文件: {MOUSE2HUMAN_PKL}")
    
    # 处理所有样本
    processed_samples = []
    for sample_name in sample_names:
        result = process_sample(sample_name)
        if result:
            processed_samples.append(sample_name)
    
    # 生成样本列表文件（用于get_embeddings.py）
    sample_list_file = OUTPUT_DIR / "sample_list.txt"
    with open(sample_list_file, 'w') as f:
        for sample_name in processed_samples:
            f.write(f"{OUTPUT_DIR / sample_name}\n")
    
    print(f"\n{'='*60}")
    print(f"预处理完成!")
    print(f"成功处理 {len(processed_samples)}/{len(sample_names)} 个样本")
    print(f"样本列表已保存到: {sample_list_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
针对 DLPFC 数据的预处理脚本
修复：强制加载 spatial 坐标，并正确保存 hf.dataset 与 data.h5ad
"""
import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
import shutil
from pathlib import Path
from scipy.sparse import issparse

# ================= 1. 严格定义你的系统路径 =================
DLPFC_DIR = Path("/data/xiaox/projects/20250801_aging/scripts/STAIG/STAIG/Dataset/Dataset/DLPFC")
STOFM_PATH = Path("/data/xiaox/projects/20250801_aging/scripts/SToFM")
GRADUATE_DESIGN_DIR = Path("/data/xiaox/projects/20250801_aging/scripts/graduate_design")

OUTPUT_DIR = GRADUATE_DESIGN_DIR / "2_stofm_processed_dlpfc"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TARGET_SLIDES = ['151507', '151508', '151673','151675', '151676']

sys.path.insert(0, str(STOFM_PATH))
from geneformer import TranscriptomeTokenizer
from geneformer.tokenizer import tokenize_cell

class SToFMTranscriptomeTokenizer(TranscriptomeTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize_anndata(self, data):
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}
        else:
            file_cell_metadata = None

        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in data.var["ensembl_id"]]
        )[0]
        
        coding_miRNA_ids = data.var["ensembl_id"].iloc[coding_miRNA_loc]
        norm_factor_vector = np.array([self.gene_median_dict[i] for i in coding_miRNA_ids])
        coding_miRNA_tokens = np.array([self.gene_token_dict[i] for i in coding_miRNA_ids])

        filter_pass_loc = np.array([i for i in range(data.shape[0])])

        tokenized_cells = []
        subview = data[filter_pass_loc, coding_miRNA_loc]

        subview_norm_array = (
            subview.X.toarray().T
            / subview.obs.n_counts.to_numpy()
            * 10_000
            / norm_factor_vector[:, None]
        )
        
        tokenized_cells += [
            tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
            for i in range(subview_norm_array.shape[1])
        ]

        if self.custom_attr_name_dict is not None:
            for k in file_cell_metadata.keys():
                file_cell_metadata[k] += subview.obs[k].tolist()

        return tokenized_cells, file_cell_metadata

def process_dlpfc_slide(slide_id):
    print(f"\n{'='*50}\n正在处理 DLPFC 切片: {slide_id}\n{'='*50}")
    
    slide_path = DLPFC_DIR / slide_id
    sample_output_dir = OUTPUT_DIR / slide_id
    
    # 【核心修复1】如果该切片存在旧文件夹，直接删掉重来，绝不用历史遗留错误文件
    if sample_output_dir.exists():
        shutil.rmtree(sample_output_dir)
    sample_output_dir.mkdir(exist_ok=True, parents=True)

    print(f"读取数据路径: {slide_path}")
    try:
        # 【核心修复2】强制 load_images=True，逼迫 Scanpy 必须读取 spatial 坐标目录
        adata = sc.read_visium(str(slide_path), count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
    except Exception as e:
        print(f"读取失败: {e}")
        return None

    truth_path = slide_path / 'truth.txt'
    if truth_path.exists():
        df_meta = pd.read_csv(truth_path, sep='\t', header=None)
        adata.obs['ground_truth'] = df_meta[1].values
        original_len = adata.shape[0]
        adata = adata[~pd.isnull(adata.obs['ground_truth'])].copy()
        print(f"✔ 应用 truth.txt 过滤 NA 节点: {original_len} -> {adata.shape[0]} 个 Spots")

    if 'gene_ids' in adata.var.columns:
        adata.var['ensembl_id'] = adata.var['gene_ids']
    else:
        adata.var['ensembl_id'] = adata.var_names
        
    adata.X = adata.X.toarray() if issparse(adata.X) else adata.X
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    adata.obs['filter_pass'] = 1

    print("初始化 Tokenizer 并转换序列...")
    tk = SToFMTranscriptomeTokenizer({}, nproc=4)
    tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)
    
    print("保存 HuggingFace Dataset...")
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)
    tokenized_dataset.save_to_disk(str(sample_output_dir / "hf.dataset"))
    
    # 【核心修复3】明确检查坐标是否存在，并保存干净的 data.h5ad
    print(f"验证 spatial 坐标是否成功加载: {'spatial' in adata.obsm}")
    print("保存带有坐标的 data.h5ad...")
    adata.write(str(sample_output_dir / "data.h5ad"))
    
    barcode_df = pd.DataFrame(index=adata.obs_names)
    barcode_df.to_csv(sample_output_dir / "filtered_barcodes.csv")
    
    print(f"✓ {slide_id} 处理完成！")
    return sample_output_dir

if __name__ == "__main__":
    processed_samples = []
    for slide in TARGET_SLIDES:
        res = process_dlpfc_slide(slide)
        if res:
            processed_samples.append(slide)
            
    sample_list_file = OUTPUT_DIR / "sample_list.txt"
    with open(sample_list_file, 'w') as f:
        for slide in processed_samples:
            f.write(f"{OUTPUT_DIR / slide}\n")
    print(f"\n全部处理完成，样本列表已保存至: {sample_list_file}")
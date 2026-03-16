# SToFM 数据处理流程

本目录包含使用SToFM模型处理小鼠空间转录组数据的完整流程。

## 文件说明

- `preprocess.py`: 数据预处理脚本，将spaceranger输出转换为SToFM输入格式
- `get_embeddings.py`: 提取embeddings的脚本
- `run_pipeline.sh`: 完整流程运行脚本

## 数据准备

### 输入数据
- 位置: `/data/xiaox/projects/20250801_aging/1_spaceranger_count/`
- 格式: spaceranger输出目录（每个样本一个文件夹）
- 样本数: 10个样本

### 输出目录
- 预处理数据: `/data/xiaox/projects/20250801_aging/2_stofm_processed/`
- Embeddings: `/data/xiaox/projects/20250801_aging/3_stofm_embeddings/`

## 使用步骤

### 步骤1: 数据预处理

运行预处理脚本，将小鼠spaceranger数据转换为SToFM所需格式：

```bash
cd /data/xiaox/projects/20250801_aging/scripts/graduate_design
python3 preprocess.py
```

**预处理过程包括：**
1. 读取spaceranger数据（使用`scanpy.read_visium`）
2. 将小鼠基因ID映射到人类Ensembl ID（使用`mouseid2humanid.pkl`）
3. 添加必要的属性（`n_counts`, `filter_pass`, `ensembl_id`）
4. 使用SToFMTranscriptomeTokenizer进行tokenization
5. 保存为`hf.dataset`和`data.h5ad`格式

**输出：**
- 每个样本在`2_stofm_processed/`下生成一个目录
- 每个目录包含：
  - `data.h5ad`: 处理后的AnnData对象
  - `hf.dataset/`: HuggingFace dataset格式的tokenized数据

### 步骤2: 提取Embeddings

Checkpoint文件已解压到 `checkpoint/ckpt/` 目录，可以直接运行：

```bash
python3 get_embeddings.py \
  --batch_size 4 \
  --gpu 0
```

**参数说明：**
- `--cell_encoder_path`: Cell encoder checkpoint路径（默认: `checkpoint/ckpt/cell_encoder`）
- `--config_path`: 模型config.json路径（默认: `checkpoint/ckpt/config.json`）
- `--model_path`: SE(2) Transformer模型路径（默认: `checkpoint/ckpt/se2transformer.pth`）
- `--batch_size`: 批处理大小（默认4，根据GPU内存调整）
- `--gpu`: 使用的GPU编号（默认0）

如果checkpoint在其他位置，可以手动指定路径。

**参数说明：**
- `--cell_encoder_path`: Cell encoder checkpoint路径
- `--config_path`: 模型config.json路径
- `--model_path`: SE(2) Transformer模型路径
- `--batch_size`: 批处理大小（默认4，根据GPU内存调整）
- `--gpu`: 使用的GPU编号（默认0）

**输出：**
- 每个样本目录下生成`stofm_emb.npy`文件
- 包含所有细胞的256维embeddings

### 完整流程（一键运行）

```bash
bash run_pipeline.sh
```

注意：确保checkpoint文件已解压到 `checkpoint/ckpt/` 目录。

## 下游任务

提取embeddings后，可以用于各种下游任务：

1. **细胞类型注释**
   - 使用embeddings训练分类器
   - 或使用聚类方法进行无监督注释

2. **组织区域分割**
   - 基于空间坐标和embeddings进行区域识别

3. **差异表达分析**
   - 结合embeddings和基因表达数据

4. **空间模式分析**
   - 利用embeddings和空间坐标分析空间分布模式

## 注意事项

1. **基因映射**: 小鼠基因会自动映射到人类基因词汇表。如果某些基因无法映射，会被保留原ID。

2. **内存使用**: 
   - 预处理阶段需要足够内存处理所有样本
   - Embedding提取阶段需要GPU内存，根据GPU调整`batch_size`

3. **数据格式**: 
   - 确保spaceranger输出目录结构正确
   - 每个样本目录应包含`outs/`子目录

4. **依赖环境**: 
   - 需要安装scanpy, geneformer等依赖
   - 参考SToFM的requirements.txt

## 故障排除

### 问题1: 无法找到gene_ids列
- **原因**: spaceranger输出格式可能不同
- **解决**: 脚本会自动尝试使用var_names作为基因ID

### 问题2: Tokenization失败
- **原因**: 基因映射不完整或数据格式问题
- **解决**: 检查`mouseid2humanid.pkl`文件是否存在，检查数据质量

### 问题3: GPU内存不足
- **原因**: batch_size太大
- **解决**: 减小`--batch_size`参数（如改为2或1）

## 联系与支持

如有问题，请参考：
- SToFM原始文档: `scripts/SToFM/SToFM/README.md`
- SToFM论文: https://arxiv.org/abs/2507.11588

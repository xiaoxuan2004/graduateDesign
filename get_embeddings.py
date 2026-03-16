"""
提取SToFM embeddings的脚本
使用预处理后的数据提取细胞embeddings
"""
import os
import sys
import argparse
from pathlib import Path

# 添加SToFM路径
BASE_DIR = Path("/data/xiaox/projects/20250801_aging")
STOFM_DIR = BASE_DIR / "scripts" / "SToFM" / "SToFM"
sys.path.insert(0, str(STOFM_DIR))

# 默认路径
DEFAULT_PROCESSED_DIR = BASE_DIR / "2_stofm_processed"
DEFAULT_OUTPUT_DIR = BASE_DIR / "3_stofm_embeddings"

# 默认checkpoint路径（已解压）
CHECKPOINT_DIR = BASE_DIR / "scripts" / "graduate_design" / "checkpoint" / "ckpt"
DEFAULT_CELL_ENCODER_PATH = str(CHECKPOINT_DIR / "cell_encoder")
DEFAULT_CONFIG_PATH = str(CHECKPOINT_DIR / "config.json")
DEFAULT_MODEL_PATH = str(CHECKPOINT_DIR / "se2transformer.pth")


def create_get_embeddings_script(
    processed_dir,
    output_dir,
    cell_encoder_path,
    config_path,
    model_path,
    batch_size=4,
    output_filename="stofm_emb.npy"
):
    """创建运行get_embeddings.py的命令"""
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成样本列表文件
    sample_list_file = processed_dir / "sample_list.txt"
    if not sample_list_file.exists():
        # 如果没有列表文件，自动生成
        sample_dirs = [d for d in processed_dir.iterdir() 
                      if d.is_dir() and (d / "hf.dataset").exists()]
        with open(sample_list_file, 'w') as f:
            for sample_dir in sorted(sample_dirs):
                f.write(f"{sample_dir}\n")
        print(f"已生成样本列表文件: {sample_list_file}")
    
    # 构建命令
    get_embeddings_script = STOFM_DIR / "get_embeddings.py"
    
    # 切换到SToFM目录运行（因为需要导入model模块）
    cmd = f"""cd {STOFM_DIR} && python get_embeddings.py \\
    --cell_encoder_path {cell_encoder_path} \\
    --config_path {config_path} \\
    --model_path {model_path} \\
    --data_path {sample_list_file} \\
    --output_filename {output_filename} \\
    --batch_size {batch_size} \\
    --seed 42"""
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="提取SToFM embeddings")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="预处理后的数据目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="embeddings输出目录"
    )
    parser.add_argument(
        "--cell_encoder_path",
        type=str,
        default=DEFAULT_CELL_ENCODER_PATH,
        help="Cell encoder checkpoint路径"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="模型config.json路径"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="SE(2) Transformer模型路径"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批处理大小"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="stofm_emb.npy",
        help="输出文件名"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="使用的GPU编号"
    )
    
    args = parser.parse_args()
    
    # 检查checkpoint路径
    if not Path(args.cell_encoder_path).exists():
        print(f"错误: Cell encoder路径不存在: {args.cell_encoder_path}")
        print("请从 https://drive.google.com/drive/folders/1mHE8gf8MAPwzZoEB0vwOOfQ4lz3H_-xo?usp=sharing 下载checkpoint")
        return
    
    if not Path(args.config_path).exists():
        print(f"错误: Config路径不存在: {args.config_path}")
        return
    
    if not Path(args.model_path).exists():
        print(f"错误: 模型路径不存在: {args.model_path}")
        return
    
    # 生成命令
    cmd = create_get_embeddings_script(
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        cell_encoder_path=args.cell_encoder_path,
        config_path=args.config_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        output_filename=args.output_filename
    )
    
    # 设置CUDA设备
    env_cmd = f"export CUDA_VISIBLE_DEVICES={args.gpu}\n"
    
    # 保存为shell脚本
    script_file = Path(args.output_dir) / "run_get_embeddings.sh"
    with open(script_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(env_cmd)
        f.write(cmd)
    
    # 添加执行权限
    os.chmod(script_file, 0o755)
    
    print("="*60)
    print("已生成运行脚本!")
    print(f"脚本路径: {script_file}")
    print("\n运行命令:")
    print(f"bash {script_file}")
    print("="*60)
    
    # 询问是否立即运行
    response = input("\n是否立即运行? (y/n): ")
    if response.lower() == 'y':
        import subprocess
        print("\n开始运行...")
        subprocess.run(["bash", str(script_file)])


if __name__ == "__main__":
    main()

"""
验证checkpoint路径配置是否正确
"""
from pathlib import Path

BASE_DIR = Path("/data/xiaox/projects/20250801_aging")
CHECKPOINT_DIR = BASE_DIR / "scripts" / "graduate_design" / "checkpoint" / "ckpt"

print("="*60)
print("检查Checkpoint路径配置")
print("="*60)

# 检查各个文件
cell_encoder_path = CHECKPOINT_DIR / "cell_encoder"
config_path = CHECKPOINT_DIR / "config.json"
model_path = CHECKPOINT_DIR / "se2transformer.pth"
cell_bert_path = cell_encoder_path / "cell_bert"
cell_proj_path = cell_encoder_path / "cell_proj.bin"

checks = [
    ("Cell encoder目录", cell_encoder_path, True),
    ("Cell bert目录", cell_bert_path, True),
    ("Cell proj.bin", cell_proj_path, False),
    ("Config.json", config_path, False),
    ("SE2Transformer模型", model_path, False),
]

all_ok = True
for name, path, is_dir in checks:
    exists = path.exists()
    if is_dir:
        is_correct = exists and path.is_dir()
    else:
        is_correct = exists and path.is_file()
    
    status = "✓" if is_correct else "✗"
    print(f"{status} {name}: {path}")
    if not is_correct:
        all_ok = False
        print(f"  错误: 路径不存在或类型不正确")

print("="*60)
if all_ok:
    print("✓ 所有checkpoint文件路径配置正确!")
    print("\n可以直接运行:")
    print("  python3 get_embeddings.py --batch_size 4 --gpu 0")
else:
    print("✗ 部分checkpoint文件缺失，请检查!")
print("="*60)

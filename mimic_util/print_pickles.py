import pickle
from pathlib import Path
import pandas as pd

def explore(obj, indent=0, max_rows=3):
    """递归打印对象结构"""
    prefix = " " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}: {type(v)}")
            explore(v, indent + 2, max_rows)
    elif isinstance(obj, pd.DataFrame):
        print(f"{prefix}DataFrame 形状: {obj.shape}")
        print(obj.head(max_rows))
    elif isinstance(obj, pd.Series):
        print(f"{prefix}Series 长度: {len(obj)}")
        print(obj.head(max_rows))
    else:
        # 其他类型，打印前几个内容
        try:
            print(f"{prefix}{str(obj)[:200]}")
        except Exception:
            print(f"{prefix}{obj}")

# ---- 使用示例 ----
data_dir = Path("/hpc/home/yy450/link_kamaleswaranlab/EmoryDataset/EMR_Supertables/2022")

# 只读一个文件
pkl_file = next(data_dir.glob("*.pickle"))
print("正在读取文件:", pkl_file)

with open(pkl_file, "rb") as f:
    obj = pickle.load(f)

explore(obj)

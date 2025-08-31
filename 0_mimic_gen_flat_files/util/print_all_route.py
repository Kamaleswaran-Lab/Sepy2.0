import pandas as pd

def get_unique_routes(csv_file, output_file=None):
    """
    读取CSV文件，提取route列的所有唯一值
    """
    try:
        # 避免 dtype warning
        df = pd.read_csv(csv_file, low_memory=False)

        if "route" not in df.columns:
            raise ValueError(f"CSV文件中未找到 'route' 列，实际列有: {df.columns.tolist()}")

        # 获取唯一值，去掉缺失值
        unique_routes = df["route"].dropna().unique()
        unique_routes_set = set(unique_routes)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                for r in sorted(unique_routes_set):
                    f.write(str(r) + "\n")

        return unique_routes_set

    except Exception as e:
        print(f"出错了: {e}")
        return set()

if __name__ == "__main__":
    csv_path = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_exports/hosp_prescriptions.csv"
    result = get_unique_routes(csv_path, "unique_routes.txt")
    print("所有可能的 route 值：")
    for r in sorted(result):
        print(r)

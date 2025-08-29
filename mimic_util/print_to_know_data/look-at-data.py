import os
import csv

folder_path = "/hpc/home/yy450/link_kamaleswaranlab/EmoryDataset/EMR_RAW/2022"   # 修改成你的目录
# folder_path = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_exports"   # 修改成你的目录

output_file = "/hpc/home/yy450/link_dctrl_yy450/Sepy2.0/mimic_util/print_to_know_data_1/dsv_preview_emory.txt"  # 输出文件名
CANDIDATE_DELIMITERS = [",", "\t", "|", ";", "^", "~"]

def read_head(file_path, n=10):
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
            f.seek(0)

            # 尝试自动检测分隔符
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS))
                delimiter = dialect.delimiter
            except Exception:
                # fallback：找到最可能的分隔符
                delimiter = None
                for delim in CANDIDATE_DELIMITERS:
                    if delim in sample:
                        delimiter = delim
                        break
                if delimiter is None:
                    delimiter = ","  # 默认

            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            rows = []
            for i, row in enumerate(reader):
                if i >= n:
                    break
                rows.append(row)
            return header, rows, delimiter
    except Exception as e:
        return None, [], None

with open(output_file, "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv") or file.endswith(".dsv"):
                file_path = os.path.join(root, file)
                out.write("="*80 + "\n")
                out.write(f"📂 文件: {file_path}\n")
                out.write("="*80 + "\n")

                header, rows, delim = read_head(file_path)
                if header:
                    out.write(f"分隔符: '{delim}'\n")
                    out.write("表头:\n")
                    out.write(str(header) + "\n\n")

                    # === 新增功能：示例数据对照表 ===
                    if rows:
                        first_row = rows[0]
                        out.write("列名 → 示例值:\n")
                        for col, val in zip(header, first_row):
                            out.write(f"  {col}: {val}\n")
                        out.write("\n")

                    # === 前10行数据 ===
                    out.write("前10行数据:\n")
                    for row in rows:
                        out.write(str(row) + "\n")
                else:
                    out.write("⚠️ 无法解析此文件\n")
                out.write("\n\n")

print(f"\n✅ 所有 CSV/DSV 文件已处理完成，结果已保存到: {output_file}")

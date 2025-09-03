import csv

def count_csv_rows(file_path):
    """统计 CSV 文件的行数"""
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return sum(1 for _ in reader)

def compare_csv_rows(file1, file2):
    rows1 = count_csv_rows(file1)
    rows2 = count_csv_rows(file2)
    return rows1, rows2

if __name__ == "__main__":
    file1 = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_exports/icu_inputevents.csv"
    file2 = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/mimic_flat_files/INFUSIONMEDS.csv"
    
    rows1, rows2 = compare_csv_rows(file1, file2)
    print(f"{file1} 行数: {rows1}")
    print(f"{file2} 行数: {rows2}")

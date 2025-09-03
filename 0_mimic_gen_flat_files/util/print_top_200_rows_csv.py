import csv

with open("/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_exports/icu_ingredientevents.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i >= 200:  # 超过200行就停止
            break
        print(row)

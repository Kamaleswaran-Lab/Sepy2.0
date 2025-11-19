import os
import csv

# folder_path = "/hpc/home/yy450/link_kamaleswaranlab/EmoryDataset/EMR_RAW/2022"   # Change to your directory
# folder_path = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_exports"   # Change to your directory
# folder_path = "/hpc/dctrl/yy450/link_mimic_iv/mimic-iv-3.1-decompress/note"   # Change to your directory
# folder_path = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_concepts_exports"   # Change to your directory
# folder_path = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_mapping"   # Change to your directory
# folder_path = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/mimic_flat_files"   # Change to your directory
folder_path = "/hpc/dctrl/yy450/link_work/colorado-sample"   # Change to your directory


output_file = "/hpc/dctrl/yy450/Sepy2.0/mimic_util/print_to_know_data/csv_raw_preview_colorado.txt"  # Output file name
CANDIDATE_DELIMITERS = [",", "\t", "|", ";", "^", "~"]

def read_head(file_path, n=30):
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
            f.seek(0)

            # Try to auto-detect delimiter
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS))
                delimiter = dialect.delimiter
            except Exception:
                # Fallback: find the most likely delimiter
                delimiter = None
                for delim in CANDIDATE_DELIMITERS:
                    if delim in sample:
                        delimiter = delim
                        break
                if delimiter is None:
                    delimiter = ","  # Default

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
                out.write(f"📂 File: {file_path}\n")
                out.write("="*80 + "\n")

                header, rows, delim = read_head(file_path)
                if header:
                    out.write(f"Delimiter: '{delim}'\n")
                    out.write("Header:\n")
                    out.write(str(header) + "\n\n")

                    # === New feature: Column-to-sample-value mapping ===
                    if rows:
                        first_row = rows[0]
                        out.write("Column → Sample value:\n")
                        for col, val in zip(header, first_row):
                            out.write(f"  {col}: {val}\n")
                        out.write("\n")

                    # === First 10 rows of data ===
                    out.write("First 10 rows of data:\n")
                    for row in rows:
                        out.write(str(row) + "\n")
                else:
                    out.write("⚠️ Unable to parse this file\n")
                out.write("\n\n")

print(f"\n✅ All CSV/DSV files have been processed, results saved to: {output_file}")




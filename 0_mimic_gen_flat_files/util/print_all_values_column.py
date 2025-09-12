import pandas as pd
import csv

def detect_delimiter(file_path: str, sample_size: int = 2048) -> str:
    """Auto-detect delimiter by reading a sample of the file."""
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(sample_size)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return ","  # fallback to comma if detection fails

def extract_column_values(file_path: str, column_name: str, delimiter: str = None, chunksize: int = 100000):
    """Extract unique values from a given column in a large CSV/DSV file using chunked reading."""
    try:
        if delimiter is None:
            delimiter = detect_delimiter(file_path)

        unique_values = set()

        for chunk in pd.read_csv(
            file_path,
            delimiter=delimiter,
            engine="python",
            chunksize=chunksize,
        ):
            if column_name not in chunk.columns:
                print(f"Column '{column_name}' not found! Available columns: {list(chunk.columns)}")
                return
            unique_values.update(chunk[column_name].dropna().unique())

        print(f"All possible values for column '{column_name}':")
        for val in unique_values:
            print(val)

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    file_path = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_exports/icu_inputevents.csv"
    column_name = "itemid"
    delimiter = None  # None -> auto-detect, or set manually like "," or "|"
    extract_column_values(file_path, column_name, delimiter)





# import pandas as pd
# import csv

# def detect_delimiter(file_path: str, sample_size: int = 2048) -> str:
#     """
#     Try to automatically detect the delimiter by reading a sample of the file.
#     """
#     with open(file_path, "r", newline="", encoding="utf-8") as f:
#         sample = f.read(sample_size)
#         sniffer = csv.Sniffer()
#         try:
#             dialect = sniffer.sniff(sample)
#             return dialect.delimiter
#         except csv.Error:
#             return ","  # fallback to comma if detection fails


# def extract_column_values(file_path: str, column_name: str, delimiter: str = None):
#     """
#     Extract all possible unique values from a given column in a CSV/DSV file.

#     :param file_path: Path to the CSV/DSV file
#     :param column_name: Name of the column to analyze
#     :param delimiter: Delimiter for the file (default None -> auto-detect)
#     """
#     try:
#         if delimiter is None:
#             delimiter = detect_delimiter(file_path)
#             # print(f"Detected delimiter: '{delimiter}'")

#             df = pd.read_csv(file_path, delimiter=delimiter, engine="python")
        
#         if column_name not in df.columns:
#             print(f"Column '{column_name}' not found! Available columns: {list(df.columns)}")
#             return
        
#         unique_values = df[column_name].dropna().unique()
        
#         print(f"All possible values for column '{column_name}':")
#         for val in unique_values:
#             print(val)
#     except Exception as e:
#         print(f"Error reading file: {e}")


# if __name__ == "__main__":
#     # Edit these parameters directly
#     file_path = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_exports/icu_inputevents.csv"
#     # file_path = "/hpc/home/yy450/link_kamaleswaranlab/EmoryDataset/EMR_RAW/2022/CJSEPSIS_INFUSIONMEDS_2022.dsv"
#     column_name = "itemid"
#     delimiter = None  # None -> auto-detect, or set manually like "," or "|"
    
#     extract_column_values(file_path, column_name, delimiter)

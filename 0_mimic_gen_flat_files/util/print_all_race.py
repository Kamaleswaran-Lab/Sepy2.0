import csv

def extract_unique_races(csv_file):
    races = set()  # use a set to keep only unique values

    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)  # read rows as dictionaries
        for row in reader:
            race = row.get("race_code", "").strip()
            if race:  # avoid empty values
                races.add(race)

    return races

if __name__ == "__main__":
    # file_path = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/builtdata/csv_exports/hosp_admissions.csv"
    file_path = "/hpc/home/yy450/link_dctrl_yy450/Sepy2.0/mimic_util/print_to_know_data/dsv_preview_emory.txt"
    unique_races = extract_unique_races(file_path)
    
    print("Unique races found in the dataset:")
    for race in sorted(unique_races):
        print(race)

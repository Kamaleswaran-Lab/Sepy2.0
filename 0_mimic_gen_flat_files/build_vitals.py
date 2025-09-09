import pandas as pd
import os

# Base dir for your mimic exports
base_dir = "/hpc/group/kamaleswaranlab/mimic_iv/builtdata/csv_exports"
items_path = os.path.join(base_dir, "icu_d_items.csv")

# 1. Load d_items
items = pd.read_csv(items_path)

print("✅ Loaded icu_d_items.csv")
print(f"Total items: {len(items)}")

# 2. Filter candidates for vital signs
keywords = ["temperature", "weight", ]

candidates = items[
    items["label"].str.contains("|".join(keywords), case=False, na=False)
]

print(f"✅ Found {len(candidates)} candidate rows for vitals")

# 3. Show candidate itemids
print("\n--- Candidate Vital ItemIDs ---")
print(candidates[["itemid", "label", "category", "unitname"]].sort_values("label").to_string())

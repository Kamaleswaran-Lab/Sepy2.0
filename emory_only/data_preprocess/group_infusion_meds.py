import pandas as pd
import os
from pathlib import Path


def main():
    root = Path("/labs/collab/K-lab-MODS/MODS-PHI/Emory_Data")
    infusion_meds_mapping = pd.DataFrame(columns = ['formulary_name', 'med_name_generic', 'med_name'])
    INFUSIONS = []
    for year in range(2015, 2022):
        infusions = pd.read_csv(root / f"{year}" / f"CJSEPSIS_INFUSIONMEDS_{year}.dsv", sep = "|")
        print(year, infusions.shape)
        infusions.columns = infusions.columns.str.lower()
        infusions_unique = infusions[['formulary_name', 'med_name_generic', 'med_name']].drop_duplicates()
        INFUSIONS.append(infusions_unique)
    infusion_meds_mapping = pd.concat(INFUSIONS)
    infusion_meds_mapping.to_csv("../groupings/em_infusion_meds_mapping.csv", index=False)

if __name__ == "__main__":
    main()
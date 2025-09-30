### VASOPRESSOR_MEDS File

#### Purpose
The **VASOPRESSOR_MEDS** file standardizes vasopressor infusion records from MIMIC-IV concepts into the format expected by the Sepy2.0 pipeline.  
It captures **drug administration episodes** (start/stop times, dose, unit) for six common vasopressors used in ICU patients.

#### Source Tables
- 🟢 **`norepinephrine.csv`**  
- 🟢 **`epinephrine.csv`**  
- 🟢 **`dopamine.csv`**  
- 🟢 **`dobutamine.csv`**  
- 🟢 **`phenylephrine.csv`**  
- 🟢 **`vasopressin.csv`**  
(from: `/builtdata/csv_concepts_exports/`)  

- 🟡 **`icu_icustays.csv`** (to provide `subject_id` and `hadm_id` from `stay_id`).

#### Processing Logic
1. For each vasopressor concept file:
   - Load CSV.  
   - Standardize columns:
     - `starttime` → `med_order_time`
     - `endtime` → `med_stop`
     - `vaso_rate` → `med_action_dose`
   - Add a new column `drug` with the vasopressor name.  
   - Add constant `med_action_dose_unit = "mcg/kg/min"` (all rates are pre-normalized in MIMIC concepts).  

2. Concatenate all six vasopressor DataFrames.  

3. Merge with 🟡 `icu_icustays.csv` on `stay_id` to add `subject_id` and `hadm_id`.  

4. Add Sepy pipeline–compatible IDs:
   - `csn = hadm_id`  
   - `pat_id = subject_id`

#### Final Columns
| Column Name         | Source / Logic                                         |
|---------------------|--------------------------------------------------------|
| `csn`               | `hadm_id` from 🟡 `icu_icustays.csv`                   |
| `pat_id`            | `subject_id` from 🟡 `icu_icustays.csv`                |
| `stay_id`           | From vasopressor concept tables                        |
| `drug`              | Hard-coded per source file (e.g., norepinephrine)      |
| `med_order_time`    | `starttime`                                            |
| `med_stop`          | `endtime`                                              |
| `med_action_dose`   | `vaso_rate`                                            |
| `med_action_dose_unit` | `"mcg/kg/min"`                                      |

#### Special Notes
- All six vasopressors are stacked into a single file.  
- If multiple vasopressors overlap in time, each is retained separately.  
- Units are standardized to `mcg/kg/min`.  
- `vaso_amount` columns are ignored since rates are sufficient for pipeline use.  

#### Check
- Verify that every row in `df_vasopressor_meds.csv` has a valid `csn` and `pat_id` after merge.  
- Count rows by `drug` and compare with raw concept CSVs to ensure no records were lost.  
- Confirm that all timestamps are properly parsed as datetimes.  


---


### ANTI_INFECTIVE_MEDS File

#### Purpose
The **ANTI_INFECTIVE_MEDS** file standardizes antibiotic administration records from MIMIC-IV concepts into the format expected by the Sepy2.0 pipeline.  
It captures **anti-infective drug exposures** for use in defining *suspicion of infection*.

#### Source Tables
- 🟢 **`antibiotic.csv`** (from `/builtdata/csv_concepts_exports/`).  
  - Contains: `subject_id, hadm_id, stay_id, antibiotic, route, starttime, stoptime`.  

#### Processing Logic
1. Load 🟢 `antibiotic.csv`.  

2. Standardize columns:
   - `antibiotic` → `drug`  
   - `starttime` → `med_order_time`  
   - `stoptime` → `med_stop`  

3. Keep `route` column (provides IV/PO route if available).  

4. Add Sepy pipeline–compatible IDs:
   - `csn = hadm_id`  
   - `pat_id = subject_id`

#### Final Columns
| Column Name      | Source / Logic                                |
|------------------|-----------------------------------------------|
| `csn`            | `hadm_id`                                     |
| `pat_id`         | `subject_id`                                  |
| `stay_id`        | From 🟢 `antibiotic.csv`                       |
| `drug`           | `antibiotic`                                  |
| `route`          | `route`                                       |
| `med_order_time` | `starttime`                                   |
| `med_stop`       | `stoptime`                                    |

#### Special Notes
- This file directly uses the antibiotic concept extraction from MIMIC-IV, so **no grouping file is needed**.  
- Some antibiotics may have missing `stay_id` (if given outside ICU) — these are still retained because `csn` links at admission level.  
- Route is retained for reference but not strictly required for suspicion-of-infection logic.  

#### Check
- Verify that every row has a valid `csn` and `pat_id`.  
- Ensure `med_order_time <= med_stop` (drop or correct if invalid).  
- Check the top antibiotic names (`drug.value_counts()`) to confirm expected distribution (vancomycin, cefepime, etc.).  


---

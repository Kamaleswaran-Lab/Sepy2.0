# infusion_meds

## Background
In the original Sepy2.0 pipeline (`_process_infusion_meds`), built for **Emory/Grady**:
- The single file `INFUSIONMEDS.csv` contained all IV medications (vasopressors + antibiotics).
- The pipeline would:
  1. Import the full file.
  2. Use **grouping files** to separate vasopressors and anti-infectives.
  3. Pivot/unstack to produce `df_vasopressor_meds` and `df_anti_infective_meds`.

## Adaptation for MIMIC
MIMIC-IV does not have a unified `INFUSIONMEDS` file. Instead, it provides **concept tables**:
- `norepinephrine.csv`
- `epinephrine.csv`
- `dopamine.csv`
- `dobutamine.csv`
- `phenylephrine.csv`
- `vasopressin.csv`
- `antibiotic.csv`

### Steps
1. **Preprocessing scripts**  
   - Combine the 6 vasopressor concept files into `df_vasopressor_meds.csv`.  
   - Clean `antibiotic.csv` into `df_anti_infective_meds.csv`.  

   Both outputs follow a unified schema:
   ```text
   csn, pat_id, stay_id, drug, med_order_time, med_stop, med_action_dose, med_action_dose_unit
2. Config changes (**mimic_config.yaml**)
   flatfile_types:
  - ["VASOPRESSOR_MEDS", "df_vasopressor_meds.csv"]
  - ["ANTI_INFECTIVE_MEDS", "df_anti_infective_meds.csv"]
  - ["INFUSIONMEDS", "EMPTY_FILE.csv"]   # dummy placeholder to keep pipeline happy
3. Function modification (`_process_infusion_meds`)
  - No longer depends on grouping files.
  - Calls `_common_import` twice to load vasopressor and antibiotic files.
  - Applies encounter filtering logic (keep only patients in cohort).
  - Produces:
    - `self.df_vasopressor_meds`
    - `self.df_anti_infective_meds`
    - `self.df_infusion_meds = pd.DataFrame()` (kept for compatibility).


# labs

## Background
In the original Sepy2.0 pipeline (`_process_labs`), built for **Emory/Grady**:
- The file `LABS.csv` contained *all* lab results (numeric + string).
- The pipeline would:
  1. Import the full file.
  2. Use a **grouping file** (mapping Emory component IDs → standardized names).
  3. `merge` the grouping with `LABS.csv` to attach a `super_table_col_name`.
  4. `set_index` + `unstack` to pivot labs into wide format.
  5. Separate **numeric labs** (e.g. sodium, potassium) and **string labs** (e.g. COVID PCR result).
  6. Produce `self.df_labs` with standardized columns for supertables.

This worked because Emory’s `component_id` values were already curated and mapped.

---

## Adaptation for MIMIC
MIMIC-IV stores lab definitions in `hosp_d_labitems.csv`, with:
- `itemid` → numeric identifier (like Emory’s `component_id`)
- `label` → descriptive name

### Steps
1. **Build a mapping file**
   - Start from the Emory-required lab list (all the labs expected in the supertable).
   - Manually map each `super_table_col_name` to one or more `label` values in `hosp_d_labitems.csv`.
   - Look up corresponding `itemid` values (from column `itemid`).
   - Produce a new CSV `mimic_mapping_labs_with_id.csv`:
     ```text
     import,super_table_col_name,component,component_id
     Yes,sodium,Sodium,50983
     Yes,sodium,Sodium,52623
     Yes,glucose,Glucose,50809
     ...
     ```
   - `import` is always **Yes**, `component` is the MIMIC label, `component_id` is the MIMIC itemid.

2. **Config changes (`mimic_config.yaml`)**
     flatfile_types:
       - ["LABS", "LABS.csv"]
     grouping_types:
       - ["grouping_labs", "mimic_mapping_labs_with_id.csv"]
3. **Function modification (`_process_labs`)**
    - Instead of matching on **`component`** (string), now the `merge` is done on **`component_id`**.  
    - `component_id` values come from the mapping file prepared in Step 1.  
    - After merging:
      - Add `super_table_col_name` column.  
      - Fill missing `collection_time` (default to `lab_result_time - 1hr`).  
      - Create a **MultiIndex** (patient, csn, lab id, time).  
      - Split labs into **numeric** and **string** subsets.  
      - Convert numeric lab results into floats.  
      - Recombine and align columns with the expected set (`self.config.all_lab_col_names`).  

4. **Output**
    - `self.df_labs` is produced as a **wide-format DataFrame**:
      - **Index** = (csn × timestamps).  
      - **Columns** = standardized lab features (e.g., sodium, potassium, lactate).  
    - This standardized structure feeds directly into the **supertable construction** stage.

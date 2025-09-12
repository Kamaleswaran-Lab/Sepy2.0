# MIMIC Preprocess Doc (Generate Flat Files)

>This folder contains preprocessing scripts and details to convert raw/concepts MIMIC-IV data into the standardized flat files required by the **Sepy2.0 pipeline**. 
This README provides documentation of specific mapping decisions and special handling notes.

## What you need to know
### Input: 
1. 🟡 **Raw CSV files**. (Even though I called it raw, it is still actually derived from PostgreSQL after running the build scripts, which means there is still sort of preprocessing ahead of time.)\
Path: `kamaleswaranlab/mimic_iv/builtdata/csv_exports`
2. 🟢 **Derived CSV files**. These are the files derived from the PostgreSQL derived-tables schemas. These tables are created by running `make-concepts.sql` following the official MIMIC-IV repo.\
Path: `kamaleswaranlab/mimic_iv/builtdata/csv_concepts_exports`
### Output:
Flat files that meet the required format described in [Sepy2.0 repo](https://github.com/Kamaleswaran-Lab/Sepy2.0/blob/main/docs/SepyIMPORT_pipeline.md), which can be input to the pipeline for further processing.\
Path: `kamaleswaranlab/mimic_iv/mimic_flat_files`

## How I generated these flat files for MIMIC-IV? (MIMIC -IV Flat files details)

### ENCOUNTER File

#### Purpose
The ENCOUNTER file standardizes **admission-level records** from MIMIC-IV to match the Emory pipeline specification.  
It captures each inpatient admission (hospital encounter) with demographic, admission/discharge, and ICU stay summary information.

#### Source Tables
- 🟡 **`hosp_admissions.csv`** (admission-level metadata: admission type, admission/discharge times, locations).  
- 🟡 **`hosp_patients.csv`** (static demographics: `anchor_age`).  
- 🟢 **`icustay_detail.csv`** (derived ICU stay detail: length of stay, in/out times).  

#### Processing Logic
1. Load 🟡 `hosp_admissions.csv`, 🟡 `hosp_patients.csv`, and 🟢 `icustay_detail.csv`.  
2. Parse datetime fields (`admittime`, `dischtime`, `edregtime`, `icu_intime`, `icu_outtime`).  
3. Aggregate ICU stays per `hadm_id` to compute:  
   - `total_icu_days` = sum of ICU LOS across all stays.  
   - `first_icu_intime` = earliest ICU admission time.  
   - `last_icu_outtime` = latest ICU discharge time.
     
   Example: If a patient has two ICU stays of 2.0 and 1.0 days → `total_icu_days = 3.0`.

4. Merge **admissions** (main table) with **patients** (to get `anchor_age`).  
5. Left join ICU summary onto admissions (ensures all inpatients are retained).  
6. Construct final output with standardized column names.  

#### Final Columns
| Column Name                   | Source / Logic                                                                 |
|-------------------------------|--------------------------------------------------------------------------------|
| `csn`                         | `hadm_id`; 🟡 `hosp_admissions.csv`                                            |
| `pat_id`                      | `subject_id`; 🟡 `hosp_admissions.csv`                                         |
| `hospital_admission_date_time`| `admittime`; 🟡 `hosp_admissions.csv`                                          |
| `hospital_discharge_date_time`| `dischtime`; 🟡 `hosp_admissions.csv`                                          |
| `ed_presentation_time`        | `edregtime`; 🟡 `hosp_admissions.csv` (may be empty if not through ED)         |
| `encounter_type`              | Hard-coded `"IN"`; applied to all rows                                         |
| `age`                         | `anchor_age`; 🟡 `hosp_patients.csv`                                           |
| `discharge_to`                | `discharge_location`; 🟡 `hosp_admissions.csv`                                 |
| `pre_admit_location`          | `admission_location`; 🟡 `hosp_admissions.csv`                                 |
| `total_icu_days`              | Aggregated `los_icu`; 🟢 `icustay_detail.csv`, 0 if never admitted to ICU      |
| `admit_reason`                | `admission_type`; 🟡 `hosp_admissions.csv` (placeholder for Emory’s field)     |

#### Special Notes
- All **inpatients are retained** (including those without ICU stays).  
- `total_icu_days` is 0 if no ICU stay exists for that admission.  
- `admit_reason` is mapped to `admission_type` because MIMIC does not provide a direct field.  
- File size can be large depending on cohort (hundreds of thousands of admissions).  

#### Check (How I check if there is no issue in the generated flat file)
- Compare the number of unique `csn` in **ENCOUNTER.csv** with the number of unique `hadm_id` in 🟡 `hosp_admissions.csv` — they should match.  
- Verify that all `hadm_id` in admissions exist in ENCOUNTER (`no missing`).  
- Verify that there are no extra `csn` in ENCOUNTER that do not exist in admissions.  


---


### DEMOGRAPHICS File

#### Purpose
The DEMOGRAPHICS file standardizes **patient-level records** from MIMIC-IV to match the Emory pipeline specification.  
It captures static demographic features (sex, race, age) for each patient who has at least one hospital admission.

#### Source Tables
- 🟡 **`hosp_patients.csv`** (patient-level demographics: `subject_id`, `gender`, `anchor_age`).  
- 🟡 **`hosp_admissions.csv`** (admission-level metadata: `race`).  

#### Processing Logic
1. Load 🟡 `hosp_patients.csv` and 🟡 `hosp_admissions.csv`.  
2. Restrict patients to those with at least one admission (intersection with `subject_id` in admissions).  
3. Standardize `gender`:  
   - `"M"` → `"Male"`  
   - `"F"` → `"Female"`  
4. For `race_code`, extract the first non-null `race` per patient from admissions.  
5. For `ethnicity_code`, leave the column empty (MIMIC does not provide de-identified ethnicity codes).  
6. Construct final output with standardized column names.  

#### Final Columns
| Column Name     | Source / Logic                                                                 |
|-----------------|--------------------------------------------------------------------------------|
| `pat_id`        | `subject_id`; 🟡 `hosp_patients.csv`                                           |
| `gender`        | Standardized from `gender`; 🟡 `hosp_patients.csv` (`M/F` → `Male/Female`)     |
| `race_code`     | `race`; 🟡 `hosp_admissions.csv` (first non-null race per patient)             |
| `ethnicity_code`| Empty column (placeholder, not available in MIMIC-IV)                         |

#### Special Notes
- Patients who exist in 🟡 `hosp_patients.csv` but never appear in 🟡 `hosp_admissions.csv` are excluded.  
- If a patient has multiple admissions with different `race`, only the first non-null record is retained.  
- `ethnicity_code` is intentionally left blank because MIMIC-IV does not provide de-identified ethnicity.  
- This file is **patient-level**, while ENCOUNTER is **admission-level**; thus, one patient may map to multiple encounters.  

#### Check (How I check if there is no issue in the generated flat file)
- Verify that all `pat_id` in **DEMOGRAPHICS.csv** also exist in **ENCOUNTER.csv**.  
- Check that the counts of unique patients match between DEMOGRAPHICS and ENCOUNTER.  
- Save mismatch lists for patients found only in DEMOGRAPHICS or only in ENCOUNTER.  



---


### INFUSIONMEDS File
SKIP FOR NOW

---


### LABS File


---


## LABS

### Purpose
The **LABS** file consolidates laboratory test results from MIMIC-IV and their metadata into the schema required by the Sepy 2.0 pipeline.

### Source Tables
- `hosp_labevents.csv` — raw laboratory test results  
- `hosp_d_labitems.csv` — metadata defining lab test categories and labels  

### Processing Logic
- Data is read in chunks (1M rows at a time) for memory efficiency. (Unable to load if read all at once)
- Each chunk is left-joined with `hosp_d_labitems.csv` on `itemid` to enrich results.  
- `lab_result` prefers `valuenum` if available, else falls back to `value`. (**valueuom is not recorded!**)
- `collection_time` is set to `charttime` (MIMIC-IV does not provide separately).  
- `result_status` is hardcoded as `"Final"`.  
- Rows missing `pat_id` or `component_id` are dropped. (**Some patients do not have the csn, which means they are not inpatients. If you are not interested in these patients' data, just drop them in the code.**) 
- Output is appended incrementally to `LABS.csv`.  

### Final Columns
| Column Name       | Source / Logic                                                           |
|-------------------|---------------------------------------------------------------------------|
| `csn`             | Encounter identifier (`hadm_id`)                                          |
| `pat_id`          | Patient identifier (`subject_id`)                                         |
| `component_id`    | Lab test identifier (`itemid`)                                            |
| `lab_result`      | Numeric value (`valuenum`), fallback to string result (`value`)           |
| `lab_result_time` | Result timestamp (`charttime`)                                            |
| `collection_time` | Same as `charttime`                                                       |
| `result_status`   | `"Final"` (not tracked in MIMIC)                                          |
| `proc_cat_id`     | Lab category (`category` from `hosp_d_labitems`)                          |
| `proc_cat_name`   | Same as `proc_cat_id`                                                     |
| `proc_code`       | Proxy code (`itemid`)                                                     |
| `proc_desc`       | Lab test label (`label` from `hosp_d_labitems`)                           |
| `component`       | Same as `proc_desc`                                                       |
| `loinc_code`      | Not provided in MIMIC-IV; set to `None`                                   |

### Notes
- File size is large (~10GB uncompressed).  
- Some labs lack `csn` (`hadm_id`); these are retained unless explicitly dropped.  
- **LOINC codes are not available in MIMIC-IV; external mapping required if needed.**  


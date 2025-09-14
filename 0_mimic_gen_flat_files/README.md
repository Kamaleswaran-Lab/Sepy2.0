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

#### Purpose
The LABS file standardizes **laboratory test results** from MIMIC-IV to match the Emory pipeline specification.  
It captures all laboratory events with identifiers, result values, specimen type, and associated metadata.

#### Source Tables
- 🟡 **`hosp_labevents.csv`** (individual lab results with subject, hadm, itemid, timestamps, values).  
- 🟡 **`hosp_d_labitems.csv`** (lab item dictionary providing `label`, `fluid`, `category`, and metadata).  

#### Processing Logic
1. Load 🟡 `hosp_labevents.csv` in chunks for efficiency.  
2. Merge with 🟡 `hosp_d_labitems.csv` on `itemid` to add lab metadata.  
3. Construct final standardized columns:  
   - `csn` = `hadm_id`  
   - `pat_id` = `subject_id`  
   - `component_id` = `itemid`  
   - `lab_result` = `valuenum` if available, otherwise `value`  
   - `lab_result_time` = `charttime`  
   - `collection_time` = `charttime` (MIMIC does not distinguish collection vs. result time)  
   - `result_status` = hard-coded `"Final"`  
   - `proc_cat_id` = `itemid` (placeholder; pipeline will map with grouping files if needed)  
   - `proc_cat_name` = `fluid` (specimen type: Blood, Urine, CSF, etc.)  
   - `proc_code` = `itemid`  
   - `proc_desc` = `category` (lab discipline: Chemistry, Hematology, Blood Gas, etc.)  
   - `component` = `label` (lab test name)  
   - `loinc_code` = left empty (MIMIC does not provide direct mapping in d_labitems)  

4. Drop rows missing `pat_id`, `csn`, or `component_id`.  
5. Write out results incrementally to **LABS.csv** in append mode.

#### Final Columns
| Column Name      | Source / Logic                                                                 |
|------------------|--------------------------------------------------------------------------------|
| `csn`            | `hadm_id`; 🟡 `hosp_labevents.csv`                                             |
| `pat_id`         | `subject_id`; 🟡 `hosp_labevents.csv`                                          |
| `component_id`   | `itemid`; 🟡 `hosp_labevents.csv`                                              |
| `lab_result`     | `valuenum` if not null, else `value`; 🟡 `hosp_labevents.csv`                  |
| `lab_result_time`| `charttime`; 🟡 `hosp_labevents.csv`                                           |
| `collection_time`| `charttime`; 🟡 `hosp_labevents.csv`                                           |
| `result_status`  | Hard-coded `"Final"`                                                          |
| `proc_cat_id`    | `itemid`; 🟡 `hosp_labevents.csv`                                              |
| `proc_cat_name`  | `fluid`; 🟡 `hosp_d_labitems.csv`                                              |
| `proc_code`      | `itemid`; 🟡 `hosp_labevents.csv`                                              |
| `proc_desc`      | `category`; 🟡 `hosp_d_labitems.csv`                                           |
| `component`      | `label`; 🟡 `hosp_d_labitems.csv`                                              |
| `loinc_code`     | Empty placeholder (not directly provided in MIMIC-IV)                         |

#### Special Notes
- `lab_result_time` and `collection_time` are both set to `charttime`, as MIMIC does not explicitly store collection timestamps separately.  
- `result_status` is hard-coded to `"Final"` since MIMIC does not store result status metadata.
- `proc_cat_id` and `proc_code` are both set as `itemid` because there is no specification in MIMIC. Please see [here](https://mimic.mit.edu/docs/iv/modules/hosp/d_labitems/#links-to:~:text=All%20data%20in%20labevents%20link%20to%20the%20d_labitems%20table.%20Each%20unique%20(fluid%2C%20category%2C%20label)%20tuple%20in%20the%20hospital%20database%20was%20assigned%20an%20itemid%20in%20this%20table%2C%20and%20the%20use%20of%20this%20itemid%20facilitates%20efficient%20storage%20and%20querying%20of%20the%20data) for what `itemid` means in MIMIC.
- `loinc_code` left blank unless a separate mapping file is introduced.  
- File size is large; script processes data in 1M-row chunks to manage memory efficiently.
  

---

### VITALS File

---

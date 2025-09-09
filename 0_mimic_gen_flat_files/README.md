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




















## ENCOUNTER File

1. **Encounter Type (`encounter_type`)**
   - In Emory, `encounter_type` distinguishes **ER / IN / OV**.
   - In MIMIC, all patients are inpatients (ICU/hospitalized).
   - ✅ Decision: **Set all `encounter_type` = "IN"**.

2. **Admit Reason (`admit_reason`)**
   - Emory provides explicit admit reasons.
   - MIMIC does not contain a direct field for this.
   - ✅ Decision: Use MIMIC’s `admission_type` as a **placeholder** for `admit_reason`.

3. **ICU Stays (`total_icu_days`)**
   - In MIMIC, a hospital admission (`hadm_id`) may include **multiple ICU stays**.
   - For each encounter:
     - `total_icu_days` = sum of all `los` values across ICU stays.
     - `first_icu_intime` = earliest `intime`.
     - `last_icu_outtime` = latest `outtime`.
   - ✅ Example: If a patient has two ICU stays of 2.0 and 1.0 days → `total_icu_days = 3.0`.

---


## DEMOGRAPHICS File

### Source Tables
- **MIMIC-IV `hosp_patients.csv`**  
  Provides static demographic information for all patients (`subject_id`, gender, anchor_age, etc.).

- **MIMIC-IV `hosp_admissions.csv`**  
  Provides encounter-level information. We use this table to obtain each patient’s `race` field, because race is not available in `hosp_patients.csv`.

---

### Column Mapping

| Required Column   | Source (MIMIC-IV)   | Notes |
|-------------------|---------------------|-------|
| `pat_id`          | `subject_id`        | Unique patient identifier |
| `gender`          | `gender`            | Converted from `M/F` → `Male/Female` |
| `race_code`       | `admissions.race`   | Mapped to anonymized numeric codes (see below) |
| `ethnicity_code`  | Derived from `race` | Hispanic vs Non-Hispanic categorization |

---

### Gender Mapping

- MIMIC-IV provides gender as single letters:  
  - `M` → `Male`  
  - `F` → `Female`  

- All values are converted to the full words **Male** / **Female** to match Sepy2.0 requirements.

---

### Race Mapping

MIMIC-IV `admissions.race` contains a wide range of values:  

AMERICAN INDIAN/ALASKA NATIVE ASIAN ASIAN - ASIAN INDIAN ASIAN - CHINESE ASIAN - KOREAN ASIAN - SOUTH EAST ASIAN BLACK/AFRICAN BLACK/AFRICAN AMERICAN BLACK/CAPE VERDEAN BLACK/CARIBBEAN ISLAND HISPANIC OR LATINO HISPANIC/LATINO - CENTRAL AMERICAN HISPANIC/LATINO - COLUMBIAN HISPANIC/LATINO - CUBAN HISPANIC/LATINO - DOMINICAN HISPANIC/LATINO - GUATEMALAN HISPANIC/LATINO - HONDURAN HISPANIC/LATINO - MEXICAN HISPANIC/LATINO - PUERTO RICAN HISPANIC/LATINO - SALVADORAN MULTIPLE RACE/ETHNICITY NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER OTHER PATIENT DECLINED TO ANSWER PORTUGUESE SOUTH AMERICAN UNABLE TO OBTAIN UNKNOWN WHITE WHITE - BRAZILIAN WHITE - EASTERN EUROPEAN WHITE - OTHER EUROPEAN WHITE - RUSSIAN


These values were **collapsed into major categories** and mapped to integer codes:

| Race Category (MIMIC)                                    | Mapped `race_code` |
|----------------------------------------------------------|--------------------|
| WHITE, WHITE - * (subgroups)                             | 309315 |
| BLACK/AFRICAN AMERICAN, BLACK - * (subgroups)            | 309316 |
| ASIAN, ASIAN - * (subgroups)                             | 309317 |
| HISPANIC OR LATINO, HISPANIC/LATINO - * (subgroups)      | 309318 |
| AMERICAN INDIAN/ALASKA NATIVE                            | 309319 |
| NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER                | 309320 |
| MULTIPLE RACE/ETHNICITY                                  | 309321 |
| OTHER, PORTUGUESE, SOUTH AMERICAN                        | 309321 |
| PATIENT DECLINED TO ANSWER, UNABLE TO OBTAIN, UNKNOWN    | 309322 |

---

### Ethnicity Mapping

Ethnicity is not explicitly provided in MIMIC-IV. We derived it from the `race` field:

- If `race` contains `"HISPANIC"` → `ethnicity_code = 312507` (Hispanic)  
- Otherwise → `ethnicity_code = 312508` (Non-Hispanic)

---

### Patients vs Admissions

- `patients.csv` contains **all patients** in MIMIC-IV.  
- `admissions.csv` contains **only those patients who had a hospital admission** (each row is a `hadm_id`).  
- Therefore:  
  - Some patients in `patients.csv` **do not appear in `admissions.csv`**.  
  - This is expected, as these patients may never have been admitted.  
  - When building DEMOGRAPHICS, we join `patients` with the first available `race` entry from `admissions`.  
  - Patients without any admission will still appear in DEMOGRAPHICS but default to `race_code = 309322 (Unknown)` and `ethnicity_code = 312508 (Non-Hispanic)`.

---

### Example Output

| pat_id | gender | race_code | ethnicity_code |
|--------|--------|-----------|----------------|
| 100001 | Male   | 309315    | 312508 |
| 100002 | Female | 309316    | 312508 |
| 100003 | Male   | 309318    | 312507 |
| 100004 | Female | 309322    | 312508 |


## INFUSIONMEDS File

### Source
- **MIMIC-IV `icu_inputevents.csv`** (order-level infusion events).  
- ⚠️ Not using `icu_ingredientevents` (ingredient-level, which may include more detailed ingredients, etc., Calories).
- It is unclear whether Emory expects order-level or ingredient-level; this is left for future work.

---

### Mapping

| Required Column       | Source (MIMIC-IV) | Notes |
|-----------------------|-------------------|-------|
| `csn`                 | `hadm_id`         | Encounter ID |
| `pat_id`              | `subject_id`      | Patient ID |
| `medication_id`       | `itemid`          | Will later map to grouping file |
| `med_order_time`      | `starttime`       | Used as order time |
| `med_action_time`     | `endtime`         | Used as action/stop time |
| `med_start`           | `starttime`       | Infusion start |
| `med_stop`            | `endtime`         | Infusion end |
| `med_order_route`     | `"IV"`            | All assumed IV |
| `med_action_dose`     | `amount`          | Infused dose |
| `med_action_dose_unit`| `amountuom`       | Units |

---

### Key Decisions
1. All infusions treated as **IV**.  
2. Use **order-level** (`inputevents`), not ingredient-level.  
3. Future work: confirm Emory’s expected level.



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


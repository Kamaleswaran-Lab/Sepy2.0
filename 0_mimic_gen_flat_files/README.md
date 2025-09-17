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

#### Purpose
The VITALS file standardizes **physiological vital signs** from MIMIC-IV to match the Emory pipeline specification.  
It integrates measurements from multiple sources (Concepts tables, ICU bedside charted values, and outpatient/ward OMR data) into a single flat file.  
This file captures blood pressure, heart rate, respiratory rate, oxygenation parameters, body temperature, weight, and height — all aligned by patient encounter (`csn`) and timestamp (`recorded_time`).

#### Source Tables
- 🟢 **`csv_concepts_exports/vitalsign.csv`** (core vital signs including heart rate, blood pressure, temperature, SpO₂).  
- 🟢 **`csv_concepts_exports/oxygen_delivery.csv`** (oxygen device and oxygen flow).  
- 🟡 **`icu_chartevents.csv`** + **`icu_d_items.csv`** (bedside measurements: CVP, EtCO₂, weights, height).  
- 🟡 **`hosp_omr.csv`** (outpatient/ward observations: blood pressure, weights, height).  
- 🟡 **`icu_icustays.csv`** (mapping from `stay_id` to `hadm_id`, used to assign `csn`).  

#### Processing Logic
1. **Load concept-level vitalsign data** (`vitalsign.csv`).  
   - Rename columns to standardized names:  
     - `heart_rate` → `pulse`  
     - `resp_rate` → `unassisted_resp_rate`  
     - `spo2` → `spo2`  
     - `temperature` → `temperature`  
     - `temperature_site` → `temproute`  
     - `sbp, dbp, mbp` → `sbp_line, dbp_line, map_line`  
     - `sbp_ni, dbp_ni, mbp_ni` → `sbp_cuff, dbp_cuff, map_cuff`  
   - Extract columns: `subject_id`, `stay_id`, `charttime`, and the mapped variables.  

2. **Load oxygen delivery data** (`oxygen_delivery.csv`).  
   - Map `o2_delivery_device_1` → `o2_device`  
   - Map `o2_flow` → `o2_flow_rate`  
   - Extract: `subject_id`, `stay_id`, `charttime`, `o2_device`, `o2_flow_rate`.  

3. **Merge vitalsign + oxygen**.  
   - Merge on (`subject_id`, `stay_id`, `charttime`) with `outer` join.  
   - Ensures all measurement points are retained even if only one source has data.  

4. **Extract ICU charted measurements** from `icu_chartevents`.  
   - Filter for relevant `itemid` values (via `icu_d_items`):  
     - `224639` Daily Weight  
     - `226512` Admission Weight (Kg)  
     - `226531` Admission Weight (lbs.)  
     - `226730` Height (cm)  
     - `220074` Central Venous Pressure (CVP)  
     - `228640` End-tidal CO₂ (EtCO₂)  
   - Pivot to wide format with index = (`subject_id`, `stay_id`, `charttime`), columns = measurement labels.  
   - Convert Admission Weight (lbs.) to kg using factor 0.4536.  
   - Merge the three weight-related columns into one unified `daily_weight_kg`:  
     - Prefer `Daily Weight`, if missing then `Admission Weight (Kg)`, otherwise `Admission Weight (lbs.)` (after conversion).  
   - Rename:  
     - `"Central Venous Pressure"` → `cvp`  
     - `"EtCO2"` → `end_tidal_co2`  
     - `"Height (cm)"` → `height_cm`  
   - Drop redundant intermediate weight columns.  
   - Merge back into main vitals table.  

5. **Process OMR data** (`hosp_omr.csv`).  
   - **Blood pressure**: split `"110/70"` string into `sbp_cuff`, `dbp_cuff`, and compute `map_cuff = (sbp + 2*dbp)/3`.  
   - **Weight**: convert weight in lbs to kg (`*0.453592`), store as `daily_weight_kg`.  
   - **Height**: convert height in inches to cm (`*2.54`), store as `height_cm`.  
   - Normalize OMR timestamps:  
     - Convert `chartdate` (date-only) into `charttime` at midnight (`00:00:00`).  
     - Ensures compatibility with the hourly-resolution vitals timeline.  

6. **Merge OMR data into vitals**.  
   - Merge OMR blood pressure, weight, and height separately using (`subject_id`, `charttime`).  
   - For each measurement type, use `combine_first`:  
     - Prefer ICU/Concepts values if present.  
     - If ICU/Concepts is null, fill with OMR values.  
   - Drop intermediate `_omr` columns after merging.  
   - Result: one clean column per measurement, enriched by OMR where needed.  

7. **Add hospital admission ID (`csn`)**.  
   - Merge with `icu_icustays.csv` to map `stay_id` → `hadm_id`.  
   - Rename `hadm_id` → `csn`, `subject_id` → `pat_id`, `charttime` → `recorded_time`.  

8. **Finalize table**.  
   - Drop `stay_id` (not needed in downstream pipeline).  
   - Reorder columns to match Emory pipeline specification:  


#### Final Columns
| Column Name           | Source / Logic                                                                 |
|------------------------|-----------------------------------------------------------------------------------------------------|
| `pat_id`               | 🟢 `subject_id` from **`vitalsign.csv`**; also present in 🟡 `icu_chartevents.csv`, 🟡 `hosp_omr.csv` |
| `csn`                  | 🟡 `hadm_id` from **`icu_icustays.csv`** (via `stay_id` mapping)                                    |
| `recorded_time`        | 🟢 `charttime` from **`vitalsign.csv`**; aligned with 🟢 oxygen, 🟡 chartevents, 🟡 OMR (date floored to 00:00:00) |
| `temperature`          | 🟢 `temperature` from **`vitalsign.csv`**                                                           |
| `temproute`            | 🟢 `temperature_site` from **`vitalsign.csv`**                                                      |
| `daily_weight_kg`      | 🟡 `icu_chartevents.csv` + **`icu_d_items.csv`**: itemid 224639 (`Daily Weight`), 226512 (`Admission Weight (Kg)`), 226531 (`Admission Weight (lbs.)`, converted to kg); fallback from 🟡 **`hosp_omr.csv`** weight entries (lbs converted to kg) |
| `height_cm`            | 🟡 `icu_chartevents.csv` + **`icu_d_items.csv`**: itemid 226730 (`Height (cm)`); fallback from 🟡 **`hosp_omr.csv`** height entries (inches converted to cm) |
| `sbp_line`             | 🟢 `sbp` (arterial line systolic BP) from **`vitalsign.csv`**                                       |
| `dbp_line`             | 🟢 `dbp` (arterial line diastolic BP) from **`vitalsign.csv`**                                      |
| `map_line`             | 🟢 `mbp` (arterial line mean BP) from **`vitalsign.csv`**                                           |
| `sbp_cuff`             | 🟢 `sbp_ni` from **`vitalsign.csv`**; fallback from 🟡 **`hosp_omr.csv`** blood pressure string parsing (first value before “/”) |
| `dbp_cuff`             | 🟢 `dbp_ni` from **`vitalsign.csv`**; fallback from 🟡 **`hosp_omr.csv`** blood pressure string parsing (second value after “/”) |
| `map_cuff`             | 🟢 `mbp_ni` from **`vitalsign.csv`**; fallback computed from 🟡 **`hosp_omr.csv`** values: `(sbp_cuff + 2*dbp_cuff)/3` |
| `pulse`                | 🟢 `heart_rate` from **`vitalsign.csv`**                                                            |
| `unassisted_resp_rate` | 🟢 `resp_rate` from **`vitalsign.csv`**                                                             |
| `spo2`                 | 🟢 `spo2` from **`vitalsign.csv`**                                                                  |
| `o2_device`            | 🟢 `o2_delivery_device_1` from **`oxygen_delivery.csv`**                                            |
| `cvp`                  | 🟡 `icu_chartevents.csv` + **`icu_d_items.csv`**: itemid 220074 (`Central Venous Pressure`)          |
| `end_tidal_co2`        | 🟡 `icu_chartevents.csv` + **`icu_d_items.csv`**: itemid 228640 (`EtCO₂`)                            |
| `o2_flow_rate`         | 🟢 `o2_flow` from **`oxygen_delivery.csv`**                                                         |


#### Special Notes
- Multiple input sources exist for weights, heights, and cuff blood pressures. The pipeline standardizes by merging into a single column per measurement using `combine_first`.  
- OMR data is less frequent and often only recorded once per admission (admission weight, admission BP, admission height). By normalizing OMR chartdate to midnight and merging, these values backfill ICU timelines when ICU/Concepts data is absent.  
- ICU `charttime` is more granular (hourly), while OMR is daily. This discrepancy means most OMR values will only align with midnight rows in the time index.  
- File size is substantial due to the hourly time resolution across all admissions.  


---


### VENT File

#### Purpose
The VENT file standardizes **mechanical ventilation data** from MIMIC-IV to match the Emory pipeline specification.  
It combines session-level ventilation intervals with detailed ventilator settings, providing a timeline of ventilator support aligned by patient (`pat_id`), hospital admission (`csn`), and time (`recorded_time`).  
This file captures information on ventilator mode, device type, oxygenation, pressures, and tidal volumes.

#### Source Tables
- 🟢 **`csv_concepts_exports/ventilation.csv`** (session intervals with start/stop times and ventilation status).  
- 🟢 **`csv_concepts_exports/ventilator_setting.csv`** (hourly ventilator settings: rate, tidal volumes, PEEP, FiO₂, mode).  
- 🟡 **`icu_icustays.csv`** (maps `stay_id` → `hadm_id` and `subject_id`, required for `csn` and `pat_id`).  

#### Processing Logic
1. **Process ventilation intervals** (`ventilation.csv`).  
   - Rename `starttime` → `vent_start_time`, `endtime` → `vent_stop_time`, `ventilation_status` → `vent_cat`.  
   - Merge with 🟡 `icu_icustays.csv` on `stay_id` to add `pat_id` (subject_id) and `csn` (hadm_id).  
   - Ensure datetime parsing for `vent_start_time` and `vent_stop_time`.  

2. **Process ventilator settings** (`ventilator_setting.csv`).  
   - Standardize column names:  
     - `charttime` → `recorded_time`  
     - `ventilator_type` → `vent_name`  
   - Extract columns:  
     - `vent_name`, `ventilator_mode`, `ventilator_mode_hamilton`  
     - `respiratory_rate_set`, `tidal_volume_set`, `tidal_volume_observed`, `tidal_volume_spontaneous`  
     - `peep`, `fio2`  
   - Merge `ventilator_mode` and `ventilator_mode_hamilton` into unified `vent_mode`.  

3. **Merge intervals with settings**.  
   - Join on `stay_id`.  
   - Keep only rows where `recorded_time` falls between `vent_start_time` and `vent_stop_time`.  (⚠️ **Do we want to do this?**)
   - Assign the session-level `vent_cat` to each aligned row.  

4. **Add placeholder for exhaled tidal volume**.  
   - Since it is unclear whether `tidal_volume_observed` or `tidal_volume_spontaneous` should serve as the canonical value, retain both.  
   - Add a new column `vent_tidal_rate_exhaled`, filled with `"Not Yet Decided"`.  

5. **Finalize table**.  
   - Rename fields:  
     - `respiratory_rate_set` → `vent_rate_set`  
     - `tidal_volume_set` → `vent_tidal_rate_set`  
   - Select and order columns to match pipeline specification.  
   - Save to **VENT.csv**.  

#### Final Columns
| Column Name               | Source / Logic                                                                 |
|----------------------------|--------------------------------------------------------------------------------|
| `csn`                      | 🟡 `hadm_id` from **`icu_icustays.csv`** (via `stay_id`)                       |
| `pat_id`                   | 🟡 `subject_id` from **`icu_icustays.csv`** (via `stay_id`)                    |
| `vent_rate_set`            | 🟢 `respiratory_rate_set` from **`ventilator_setting.csv`**                    |
| `vent_tidal_rate_set`      | 🟢 `tidal_volume_set` from **`ventilator_setting.csv`**                        |
| `tidal_volume_observed`    | 🟢 `tidal_volume_observed` from **`ventilator_setting.csv`**                    |
| `tidal_volume_spontaneous` | 🟢 `tidal_volume_spontaneous` from **`ventilator_setting.csv`**                  |
| `vent_tidal_rate_exhaled`  | Placeholder column, filled with `"Not Yet Decided"` (⚠️ **We need to know whether it is `tidal_volume_observed` or `tidal_volume_spontaneous`**)                            |
| `peep`                     | 🟢 `peep` from **`ventilator_setting.csv`**                                    |
| `fio2`                     | 🟢 `fio2` from **`ventilator_setting.csv`**                                    |
| `recorded_time`            | 🟢 `charttime` from **`ventilator_setting.csv`**, restricted to interval times |
| `vent_start_time`          | 🟢 `starttime` from **`ventilation.csv`**                                      |
| `vent_stop_time`           | 🟢 `endtime` from **`ventilation.csv`**                                        |
| `vent_name`                | 🟢 `ventilator_type` from **`ventilator_setting.csv`**                         |
| `vent_mode`                | Combined from `ventilator_mode` and `ventilator_mode_hamilton` (prefer non-null) |
| `vent_cat`                 | 🟢 `ventilation_status` from **`ventilation.csv`**, mapped as session-level category |

#### Special Notes
- `vent_cat` provides a high-level category of ventilation status (e.g., invasive vs. non-invasive) sourced from `ventilation.csv`.  
- Both `tidal_volume_observed` and `tidal_volume_spontaneous` are retained due to ambiguity.  
- The derived column `vent_tidal_rate_exhaled` is deliberately left undecided, pending clinical consensus.  
- Timestamps (`recorded_time`) are aligned with intervals (`vent_start_time`–`vent_stop_time`) to ensure rows only represent valid ventilator support periods. (⚠️ **Do we want to do this?**)
- Data volume is large since ventilator settings are often recorded minute-to-minute or hourly.  


---

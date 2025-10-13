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

#### Source Tables
- 🟡 **`hosp_admissions.csv`** (admission-level metadata: admission type, admission/discharge times, locations).  
- 🟡 **`hosp_patients.csv`** (static demographics: `anchor_age`).  

#### Processing Logic
1. Load 🟡 `hosp_admissions.csv`, 🟡 `hosp_patients.csv`.  
2. Parse datetime fields (`admittime`, `dischtime`, `edregtime`, `deathtime`).  
3. Merge **admissions** (main table) with **patients** (to get `anchor_age`).  
4. Construct final output with standardized column names.  

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
| `deathtime`                   | `deathtime`; 🟡 `hosp_admissions.csv`                                          |
| `insurance`                   | `insurance`; 🟡 `hosp_admissions.csv`                                             |
| `marital_status`              | `marital_status`; 🟡 `hosp_admissions.csv`                                           |
| `admission_type`              | `admission_type`; 🟡 `hosp_admissions.csv`                                           |


#### Special Notes
- All **inpatients are retained** (including those without ICU stays).  
- We don't have `total_icu_days` and `admit_reason` in MIMIC's flatfile (compared to Emory's). 

---


### DEMOGRAPHICS File

#### Purpose
The DEMOGRAPHICS file standardizes **patient-level records** from MIMIC-IV to match the Emory pipeline specification.  
It captures static demographic features (sex, race, age) for each patient.

#### Source Tables
- 🟡 **`hosp_patients.csv`** (patient-level demographics: `subject_id`, `gender`, `anchor_age`).  
- 🟡 **`hosp_admissions.csv`** (admission-level metadata: `race`).  

#### Processing Logic
1. Load 🟡 `hosp_patients.csv` and 🟡 `hosp_admissions.csv`.  
2. Standardize `gender`:  
   - `"M"` → `"Male"`  
   - `"F"` → `"Female"`  
3. We do not have `ethnicity_code` in MIMIC.

#### Final Columns
| Column Name     | Source / Logic                                                                 |
|-----------------|--------------------------------------------------------------------------------|
| `pat_id`        | `subject_id`; 🟡 `hosp_patients.csv`                                           |
| `gender`        | Standardized from `gender`; 🟡 `hosp_patients.csv` (`M/F` → `Male/Female`)     |
| `race`           | `race`; 🟡 `hosp_admissions.csv` (first non-null race per patient)             |

#### Special Notes
- If a patient has multiple admissions with different `race`, only the first non-null record is retained.  
- This file is **patient-level**, while ENCOUNTER is **admission-level**; thus, one patient may map to multiple encounters.  

---

### INFUSIONMEDS File

We directly use related files in the MIMIC Concetps Folder.

#### Source Tables
- 🟢 **`vasoactive_agent.csv`** (contains common ICU vasopressors).  
- 🟢 **`antibiotic.csv`** (contains antibiotic administration times and metadata).  
- 🟢 **`icu_icustays.csv`** (used to map `stay_id` → `pat_id`, `csn`).

---

#### ▶️ Vasopressor

##### Processing Logic
1. Load 🟢 `vasoactive_agent.csv`.  
2. Add fixed dose units:
   - `"mcg/kg/min"` for all drugs except vasopressin, which uses `"units/hour"`.
3. Merge with 🟢 `icu_icustays.csv` to attach `pat_id`, `csn` via `stay_id`.  
4. Rename time columns:  
   - `starttime` → `med_start`  
   - `endtime` → `med_stop`  
5. Reorder and select columns as per downstream pipeline needs.  
6. Output file:  
   **`mimic_flat_files/INFUSIONMEDS/df_vasopressor_meds.csv`**

##### Final Columns
| Column Name              | Source / Logic                                                            |
|--------------------------|---------------------------------------------------------------------------|
| `pat_id`                 | `subject_id`; 🟢 `icu_icustays.csv`                                        |
| `csn`                    | `hadm_id`; 🟢 `icu_icustays.csv`                                           |
| `stay_id`                | ICU stay identifier; 🟢 `vasoactive_agent.csv`                             |
| `med_start`              | `starttime`; 🟢 `vasoactive_agent.csv`                                     |
| `med_stop`               | `endtime`; 🟢 `vasoactive_agent.csv`                                       |
| `vasopressin`            | dose in `units/hour`; 🟢 `vasoactive_agent.csv`                            |
| `dopamine`               | dose in `mcg/kg/min`; 🟢 `vasoactive_agent.csv`                            |
| `epinephrine`            | dose in `mcg/kg/min`; 🟢 `vasoactive_agent.csv`                            |
| `norepinephrine`         | dose in `mcg/kg/min`; 🟢 `vasoactive_agent.csv`                            |
| `phenylephrine`          | dose in `mcg/kg/min`; 🟢 `vasoactive_agent.csv`                            |
| `dobutamine`             | dose in `mcg/kg/min`; 🟢 `vasoactive_agent.csv`                            |
| `milrinone`              | dose in `mcg/kg/min`; 🟢 `vasoactive_agent.csv`                            |
| `vasopressin_dose_unit`  | fixed string `"units/hour"`                                               |
| `dopamine_dose_unit`     | fixed string `"mcg/kg/min"`                                               |
| `epinephrine_dose_unit`  | fixed string `"mcg/kg/min"`                                               |
| `norepinephrine_dose_unit`| fixed string `"mcg/kg/min"`                                              |
| `phenylephrine_dose_unit`| fixed string `"mcg/kg/min"`                                               |
| `dobutamine_dose_unit`   | fixed string `"mcg/kg/min"`                                               |
| `milrinone_dose_unit`    | fixed string `"mcg/kg/min"`                                               |

##### Special Notes
- This flatfile contains only **vasoactive agents**, not all infusion meds.  
- The `dose_unit` columns are required by Sepy to compute weight-adjusted vasopressor dosing.
- We don't have med_order_time and med_action_time in MIMIC.

---

#### ▶️ Anti-Infective

##### Processing Logic
1. Load 🟢 `antibiotic.csv` (from MIMIC concept exports).  
2. Rename key columns:
   - `subject_id` → `pat_id`  
   - `hadm_id` → `csn`  
   - `starttime` → `med_start`  
   - `stoptime` → `med_stop`  
   - `route` → `med_order_route`  
3. Select relevant columns.  
4. Output file:  
   **`mimic_flat_files/INFUSIONMEDS/df_anti_infective_meds.csv`**

##### Final Columns
| Column Name         | Source / Logic                                                        |
|---------------------|------------------------------------------------------------------------|
| `csn`               | `hadm_id`; 🟢 `antibiotic.csv`                                          |
| `pat_id`            | `subject_id`; 🟢 `antibiotic.csv`                                       |
| `stay_id`           | ICU stay ID if available; 🟢 `antibiotic.csv`                           |
| `antibiotic`        | drug name; 🟢 `antibiotic.csv`                                          |
| `med_start`         | `starttime`; 🟢 `antibiotic.csv`                                        |
| `med_stop`          | `stoptime`; 🟢 `antibiotic.csv`                                         |
| `med_order_route`   | route of administration (e.g., IV, PO); 🟢 `antibiotic.csv`             |

##### Special Notes
- All antibiotic records are retained — no filtering is done by route or duration.  
- This file supports **infection onset** and **t_suspicion** logic in Sepy 2.0.

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


### DIALYSIS File

#### Purpose
The DIALYSIS file standardizes **renal replacement therapy (RRT) events** from MIMIC-IV to match the Emory pipeline specification.  
It captures patient-level dialysis events aligned by encounter (`csn`) and patient (`pat_id`), with the corresponding timestamps and dialysis modality.

#### Source Tables
- 🟢 **`csv_concepts_exports/rrt.csv`** (concept table that aggregates dialysis-related events, marking presence, activity, and type).  
- 🟡 **`icu_icustays.csv`** (mapping from `stay_id` → `hadm_id` and `subject_id`, required for `csn` and `pat_id`).  

#### Processing Logic
1. **Load RRT events** (`rrt.csv`).  
   - Extract dialysis event records containing `stay_id`, `charttime`, `dialysis_present`, `dialysis_active`, and `dialysis_type`.  

2. **Map ICU stay to admission and patient** (`icu_icustays.csv`).  
   - Merge `rrt.csv` with `icu_icustays.csv` on `stay_id`.  
   - This adds `subject_id` (patient identifier) and `hadm_id` (hospital admission identifier).  

3. **Standardize field names**.  
   - `subject_id` → `pat_id`  
   - `hadm_id` → `csn`  
   - `charttime` → `service_timestamp`  

4. **Select final columns**.  
   - Keep only standardized identifiers and dialysis event attributes:  
     - `csn`, `pat_id`, `service_timestamp`, `dialysis_present`, `dialysis_active`, `dialysis_type`.  

5. **Save output**.  
   - Export to **DIALYSIS.csv** under the `mimic_flat_files` directory.  

#### Final Columns
| Column Name         | Source / Logic                                                                 |
|----------------------|--------------------------------------------------------------------------------|
| `csn`               | 🟡 `hadm_id` from **`icu_icustays.csv`** (via `stay_id`)                       |
| `pat_id`            | 🟡 `subject_id` from **`icu_icustays.csv`** (via `stay_id`)                    |
| `service_timestamp` | 🟢 `charttime` from **`rrt.csv`**, indicating when the dialysis event was charted |
| `dialysis_present`  | 🟢 `dialysis_present` flag from **`rrt.csv`** (1 if dialysis present at this time) |
| `dialysis_active`   | 🟢 `dialysis_active` flag from **`rrt.csv`** (1 if actively receiving dialysis) |
| `dialysis_type`     | 🟢 `dialysis_type` from **`rrt.csv`** (e.g., IHD, CRRT, SLED)                  |

#### Special Notes
- Each row represents one dialysis-related event at a given timestamp (`service_timestamp`).  
- A single patient (`pat_id`) and hospital admission (`csn`) may have multiple rows if multiple dialysis events occur.  
- `dialysis_present` vs. `dialysis_active`:  
  - `dialysis_present` indicates that dialysis was part of the clinical context.  
  - `dialysis_active` indicates whether dialysis was actually being administered at that time.  
- The `dialysis_type` field distinguishes between modalities such as **Intermittent Hemodialysis (IHD)**, **Continuous Renal Replacement Therapy (CRRT)**, and **Sustained Low-Efficiency Dialysis (SLED)**.  
- Downstream analyses can aggregate these rows to admission-level summaries (e.g., "did patient receive dialysis", "time of first dialysis").  
- **If you want to explore more on crrt, you can join with `csv_concepts_exports/crrt.csv` in the future.**


---


### IN_OUT File

#### Purpose
The IN_OUT file standardizes **fluid input and output events** from MIMIC-IV to match the Emory pipeline specification.  
It integrates ICU medication/infusion inputs (mapped to **RxNorm**) and fluid outputs (mapped to **LOINC**) into a unified schema.  
This file provides encounter-level fluid balance context aligned by hospital admission (`csn` = `hadm_id`) and timestamps.  

#### Source Tables
- 🟡 **`icu_inputevents.csv`** (fluid and medication administration events in the ICU).  
- 🟡 **`icu_outputevents.csv`** (fluid output events in the ICU, e.g., urine, drains).  
- 🟡 **`inputevents_to_rxnorm.csv`** (mapping of `itemid` → RxNorm standardized drug/solution names).  
- 🟡 **`outputevents_to_loinc.csv`** (mapping of `itemid` → LOINC standardized measurement concepts).  
- 🟡 **`icu_icustays.csv`** (provides mapping from `stay_id` → `hadm_id`, used to assign `csn`).  

#### Processing Logic
1. **Process Inputevents** (`icu_inputevents.csv`).  
   - Read in chunks (500k rows at a time) to handle memory.  
   - Keep only relevant columns: `stay_id`, `starttime`, `itemid`, `ordercategoryname`.  
   - Merge with 🟡 `inputevents_to_rxnorm.csv` on `itemid` to map to **RxNorm concept name**.  
   - Create standardized fields:  
     - `service_ts` = `starttime` (nursing charted time).  
     - `order_ts` = `starttime` (proxy for fluid order timestamp).  
     - `order_clinical_desc` = `omop_concept_name` (RxNorm mapped label).  
     - `order_catalog_desc` = `ordercategoryname` (input event category).  
   - Merge with 🟡 `icu_icustays.csv` on `stay_id` to retrieve `hadm_id` → stored as `csn`.  

2. **Process Outputevents** (`icu_outputevents.csv`).  
   - Read in chunks (500k rows at a time).  
   - Keep only relevant columns: `stay_id`, `charttime`, `itemid`.  
   - Merge with 🟡 `outputevents_to_loinc.csv` on `itemid` to map to **LOINC concept name**.  
   - Create standardized fields:  
     - `service_ts` = `charttime` (nursing charted time).  
     - `order_ts` = `charttime` (proxy for fluid order timestamp).  
     - `order_clinical_desc` = `omop_concept_name` (LOINC mapped label).  
     - `order_catalog_desc` = `category` (output event category, e.g., urine, drains).  
   - Merge with 🟡 `icu_icustays.csv` on `stay_id` to retrieve `hadm_id` → stored as `csn`.  

3. **Combine Input + Output events**.  
   - Concatenate processed inputevents and outputevents tables.  
   - Drop `stay_id` (not needed after mapping).  
   - Reorder columns so that `csn` is the first column.  
   - Save as **IN_OUT.csv** under the flat files directory.  

#### Final Columns
| Column Name          | Source / Logic                                                                 |
|-----------------------|--------------------------------------------------------------------------------|
| `csn`                | 🟡 `hadm_id` from **`icu_icustays.csv`**, mapped via `stay_id`                 |
| `service_ts`         | 🟡 `starttime` (inputs) or `charttime` (outputs), nursing charted timestamp    |
| `order_ts`           | 🟡 Same as `service_ts`, proxy for order time (true order time not available)  |
| `order_clinical_desc`| 🟡 RxNorm name from **`inputevents_to_rxnorm.csv`** (inputs), LOINC name from **`outputevents_to_loinc.csv`** (outputs) |
| `order_catalog_desc` | 🟡 `ordercategoryname` (inputs), `category` (outputs), higher-level event category |

#### Special Notes
- **Inputs vs. Outputs**:  
  - Inputs (medications/fluids) are standardized using **RxNorm**.  
  - Outputs (urine, drains, fluid loss) are standardized using **LOINC**.  
- **Order timestamps**: MIMIC does not provide true physician order times for fluids; `order_ts` is set equal to the recorded `starttime`/`charttime`.  
- Both `service_ts` and `order_ts` are retained for compatibility with downstream pipelines, though they are identical in this schema.  
- File size can be very large due to high-frequency charting; processed incrementally in chunks.  


---


### GCS File

#### Purpose
The GCS file standardizes **Glasgow Coma Scale (GCS) assessments** from MIMIC-IV to match the Emory pipeline specification.  
It captures neurological status assessments — eye, verbal, and motor components — along with the total score, aligned by patient encounter (`csn`) and timestamp (`recorded_time`).  
An additional column `gcs_unable` is retained from MIMIC to indicate cases where scoring could not be performed (e.g., intubated patients).

#### Source Tables
- 🟢 **`csv_concepts_exports/gcs.csv`** (concept table with pre-computed GCS component and total scores).  
- 🟡 **`icu_icustays.csv`** (maps `stay_id` → `hadm_id` and `subject_id`, required for `csn` and `pat_id`).  

#### Processing Logic
1. **Load GCS concept table** (`gcs.csv`).  
   - Includes `stay_id`, `subject_id`, `charttime`, and all GCS components (`gcs_eyes`, `gcs_verbal`, `gcs_motor`, `gcs`, `gcs_unable`).  

2. **Join with ICU stays** (`icu_icustays.csv`).  
   - Merge on (`stay_id`, `subject_id`) to bring in `hadm_id` (hospital admission ID).  
   - Standardize identifiers:  
     - `subject_id` → `pat_id`  
     - `hadm_id` → `csn`  

3. **Rename and align fields**.  
   - `charttime` → `recorded_time`  
   - `gcs_eyes` → `gcs_eye_score`  
   - `gcs_verbal` → `gcs_verbal_score`  
   - `gcs_motor` → `gcs_motor_score`  
   - `gcs` → `gcs_total_score`  
   - Keep `gcs_unable` as-is (not required in Emory schema but useful for context).  

4. **Select final schema**.  
   - Retain only the standardized columns:  
     - `pat_id`, `csn`, `recorded_time`, `gcs_eye_score`, `gcs_verbal_score`, `gcs_motor_score`, `gcs_total_score`, `gcs_unable`.  

5. **Export to flat file**.  
   - Save output as **GCS.csv** under the `mimic_flat_files` directory.  

#### Final Columns
| Column Name        | Source / Logic                                                                 |
|---------------------|--------------------------------------------------------------------------------|
| `pat_id`           | 🟡 `subject_id` from **`icu_icustays.csv`**                                    |
| `csn`              | 🟡 `hadm_id` from **`icu_icustays.csv`** (via `stay_id`)                       |
| `recorded_time`    | 🟢 `charttime` from **`gcs.csv`**                                              |
| `gcs_eye_score`    | 🟢 `gcs_eyes` from **`gcs.csv`**                                               |
| `gcs_verbal_score` | 🟢 `gcs_verbal` from **`gcs.csv`**                                             |
| `gcs_motor_score`  | 🟢 `gcs_motor` from **`gcs.csv`**                                              |
| `gcs_total_score`  | 🟢 `gcs` (total GCS) from **`gcs.csv`**                                        |
| `gcs_unable`       | 🟢 `gcs_unable` from **`gcs.csv`**, retained as a supplementary column         |

#### Special Notes
- The total score `gcs_total_score` should equal the sum of `gcs_eye_score + gcs_verbal_score + gcs_motor_score` when all components are available.  
- Valid GCS totals range from **3 to 15**; any values outside this range should be investigated.  
- The `gcs_unable` field is unique to MIMIC-IV and marks assessments that could not be performed (e.g., due to intubation). It is not part of the Emory schema but is retained here for completeness.  
- Multiple rows may exist per patient per admission if repeated GCS assessments were charted.  


---


### CULTURES File

#### Purpose
The CULTURES file standardizes **microbiology culture events** from MIMIC-IV to match the Emory pipeline specification.  
It captures specimen collection, culture test details, and result timing aligned by patient encounter (`csn`) and patient identifier (`pat_id`).  
Since MIMIC-IV does not provide LOINC mappings for microbiology tests, the `loinc_code` field is filled with `"NOT AVAILABLE"`.

#### Source Tables
- 🟡 **`hosp_microbiologyevents.csv`** (records of microbiology tests, specimen type, test performed, organism/antibiotic results).  

#### Processing Logic
1. **Load microbiology events** (`hosp_microbiologyevents.csv`).  
   - Key fields: `subject_id`, `hadm_id`, `charttime`, `chartdate`, `storetime`, `spec_itemid`, `spec_type_desc`, `test_itemid`, `test_name`.  

2. **Rename and align columns**.  
   - `subject_id` → `pat_id`  
   - `hadm_id` → `csn`  
   - `charttime` → `specimen_collect_time`  
   - `chartdate` → `order_time`  
   - `storetime` → `lab_result_time`  
   - `test_itemid` → `proc_code`  
   - `test_name` → `proc_desc`  
   - `spec_itemid` → `component_id`  
   - `spec_type_desc` → `component`  

3. **Construct additional fields**.  
   - `result_status` hard-coded as `"Not Recorded"` (MIMIC does not provide status directly).  
   - `loinc_code` hard-coded as `"NOT AVAILABLE"` (no LOINC mapping available for microbiology tests in MIMIC-IV).  

4. **Select standardized columns**.  
   - Keep only the columns required by the pipeline.  

5. **Save output**.  
   - Export as **CULTURES.csv** under the `mimic_flat_files` directory.  

#### Final Columns
| Column Name            | Source / Logic                                                                 |
|-------------------------|--------------------------------------------------------------------------------|
| `csn`                  | 🟡 `hadm_id` from **`hosp_microbiologyevents.csv`**                            |
| `pat_id`               | 🟡 `subject_id` from **`hosp_microbiologyevents.csv`**                         |
| `specimen_collect_time`| 🟡 `charttime` from **`hosp_microbiologyevents.csv`**                          |
| `order_time`           | 🟡 `chartdate` from **`hosp_microbiologyevents.csv`**                          |
| `lab_result_time`      | 🟡 `storetime` from **`hosp_microbiologyevents.csv`**                          |
| `result_status`        | Hard-coded `"Not Recorded"` (status not available in MIMIC-IV)                 |
| `proc_code`            | 🟡 `test_itemid` from **`hosp_microbiologyevents.csv`**                        |
| `proc_desc`            | 🟡 `test_name` from **`hosp_microbiologyevents.csv`**                          |
| `component_id`         | 🟡 `spec_itemid` from **`hosp_microbiologyevents.csv`**                        |
| `component`            | 🟡 `spec_type_desc` from **`hosp_microbiologyevents.csv`**                     |
| `loinc_code`           | Hard-coded `"NOT AVAILABLE"` (MIMIC does not provide LOINC for microbiology)   |

#### Special Notes
- MIMIC-IV does not store `order_time` for microbiology tests; here `chartdate` is used as a proxy.  
- `result_status` is not directly available; set as `"Not Recorded"` for consistency with the pipeline schema.  
- `loinc_code` is not provided for microbiology tests in MIMIC-IV, unlike chemistry/hematology labs. It remains `"NOT AVAILABLE"`.  
- Each row corresponds to a culture test on a specimen; multiple rows per encounter may exist if several cultures were ordered.  
- **Do we need to keep the test result?**


---


### BEDLOCATION File

#### Purpose
The BEDLOCATION file standardizes **patient location movements** within the hospital from MIMIC-IV to match the Emory pipeline specification.  
It records when a patient was admitted to, transferred between, or discharged from different hospital units (e.g., ED, ICU, surgical wards).  
This file captures the **unit-level location timeline** for each encounter (`csn` = `hadm_id`).

#### Source Tables
- 🟡 **`hosp_transfers.csv`** (contains all intra-hospital transfers, with `intime` and `outtime` for each care unit).  

#### Processing Logic
1. **Load transfers** (`hosp_transfers.csv`).  
   - Key fields: `subject_id`, `hadm_id`, `careunit`, `intime`, `outtime`.  

2. **Subset and rename columns**.  
   - `subject_id` → `pat_id`  
   - `hadm_id` → `csn`  
   - `careunit` → `bed_unit`  
   - `intime` → `bed_location_start`  
   - `outtime` → `bed_location_end`  

3. **Handle missing values**.  
   - If `careunit` is missing, fill with `"Not Recorded"`.  
   - Convert `intime` and `outtime` to datetime format.  

4. **Finalize schema**.  
   - Keep only standardized columns required by the pipeline.  

5. **Save output**.  
   - Export as **BEDLOCATION.csv** under the `mimic_flat_files` directory.  

#### Final Columns
| Column Name         | Source / Logic                                                                 |
|----------------------|--------------------------------------------------------------------------------|
| `csn`               | 🟡 `hadm_id` from **`hosp_transfers.csv`**                                     |
| `pat_id`            | 🟡 `subject_id` from **`hosp_transfers.csv`**                                  |
| `bed_unit`          | 🟡 `careunit` from **`hosp_transfers.csv`**, unit where the patient stayed      |
| `bed_location_start`| 🟡 `intime` from **`hosp_transfers.csv`**, when the patient entered the unit    |
| `bed_location_end`  | 🟡 `outtime` from **`hosp_transfers.csv`**, when the patient left the unit      |

#### Special Notes
- Each row corresponds to a **single unit stay** within a hospital admission.  
- A patient may have multiple rows per `csn` if transferred between units (e.g., ED → MICU → Surgery → Discharge).  
- If `outtime` is missing (e.g., patient still admitted), it remains null.  
- Some Emory-specific columns (`bed_room`, `bed_id`, `hospital_service`, `accomodation_code`) are **not available in MIMIC-IV** and are therefore omitted.  
- This file can be used to reconstruct the **care pathway** of a patient across the hospital stay.  



---



<!--### ORPROCEDURES File

#### Purpose
The ORPROCEDURES file standardizes **surgical procedure records** from MIMIC-IV to match the Emory pipeline specification.  
It captures official ICD-coded surgical procedures performed during a hospital admission (`csn` = `hadm_id`) and provides a standardized procedure description.  
**Since MIMIC-IV does not contain detailed OR timestamps or unique OR identifiers, those fields are filled with `"NOT AVAILABLE"`.**

#### Source Tables
- 🟡 **`hosp_procedures_icd.csv`** (ICD-coded surgical procedures with hospital admission identifiers and charted dates).  
- 🟡 **`hosp_d_icd_procedures.csv`** (dictionary providing `long_title` descriptions for each ICD procedure code).  

#### Processing Logic
1. **Load procedure data** (`hosp_procedures_icd.csv`).  
   - Key fields: `subject_id`, `hadm_id`, `icd_code`, `icd_version`, `chartdate`.  

2. **Load ICD procedure dictionary** (`hosp_d_icd_procedures.csv`).  
   - Key fields: `icd_code`, `icd_version`, `long_title`.  

3. **Merge datasets**.  
   - Join `hosp_procedures_icd` with `hosp_d_icd_procedures` on (`icd_code`, `icd_version`) to attach `long_title` (standardized procedure description).  

4. **Construct standardized columns**.  
   - `pat_id` = `subject_id`  
   - `csn` = `hadm_id`  
   - `procedure_start_dttm` = `chartdate` (date-only, no exact start time available)  
   - `procedure_end_dttm` = `chartdate` (same as start date)  
   - `primary_procedure_nm` = `long_title` from dictionary  
   - **`in_or_dttm`, `out_or_dttm`, `or_procedure_id`, `service_nm` = `"NOT AVAILABLE"`**
5. **Finalize schema**.  
   - Keep only standardized columns required by the pipeline.  

6. **Save output**.  
   - Export as **ORPROCEDURES.csv** under the `mimic_flat_files` directory.  

#### Final Columns
| Column Name            | Source / Logic                                                                 |
|-------------------------|--------------------------------------------------------------------------------|
| `csn`                  | 🟡 `hadm_id` from **`hosp_procedures_icd.csv`**                                |
| `pat_id`               | 🟡 `subject_id` from **`hosp_procedures_icd.csv`**                             |
| `in_or_dttm`           | Hard-coded `"NOT AVAILABLE"` (MIMIC does not provide OR in-time)               |
| `out_or_dttm`          | Hard-coded `"NOT AVAILABLE"` (MIMIC does not provide OR out-time)              |
| `procedure_start_dttm` | 🟡 `chartdate` from **`hosp_procedures_icd.csv`**                              |
| `procedure_end_dttm`   | 🟡 `chartdate` from **`hosp_procedures_icd.csv`**                              |
| `or_procedure_id`      | Hard-coded `"NOT AVAILABLE"` (no unique OR identifier in MIMIC)                |
| `primary_procedure_nm` | 🟡 `long_title` from **`hosp_d_icd_procedures.csv`** (mapped via ICD code)     |
| `service_nm`           | Hard-coded `"NOT AVAILABLE"` (MIMIC does not include surgical service mapping) |

#### Special Notes
- The `chartdate` field in MIMIC-IV is **date-only** (no timestamp), so `procedure_start_dttm` and `procedure_end_dttm` cannot represent exact OR times.  
- Each row corresponds to one ICD-coded procedure performed during a hospital admission.  
- Multiple rows may exist per encounter (`csn`) if multiple procedures were performed.  
- Unlike Emory, MIMIC-IV does not provide fields such as OR room, service department, or OR-level identifiers. These are marked `"NOT AVAILABLE"`.  



---

-->



### DIAGNOSIS File

#### Purpose
The DIAGNOSIS file standardizes **diagnosis records** from MIMIC-IV to match the Emory pipeline specification.  
It captures all ICD-coded diagnoses (both ICD-9 and ICD-10) assigned during a hospital admission (`csn` = `hadm_id`) and provides a standardized diagnosis description (`long_title`).  
Since MIMIC-IV does not contain timestamps for individual diagnosis entries, the `dx_time_date` field is filled with `"NOT AVAILABLE"`.

#### Source Tables
- 🟡 **`hosp_diagnoses_icd.csv`** (hospital diagnoses with patient identifiers, hospital admission identifiers, ICD codes, and ICD version).  
- 🟡 **`hosp_d_icd_diagnoses.csv`** (dictionary providing `long_title` descriptions for each ICD diagnosis code).  

#### Processing Logic
1. **Load diagnosis records** (`hosp_diagnoses_icd.csv`).  
   - Key fields: `subject_id`, `hadm_id`, `icd_code`, `icd_version`.  

2. **Load ICD dictionary** (`hosp_d_icd_diagnoses.csv`).  
   - Key fields: `icd_code`, `icd_version`, `long_title`.  

3. **Merge datasets**.  
   - Join on (`icd_code`, `icd_version`) to add `long_title` (standardized diagnosis description).  

4. **Split ICD versions**.  
   - If `icd_version = 9`, store `icd_code` in `dx_code_icd9`.  
   - If `icd_version = 10`, store `icd_code` in `dx_code_icd10`.  

5. **Construct standardized columns**.  
   - `pat_id` = `subject_id`  
   - `csn` = `hadm_id`  
   - `dx_code_icd9` = ICD-9 diagnosis code if applicable, else null  
   - `dx_code_icd10` = ICD-10 diagnosis code if applicable, else null  
   - `dx_time_date` = `"NOT AVAILABLE"` (MIMIC-IV does not provide diagnosis time)  
   - `long_title` = official diagnosis description from dictionary  

6. **Finalize schema**.  
   - Keep only required columns.  
   - Drop duplicate rows if they exist.  

7. **Save output**.  
   - Export as **DIAGNOSIS.csv** under the `mimic_flat_files` directory.  

#### Final Columns
| Column Name       | Source / Logic                                                                 |
|-------------------|--------------------------------------------------------------------------------|
| `pat_id`          | 🟡 `subject_id` from **`hosp_diagnoses_icd.csv`**                              |
| `csn`             | 🟡 `hadm_id` from **`hosp_diagnoses_icd.csv`**                                 |
| `dx_code_icd9`    | 🟡 `icd_code` from **`hosp_diagnoses_icd.csv`**, if `icd_version = 9`           |
| `dx_code_icd10`   | 🟡 `icd_code` from **`hosp_diagnoses_icd.csv`**, if `icd_version = 10`          |
| `dx_time_date`    | Hard-coded `"NOT AVAILABLE"` (no diagnosis timestamp in MIMIC-IV)              |
| `long_title`      | 🟡 `long_title` from **`hosp_d_icd_diagnoses.csv`** (mapped by code + version) |

#### Special Notes
- Both ICD-9 and ICD-10 diagnoses are retained in the same table, separated into two fields.  
- The `long_title` field ensures human-readable descriptions for each ICD code.  
- If the ICD code cannot be mapped to a dictionary entry, `long_title` is set to `"UNKNOWN"`.  
- Each hospital admission (`csn`) may have multiple diagnosis rows if multiple ICD codes are assigned.  



---


### ICDPROCEDURES File

#### Purpose
The ICDPROCEDURES file standardizes **ICD-coded surgical and procedural records** from MIMIC-IV to match the Emory pipeline specification.  
It captures all ICD-9 and ICD-10 coded procedures performed during a hospital admission (`csn` = `hadm_id`) and provides a standardized procedure description (`procedure_desc`).  
Both ICD-9 and ICD-10 codes are retained in separate columns for clarity. Since MIMIC-IV does not contain detailed OR timestamps, the `procedure_date` field uses `chartdate` from the hospital ICD procedure records.

#### Source Tables
- 🟡 **`hosp_procedures_icd.csv`** (ICD-coded procedures with patient identifiers, hospital admission identifiers, ICD codes, versions, and charted dates).  
- 🟡 **`hosp_d_icd_procedures.csv`** (dictionary providing `long_title` descriptions for each ICD procedure code).  

#### Processing Logic
1. **Load procedure records** (`hosp_procedures_icd.csv`).  
   - Key fields: `subject_id`, `hadm_id`, `icd_code`, `icd_version`, `chartdate`.  

2. **Load ICD dictionary** (`hosp_d_icd_procedures.csv`).  
   - Key fields: `icd_code`, `icd_version`, `long_title`.  

3. **Merge datasets**.  
   - Join on (`icd_code`, `icd_version`) to add `long_title` (standardized procedure description).  

4. **Split ICD versions**.  
   - If `icd_version = 9`, store `icd_code` in `icd9_procedure_code` and mark `icd10_procedure_code = "NOT AVAILABLE"`.  
   - If `icd_version = 10`, store `icd_code` in `icd10_procedure_code` and mark `icd9_procedure_code = "NOT AVAILABLE"`.  

5. **Construct standardized columns**.  
   - `pat_id` = `subject_id`  
   - `csn` = `hadm_id`  
   - `icd9_procedure_code` = ICD-9 procedure code if applicable, else `"NOT AVAILABLE"`  
   - `icd10_procedure_code` = ICD-10 procedure code if applicable, else `"NOT AVAILABLE"`  
   - `procedure_desc` = `long_title` from dictionary (fallback `"UNKNOWN"` if missing)  
   - `procedure_date` = `chartdate` from `hosp_procedures_icd`  

6. **Finalize schema**.  
   - Keep only required columns.  
   - Drop duplicate rows if they exist.  

7. **Save output**.  
   - Export as **ICD_PROCEDURES.csv** under the `mimic_flat_files` directory.  

#### Final Columns
| Column Name            | Source / Logic                                                                 |
|-------------------------|--------------------------------------------------------------------------------|
| `pat_id`               | 🟡 `subject_id` from **`hosp_procedures_icd.csv`**                             |
| `csn`                  | 🟡 `hadm_id` from **`hosp_procedures_icd.csv`**                                |
| `icd9_procedure_code`  | 🟡 `icd_code` if `icd_version = 9`, else `"NOT AVAILABLE"`                     |
| `icd10_procedure_code` | 🟡 `icd_code` if `icd_version = 10`, else `"NOT AVAILABLE"`                    |
| `procedure_desc`       | 🟡 `long_title` from **`hosp_d_icd_procedures.csv`** (mapped by code + version, fallback `"UNKNOWN"`) |
| `procedure_date`       | 🟡 `chartdate` from **`hosp_procedures_icd.csv`**                              |

#### Special Notes
- Both ICD-9 and ICD-10 procedure codes are explicitly retained.  
- The `procedure_desc` field provides a human-readable description of the ICD code.  
- If the ICD code cannot be mapped to the dictionary, the description is set to `"UNKNOWN"`.  
- Each hospital admission (`csn`) may contain multiple procedures, resulting in multiple rows.  
- The `procedure_date` only has **date granularity** (no exact start/end time is available in MIMIC-IV).  


---



### CPT_PROCEDURES File

#### Purpose
The CPT_PROCEDURES file standardizes **Current Procedural Terminology (CPT/HCPCS) procedure records** from MIMIC-IV to match the Emory pipeline specification.  
It captures all CPT/HCPCS-coded procedures performed during a hospital admission (`csn` = `hadm_id`), aligned with the patient identifier (`pat_id`) and charted procedure date.  
Each procedure row includes the official code and its description, taken from the MIMIC-IV dictionary.

#### Source Tables
- 🟡 **`hosp_hcpcsevents.csv`** (hospital events with HCPCS/CPT procedure codes, chartdates, and identifiers).  
- 🟡 **`hosp_d_hcpcs.csv`** (dictionary mapping `hcpcs_cd` to official short and long descriptions).  

#### Processing Logic
1. **Load CPT/HCPCS events** (`hosp_hcpcsevents.csv`).  
   - Key fields: `subject_id`, `hadm_id`, `hcpcs_cd`, `chartdate`.  

2. **Load CPT/HCPCS dictionary** (`hosp_d_hcpcs.csv`).  
   - Key fields: `hcpcs_cd`, `short_description`, `long_description`.  
   - Renamed internally to avoid conflicts: `dict_short_description`, `dict_long_description`.  

3. **Merge events with dictionary**.  
   - Join on `hcpcs_cd` to attach human-readable descriptions.  
   - Prefer `dict_long_description` if available; fallback to `dict_short_description` if long description is missing.  

4. **Construct standardized columns**.  
   - `pat_id` = `subject_id`  
   - `csn` = `hadm_id`  
   - `procedure_cpt_code` = `hcpcs_cd`  
   - `procedure_cpt_desc` = merged description (long preferred, short as fallback)  
   - `procedure_dttm` = `chartdate`  

5. **Finalize schema**.  
   - Keep only standardized columns.  
   - Save output to **CPT_PROCEDURES.csv**.  

#### Final Columns
| Column Name            | Source / Logic                                                                 |
|-------------------------|--------------------------------------------------------------------------------|
| `pat_id`               | 🟡 `subject_id` from **`hosp_hcpcsevents.csv`**                                |
| `csn`                  | 🟡 `hadm_id` from **`hosp_hcpcsevents.csv`**                                   |
| `procedure_cpt_code`   | 🟡 `hcpcs_cd` from **`hosp_hcpcsevents.csv`**                                  |
| `procedure_cpt_desc`   | 🟡 `dict_long_description` from **`hosp_d_hcpcs.csv`**, fallback = `dict_short_description` |
| `procedure_dttm`       | 🟡 `chartdate` from **`hosp_hcpcsevents.csv`**                                 |

#### Special Notes
- CPT/HCPCS codes in MIMIC-IV cover **billing-related procedures** (e.g., imaging, endoscopy, outpatient services) and differ from ICD-coded surgical procedures.  
- The `procedure_cpt_desc` field ensures each code has a human-readable description; if the dictionary lacks a long description, the short one is used.  
- The `procedure_dttm` column only has **date granularity** (no timestamp), as MIMIC-IV records CPT procedures by date only.  
- Multiple rows may exist per admission (`csn`) if multiple CPT/HCPCS codes were billed on the same day.  



---

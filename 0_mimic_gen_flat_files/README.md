# MIMIC Preprocess Note

This folder contains preprocessing scripts to convert raw MIMIC-IV data into the standardized flat files required by the **Sepy2.0 pipeline**. 
This README provides documentation of specific mapping decisions and special handling notes.
 


## Encounter Mapping Decisions

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

# Sepy Data Processing Pipeline

This document provides a detailed description of the data processing pipeline implemented in the `sepyIMPORT` class, designed for processing clinical data from electronic health records (EHR). The pipeline is designed to be reusable for any dataset derived from various healthcare systems.

## Overview

The `sepyIMPORT` class provides a framework for importing, cleaning, and processing various types of clinical data from flat files. It handles different data categories including patient demographics, vital signs, lab results, medication administration, ventilation settings, and more.

## Required Flat Files

The pipeline expects the following flat files, each containing specific clinical data:

| File Identifier | Description | Purpose |
|----------------|-------------|---------|
| `ENCOUNTER` | Patient encounter data | Contains information about each hospital encounter - admission and discharge times |
| `DEMOGRAPHICS` | Patient demographic information | Contains patient characteristics like age, sex, race, etc. |
| `INFUSIONMEDS` | Medication infusion data | Records of medication administration for all encounters |
| `LABS` | Laboratory test results | Results of laboratory tests performed for all encounters |
| `VITALS` | Regularly recorded vital signs | Recordings of vital signs like blood pressure, heart rate, etc. |
| `VENT` | Ventilator settings and measurements | Data from mechanical ventilators for all encounters that were on vent |
| `DIALYSIS` | Dialysis treatment data | Records of renal replacement therapy |
| `IN_OUT` | Fluid input and output data | Records of fluid administration and output |
| `GCS` | Glasgow Coma Scale assessments | Neurological status assessments |
| `CULTURES` | Microbiology culture results | Results from microbiological tests |
| `BEDLOCATION` | Patient location data | Tracks patient movements between hospital units |
| `ORPROCEDURES` | Surgical and other procedures | Records of procedures performed |
| `DIAGNOSIS` | Patient diagnoses | Records of clinical diagnoses, including ICD codes |

## Expected Columns in Each Flat File
## (M: Mandatory features - missing/differently named features will cause the pipeline to break)
## (R: Reproducibility features: won't cause the code to break, but needed to match the Emory/Grady datasets)

### ENCOUNTER File
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `hospital_admission_date_time`: Time of admission
- M: `hospital_discharge_date_time`: Time of discharge
- M: `ed_presentation_time`: Time of ED presentation if applicable
- M: `encounter_type`: ER/IN (Emergency or Inpatient)
- M: `age`: age
- R: `discharge_to` : Where the patient was discharged to - HOME SELF CARE/HOSPICE/EXPIRED etc.
- R: `pre_admit_location` : TRANSFER/NON-HC FACILITY etc.
- R: `total_icu_days`: Useful for filtering ICU encounters
- Additional columns can be dropped using the `drop_cols` parameter in the data_config file.

### DEMOGRAPHICS File
- M: `pat_id`: Patient identifier (Used as index)
- M: `gender`: Patient biological sex
- M: `race_code`: Patient race CODE (because Emory and Grady were deidentified)
- M: `ethnicity_code`: Patient ethnicity CODE (because Emory and Grady were deidentified)
- Additional columns can be dropped using the `drop_cols` parameter

### INFUSIONMEDS File
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `medication_id`: Identifier for the medication (used to map to medication type for grouping)
- M: `med_order_time`: Time the medication was ordered
- M: `med_action_time`: Time the medication was administered (when med_start is not recorded)
- M: `med_start`: Time the medication administration (like begin bag) was started
- M: `med_stop`: Time the medication was stopped
- M: `med_order_route`: Adminstration route (IV etc.) 
- M: `med_action_dose`: Dosage of medication administered (Needed for vasopressors)
- M: `med_action_dose_unit`: Unit of the medication dose (Needed for vasopressors)
- Columns specified in `drop_cols` are removed

TODO: Basically, vasopressors and anti-infective meds are instances of two broader categories of data types that you'd like to handle w.r.t medications. 
Anti-infective meds: any med in this group can be given for "suspicion of infection", so it is only processed to check if there was one or not 
Vasopressors: We create separate columns in the supertables as indicator variables for each of the 6 vasopressor 

### LABS File
- M: `csn`: Unique encounter identifier
- M: `pat_id`: Patient identifier
- M: `component_id`: Identifier for the lab test (Used to mgroup labs into similar measurements)
- M: `lab_result`: Result of the lab test
- M: `lab_result_time`: Time the result was reported
- M: `collection_time`: Time the specimen was collected
- M: `result_status`: Status of the result (final, preliminary, etc.)
- M: `proc_cat_id`: Procedure category (Blood, CSF etc.) number
- M: `proc_cat_name`: Procedure category name
- M: `proc_code`: Procedure code
- M: `proc_desc`: Procedure description (text)
- M: `component` : component name (change SepyImport file import_labs if you wanna remove some of these from the list) 
- M: `loinc_code`: Loinc code

### VITALS File
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `recorded_time`: Time the vital signs were recorded
- Columns for various vital signs (specified in `vital_col_names`)
- Additional columns can be dropped using the `drop_cols` parameter
- Option to merge columns with different names into one vitals feature. Since the vitals flatfile is expected to ALREADY BE PIVOTED. That happens at a later stage for labs etc.

### VENT File
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- R: `vent_rate_set`: Ask?
- R: `vent_tidal_rate_set`: Ask? 
- M: `vent_tidal_rate_exhaled`: tidal volume 
- M: `peep`: peep
- M: `fio2`: fio2 (vent)
- M: `o2_flow_rate`: Oxygen flow rate (nasal cannula)
- M: `o2_flow_rate_units`: Units for oxygen flow rate 
- M: `recorded_time`: For the measurements - like fio2, peep etc
- M: `vent_start_time`: When this ventilation session started
- M: `vent_stop_time`: When this ventilation session ended

### DIALYSIS File
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `service_timestamp`: Recorded timestamp of dialysis start
- Additional columns can be dropped using the `drop_cols` parameter

### IN_OUT File
- `csn`: Unique encounter identifier (used as index)
- `service_ts`: Timestamp of start 
- `order_ts`: Timestamp of fluid order 
- `order_clinical_desc`: Description of order 
- `order_catalog_desc`: Description of kind of fluid 
- Additional columns can be dropped using the `drop_cols` parameter

### GCS File
- M" `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `recorded_time`: Time the GCS assessment was recorded
- M: `gcs_eye_score`: Eye opening score (1-4)
- M: `gcs_verbal_score`: Verbal response score (1-5)
- M: `gcs_motor_score`: Motor response score (1-6)
- M: `gcs_total_score`: Total GCS score (3-15)
- Additional columns can be dropped using the `drop_cols` parameter

### CULTURES File
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `specimen_collect_time`: Time specimen was collected
- M: `order_time`: Time specimen was ordered
- M: `lab_result_time`: Time of result 
- M: `result_status`: Status (Completed, In process, Ordered etc.)
- M: `proc_code`: procedure code 
- M: `proc_desc`: procedure description 
- M: `component_id`: component id number
- M: `component`: Name of the component
- M: `loinc_code`: Loinc mapping
- Additional columns can be dropped using the `drop_cols` parameter

### BEDLOCATION File
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `bed_unit`: Identifier for the hospital unit
- M: `bed_location_start`: Time the patient arrived at the location
- M: `bed_location_end`: Time the patient left the location
- Additional columns can be dropped using the `drop_cols` parameter

### ORPROCEDURES File
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `in_or_dttm`: Time patient was taken into the OR
- M: `out_or_dttm`: Time patient was taken out of the OR
- M: `procedure_start_dttm`: procedure start time
- M: `procedure_end_dttm`: procedure end time 
- M: `or_procedure_id`: procedure ID
- M: `primary_procedure_nm`: Name of the procedure
- R: `service_nm`: Name of the service department (Urology Sergery, Neurology Surgery etc.)
- Additional columns can be dropped using the `drop_cols` parameter

### DIAGNOSIS File
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `dx_code_icd10`: ICD-10 diagnosis code
- M: `dx_time_date`: Time the diagnosis was recorded
- Additional columns can be dropped using the `drop_cols` parameter

### ICD Procedures 
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `icd10_procedure_code`: Map all ICD9 to ICD10 please (look at data_processing_helper_functions)
- M: `procedure_desc`: Description of the procedure
- M: `procedure_date`: Date the procedure was performed  
- Additional columns can be dropped using the `drop_cols` parameter

### CPT Procedures 
- M: `csn`: Unique encounter identifier (used as index)
- M: `pat_id`: Patient identifier
- M: `procedure_cpt_code`: Current Procedural Terminology procedure code
- M: `procedure_cpt_desc`: Description of the procedure
- M: `procedure_dttm`: Date and time the procedure was performed  
- Additional columns can be dropped using the `drop_cols` parameter


## Additional Required Information (Groupings)

The pipeline requires several mapping and grouping files to properly categorize and process the data:
All columns specified are mandatory. 

### Medication Groupings
File: Specified by `dataConfig["dictionary_paths"]["grouping_types"]["infusion_meds"]`

This file maps medication IDs to clinical categories and includes:
- `medication_id`: Unique identifier for medications
- `med_name`: Name of medicine
- `super_table_col_name`: Standardized name for the medication (Name that shows up in the final dataset)
- `med_class`: Medication class (e.g., "anti-infective", "vasopressor")

### Lab Groupings
File: Specified by `dataConfig["dictionary_paths"]["grouping_types"]["grouping_labs"]`

This file maps lab component IDs to standardized names and includes:
- `component_id`: Unique identifier for lab tests
- `super_table_col_name`: Standardized name for the lab test
- `import`: Flag indicating whether to import this lab ("Yes" or "No")

### Bed Labels
File: Specified by `dataConfig["dictionary_paths"]["grouping_types"]["bed_labels"]`

This file classifies hospital bed units and includes:
- `bed_unit`: Identifier for the hospital unit
- `icu`: Binary flag indicating if the unit is an ICU (1) or not (0)
- `imc`: Binary flag indicating if the unit is an intermediate care unit (1) or not (0)
- `ed`: Binary flag indicating if the unit is an emergency department (1) or not (0)
- `procedure`: Binary flag indicating if the unit is a procedure area (1) or not (0)
- `icu_type`: Neuro/CTICU/Medical ICU/ Surgical ICU
- `unit_type`: Ward/ICU etc. (May be redundant)

### Fluid Groupings
File: Specified by `dataConfig["dictionary_paths"]["grouping_types"]["grouping_fluids"]`

This file categorizes fluid inputs and outputs and includes:
- Columns that map fluid administration/output types to standardized categories

### Comorbidity Mapping Files
Files:
- `ICD10_quan_deyo`: Maps ICD-10 codes to Quan-Deyo comorbidity categories
- `ICD10_quan_elix`: Maps ICD-10 codes to Quan-Elixhauser comorbidity categories

These files include:
- `ICD10`: ICD-10 diagnosis code
- `quan_deyo` or `quan_elix`: Comorbidity category

### Vent Mode Mapping Files
File: Specified by `dataConfig["dictionary_paths"]["grouping_types"]["grouping_vent_mode"]`

- `vent_name`: Name of the ventilator device 
- `vent_cat`: Category (Invasive/Non-Invasive-Nasal Cannula) 

## Data Processing Steps

For each data type, the processing follows these general steps:

# SepyIMPORT: 
1. **Import**: Read the raw data from CSV files
2. **Clean**: Convert data types, drop unnecessary columns, convert nan types to nan
3. **Merge**: Join data with relevant grouping/mapping files
4. **Store**: Save the processed data in class attributes for further use by SepyDICT

## Usage Example

```python
# Initialize the sepyIMPORT class with configuration
importer = sepyIMPORT(file_dictionary, sepyIMPORTConfigs, dataConfig)

# Import and process data
importer.import_demographics(drop_cols, index_col, date_cols)
importer.import_vitals(drop_cols, index_col, date_cols, merge_cols)
importer.import_labs(drop_cols, group_cols, date_cols, index_col, numeric_cols)
# ... import other data types as needed

# Access processed data
demographics_df = importer.df_demographics
vitals_df = importer.df_vitals
labs_df = importer.df_labs
```

## Notes on Implementation

- The pipeline uses pandas for data manipulation
- Timestamps are parsed as datetime objects
- Numeric values are converted using pandas' `to_numeric` function with `errors='coerce'`
- Missing values are handled according to the configuration in `sepyIMPORTConfigs["na_values"]`
- Helper functions in the `utils` module are used for common operations 
# Configuration Files

This document describes the three configuration files required for the Sepy data processing pipeline. These configuration files control the behavior of the data import process, specify file paths, and define data processing parameters.

## 1. File Dictionary Configuration

The file dictionary maps logical file identifiers to physical file paths. This allows the pipeline to locate the necessary data files regardless of the actual directory structure.

Example structure:

```yaml
ENCOUNTER: /path/to/encounters.csv
DEMOGRAPHICS: /path/to/demographics.csv
INFUSIONMEDS: /path/to/infusion_meds.csv
LABS: /path/to/labs.csv
VITALS: /path/to/vitals.csv
VENT: /path/to/vent.csv
DIALYSIS: /path/to/dialysis.csv
IN_OUT: /path/to/in_out.csv
GCS: /path/to/gcs.csv
CULTURES: /path/to/cultures.csv
BEDLOCATION: /path/to/bed_locations.csv
ORPROCEDURES: /path/to/or_procedures.csv
DIAGNOSIS: /path/to/diagnosis.csv
RADIOLOGY_NOTES: /path/to/radiology_notes.csv
CLINICAL_NOTES: /path/to/clinical_notes.csv
ICD10_quan_deyo: /path/to/icd10_quan_deyo_mapping.csv
ICD10_quan_elix: /path/to/icd10_quan_elix_mapping.csv
```

### Key Components:

- **File Identifiers**: The keys in the dictionary (e.g., `ENCOUNTER`, `DEMOGRAPHICS`)
- **File Paths**: The values in the dictionary, representing the physical location of each file

## 2. Import Configuration (sepyIMPORTConfigs)

This configuration controls the behavior of the import process, including how data is cleaned and processed.

Example structure:

```yaml
na_values:
  - NULL
  - null
  - ""
  - " "
  - NaN
  - NA
  - N/A

vital_col_names:
  - heart_rate
  - respiratory_rate
  - temperature
  - systolic_bp
  - diastolic_bp
  - mean_arterial_pressure
  - oxygen_saturation
  - rhythm
  
vasopressor_units:
  - norepinephrine_dose_unit
  - epinephrine_dose_unit
  - dopamine_dose_unit
  - phenylephrine_dose_unit
  - vasopressin_dose_unit

numeric_lab_col_names:
  - WBC
  - RBC
  - HGB
  - HCT
  - PLT
  - SODIUM
  - POTASSIUM
  - CHLORIDE
  - BICARBONATE
  - BUN
  - CREATININE
  - GLUCOSE
  # Additional numeric lab names...

string_lab_col_names:
  - COVID19
  - TROPONIN_QUAL
  # Additional string lab names...

# Text processing parameters
text_processing:
  # Radiology notes configuration
  radiology_notes:
    text_col: "report_text"           # Column containing the report text
    clean_text: true                  # Whether to clean/normalize the text
    max_length: null                  # Maximum text length to retain (null = no limit)

  # Clinical notes configuration
  clinical_notes:
    text_col: "note_text"             # Column containing the note text
    clean_text: true                  # Whether to clean/normalize the text
    extract_sections: true            # Whether to extract common clinical note sections
    max_length: null                  # Maximum text length to retain (null = no limit)
```

### Key Components:

- **na_values**: List of values to be treated as missing data
- **vital_col_names**: List of column names that contain vital sign measurements
- **vasopressor_units**: List of column names that contain units for vasopressor medications
- **numeric_lab_col_names**: List of lab tests that should be processed as numeric values
- **string_lab_col_names**: List of lab tests that should be processed as string values

## 3. Data Configuration (dataConfig)

This configuration specifies paths to grouping and mapping files that are used to categorize and standardize the data.

Example structure:

```yaml
dictionary_paths:
  grouping_types:
    infusion_meds: /path/to/infusion_med_groupings.csv
    grouping_labs: /path/to/lab_groupings.csv
    bed_labels: /path/to/bed_labels.csv
    grouping_fluids: /path/to/fluid_groupings.csv
```

### Key Components:

- **dictionary_paths**: Contains paths to mapping and grouping files
  - **grouping_types**: Specific mapping files
    - **infusion_meds**: Path to the file mapping medication IDs to standardized names
    - **grouping_labs**: Path to the file mapping lab component IDs to standardized names
    - **bed_labels**: Path to the file categorizing hospital bed units
    - **grouping_fluids**: Path to the file categorizing fluid inputs and outputs

## Usage Example

```python
import yaml

# Load configurations
with open('file_dictionary.yaml', 'r') as file:
    file_dictionary = yaml.safe_load(file)
    
with open('import_config.yaml', 'r') as file:
    sepyIMPORTConfigs = yaml.safe_load(file)
    
with open('data_config.yaml', 'r') as file:
    dataConfig = yaml.safe_load(file)

# Initialize the sepyIMPORT class with these configurations
importer = sepyIMPORT(file_dictionary, sepyIMPORTConfigs, dataConfig)
```

## Customization

These configuration files can be customized for different datasets and institutional needs:

1. **Adapt File Paths**: Update file paths in the file dictionary to point to your institution's data
2. **Add or Remove Lab Tests**: Modify the `numeric_lab_col_names` and `string_lab_col_names` lists to match available lab tests
3. **Update Vital Signs**: Add or remove vital signs in `vital_col_names` based on available data
4. **Customize NA Values**: Adjust the `na_values` list to match your institution's representation of missing data
5. **Update Grouping Files**: Create mapping files that reflect your institution's classification of medications, labs, and bed units 
6. **Configure Text Processing**: Adjust the parameters in the `text_processing` section to customize how radiology and clinical notes are processed:
   - Set `text_col` to match your dataset's column name containing the note text
   - Set `clean_text` to `false` if you want to preserve the original text format
   - Adjust `max_length` to limit the text length for processing very large notes
   - For clinical notes, set `extract_sections` to `true` to automatically identify and extract common note sections (history, physical exam, assessment, plan) 
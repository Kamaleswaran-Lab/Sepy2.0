# Grouping Files Documentation

This document describes the structure and purpose of the various grouping files used in the Sepy data processing pipeline. These files are essential for standardizing and categorizing the clinical data.

## Overview

Grouping files serve as dictionaries or mappings between raw clinical data identifiers and standardized categories. They enable the pipeline to:

1. Convert institution-specific codes to standardized terminology
2. Group similar clinical concepts together
3. Filter data to include only relevant items
4. Apply consistent labels across different data sources

## Required Grouping Files

### 1. Medication Groupings

**File Path**: Specified by `dataConfig["dictionary_paths"]["grouping_types"]["infusion_meds"]`

**Purpose**: Maps medication identifiers to standardized names and categories.

**Structure**:

| Column Name | Description | Example Values |
|-------------|-------------|----------------|
| `medication_id` | Unique identifier for the medication | 12345, 67890 |
| `super_table_col_name` | Standardized name for the medication | norepinephrine, vancomycin |
| `med_class` | Category of the medication | anti-infective, vasopressor |
| Other columns | Additional metadata | (varies) |

**Processing**: 
- The pipeline uses this file to categorize medications into groups like anti-infectives and vasopressors
- It enables standardization of medication names across different institutions
- It allows filtering to include only medications of interest

### 2. Lab Groupings

**File Path**: Specified by `dataConfig["dictionary_paths"]["grouping_types"]["grouping_labs"]`

**Purpose**: Maps lab test identifiers to standardized names and indicates which tests to import.

**Structure**:

| Column Name | Description | Example Values |
|-------------|-------------|----------------|
| `component_id` | Unique identifier for the lab test | 10001, 10002 |
| `super_table_col_name` | Standardized name for the lab test | WBC, LACTATE |
| `import` | Flag indicating whether to import this lab | Yes, No |
| `physionet` | Flag indicating if used in PhysioNet | Yes, No |
| Other columns | Additional metadata | (varies) |

**Processing**:
- Used to filter lab tests to include only those marked with `import = "Yes"`
- Standardizes lab test names across different institutions
- Separates numeric and string-valued lab results for appropriate processing

### 3. Bed Labels

**File Path**: Specified by `dataConfig["dictionary_paths"]["grouping_types"]["bed_labels"]`

**Purpose**: Categorizes hospital bed units to identify different types of care settings.

**Structure**:

| Column Name | Description | Example Values |
|-------------|-------------|----------------|
| `bed_unit` | Identifier for the hospital unit | MICU, SICU, ED |
| `icu` | Flag indicating ICU status | 0, 1 |
| `imc` | Flag indicating intermediate care status | 0, 1 |
| `ed` | Flag indicating emergency department status | 0, 1 |
| `procedure` | Flag indicating procedure area status | 0, 1 |
| Other columns | Additional unit metadata | (varies) |

**Processing**:
- Used to classify patient locations into categories (ICU, IMC, ED, procedure areas)
- Enables tracking of patient movements between different types of care settings
- Standardizes unit names across different institutions

### 4. Fluid Groupings

**File Path**: Specified by `dataConfig["dictionary_paths"]["grouping_types"]["grouping_fluids"]`

**Purpose**: Categorizes fluid inputs and outputs into standardized groups.

**Structure**:

| Column Name | Description | Example Values |
|-------------|-------------|----------------|
| Identifier column | Unique identifier for the fluid type | (varies) |
| Category columns | Classifications for different fluid types | Crystalloid, Colloid, Blood Product |
| Other columns | Additional metadata | (varies) |

**Processing**:
- Standardizes fluid types across different institutions
- Enables grouping of similar fluid types for analysis
- Helps distinguish between different types of fluid inputs and outputs

### 5. Comorbidity Mapping Files

**File Paths**: 
- `ICD10_quan_deyo`: Maps to Quan-Deyo comorbidity categories
- `ICD10_quan_elix`: Maps to Quan-Elixhauser comorbidity categories

**Purpose**: Maps ICD-10 diagnosis codes to standardized comorbidity categories.

**Structure**:

| Column Name | Description | Example Values |
|-------------|-------------|----------------|
| `ICD10` | ICD-10 diagnosis code | I10, E11.9 |
| Comorbidity categories | Flags for different comorbidities | 0, 1 |
| Other columns | Additional metadata | (varies) |

**Processing**:
- Used to categorize diagnoses into standardized comorbidity groups
- Enables calculation of comorbidity indices
- Standardizes comorbidity definitions across different institutions

## Creating Custom Grouping Files

When adapting the pipeline to a new dataset or institution, you'll need to create custom grouping files that map your institution's specific codes and identifiers to the standardized format expected by the pipeline.

### General Steps:

1. **Identify Source Data**: Determine what codes/identifiers are used in your source data
2. **Create Mapping**: Map these codes to standardized names and categories
3. **Validate**: Ensure the mapping is complete and accurate
4. **Format**: Save the mapping as a CSV file with the required columns
5. **Configure**: Update the dataConfig to point to your custom mapping files

### Example: Creating a Lab Grouping File

1. Extract a list of all unique lab component IDs from your source data
2. For each ID, determine:
   - The standardized name to use (`super_table_col_name`)
   - Whether it should be imported (`import` = "Yes" or "No")
   - Whether it's a numeric or string value
3. Create a CSV with columns: `component_id`, `super_table_col_name`, `import`, etc.
4. Save this file and update the `dataConfig["dictionary_paths"]["grouping_types"]["grouping_labs"]` path

## Best Practices

1. **Standardize Names**: Use consistent, clear names for standardized columns
2. **Document Mappings**: Keep documentation of how source codes map to standardized categories
3. **Version Control**: Keep grouping files under version control to track changes
4. **Regular Updates**: Update grouping files as new codes or categories are introduced
5. **Validation**: Regularly validate grouping files against source data to ensure completeness 
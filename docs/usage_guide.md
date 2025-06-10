# Sepy Data Pipeline: Usage Guide

This guide provides step-by-step instructions for implementing the Sepy data processing pipeline with a new clinical dataset. It covers everything from initial setup to data validation and troubleshooting.

## Prerequisites

Before starting, ensure you have:

1. Python 3.6+ installed
2. Required packages: pandas, numpy, pyyaml
3. Access to the clinical data files
4. Basic understanding of the data's structure and content

## Implementation Steps

### Step 1: Install Required Packages

```bash
pip install pandas numpy pyyaml
```

### Step 2: Prepare Your Data Files

1. **Organize Flat Files**: Ensure all your clinical data files are in CSV format
2. **Review File Structure**: Confirm each file contains the required columns as described in `data_pipeline.md`
3. **Clean Data**: Address any obvious data quality issues before processing

### Step 3: Create Grouping Files

Create or adapt the following mapping files for your institution:

1. **Medication Groupings**: Map medication IDs to standardized names and categories
2. **Lab Groupings**: Map lab test IDs to standardized names
3. **Bed Labels**: Categorize hospital units (ICU, ED, etc.)
4. **Fluid Groupings**: Categorize fluid inputs and outputs
5. **Comorbidity Mappings**: Map ICD codes to comorbidity categories

Refer to `grouping_files.md` for detailed specifications of each file.

### Step 4: Create Configuration Files

Create three YAML configuration files:

1. **File Dictionary**: Map file identifiers to physical file paths
2. **Import Configuration**: Define data cleaning and processing parameters
3. **Data Configuration**: Specify paths to grouping files

Refer to `configuration_files.md` for detailed specifications of each file.

Example:

```yaml
# file_dictionary.yaml
ENCOUNTER: /path/to/encounters.csv
DEMOGRAPHICS: /path/to/demographics.csv
# ... other file paths

# import_config.yaml
na_values:
  - NULL
  - ""
  # ... other NA values
vital_col_names:
  - heart_rate
  # ... other vital signs

# data_config.yaml
dictionary_paths:
  grouping_types:
    infusion_meds: /path/to/infusion_med_groupings.csv
    # ... other grouping files
```

### Step 5: Initialize the Pipeline

Create a Python script to initialize and run the pipeline:

```python
import pandas as pd
import numpy as np
import yaml
import logging
from sepyIMPORT import sepyIMPORT

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load configurations
with open('file_dictionary.yaml', 'r') as file:
    file_dictionary = yaml.safe_load(file)
    
with open('import_config.yaml', 'r') as file:
    sepyIMPORTConfigs = yaml.safe_load(file)
    
with open('data_config.yaml', 'r') as file:
    dataConfig = yaml.safe_load(file)

# Initialize the sepyIMPORT class
importer = sepyIMPORT(file_dictionary, sepyIMPORTConfigs, dataConfig)
```

### Step 6: Import and Process Data

Import each data type using the appropriate method:

```python
# Define common parameters
demographic_drop_cols = ['unnecessary_column1', 'unnecessary_column2']
demographic_date_cols = ['admit_time', 'discharge_time']
demographic_index_col = 'csn'

# Import demographics
importer.import_demographics(
    drop_cols=demographic_drop_cols,
    index_col=demographic_index_col,
    date_cols=demographic_date_cols
)

# Import vitals
vitals_drop_cols = ['unnecessary_column1', 'unnecessary_column2']
vitals_date_cols = ['recorded_time']
vitals_index_col = 'csn'
vitals_merge_cols = [['bp_systolic', 'systolic_bp', 'systolic_bp']]  # Example merge columns

importer.import_vitals(
    drop_cols=vitals_drop_cols,
    index_col=vitals_index_col,
    date_cols=vitals_date_cols,
    merge_cols=vitals_merge_cols
)

# Continue importing other data types...
```

### Step 7: Access and Validate Processed Data

After importing, access the processed data through the importer's attributes:

```python
# Access processed DataFrames
demographics_df = importer.df_demographics
vitals_df = importer.df_vitals
labs_df = importer.df_labs

# Validate data
print(f"Demographics shape: {demographics_df.shape}")
print(f"Vitals shape: {vitals_df.shape}")
print(f"Labs shape: {labs_df.shape}")

# Check for missing values
print("Missing values in demographics:")
print(demographics_df.isnull().sum())

# Explore data
print("Demographics sample:")
print(demographics_df.head())
```

### Step 8: Save Processed Data (Optional)

Save the processed data for future use:

```python
# Save to CSV
importer.df_demographics.to_csv('processed_demographics.csv')
importer.df_vitals.to_csv('processed_vitals.csv')
importer.df_labs.to_csv('processed_labs.csv')

# Or save to pickle for faster loading
importer.df_demographics.to_pickle('processed_demographics.pkl')
importer.df_vitals.to_pickle('processed_vitals.pkl')
importer.df_labs.to_pickle('processed_labs.pkl')
```

## Common Issues and Troubleshooting

### Missing Columns

**Problem**: The pipeline expects columns that don't exist in your data.
**Solution**: Review your data's structure and either:
1. Add the missing columns (with appropriate values)
2. Modify the pipeline code to handle the missing columns
3. Create a preprocessing step to rename existing columns to match expected names

### Data Type Errors

**Problem**: Errors during numeric conversion or date parsing.
**Solution**:
1. Check data format consistency
2. Add problematic values to the `na_values` list in the import configuration
3. Implement custom preprocessing for problematic columns

### Memory Issues

**Problem**: Processing large datasets causes memory errors.
**Solution**:
1. Increase system memory
2. Process data in chunks
3. Filter data to include only necessary rows/columns before processing
4. Use more efficient data types (e.g., categorical data for strings)

### Missing Mappings

**Problem**: Entities in your data don't have corresponding entries in mapping files.
**Solution**:
1. Update mapping files to include missing entities
2. Log missing entities for review
3. Implement a fallback strategy for unmapped entities

## Best Practices

1. **Start Small**: Test with a subset of data before processing the entire dataset
2. **Validate Early**: Check data quality after each processing step
3. **Document Customizations**: Keep track of any modifications to the standard pipeline
4. **Version Control**: Maintain version control for your code and configuration files
5. **Log Everything**: Implement comprehensive logging to track processing steps and issues

## Next Steps

After successfully processing your data, you can:

1. Perform exploratory data analysis
2. Develop clinical prediction models
3. Create visualizations and dashboards
4. Integrate with other clinical systems or research projects 
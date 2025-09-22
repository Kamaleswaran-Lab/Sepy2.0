# Column Mapping System for sepyIMPORT

## Overview

The column mapping system allows sepyIMPORT to work with datasets that have different column naming conventions. Instead of requiring datasets to match exact column names, you can now specify mappings in your configuration file.

## How It Works

1. **Define mappings** in your YAML configuration file
2. **sepyIMPORT automatically translates** dataset-specific column names to standardized internal names
3. **All processing continues** using the standardized names
4. **Zero code changes** needed for new datasets - just update the configuration

## Configuration Syntax

Add a `column_mappings` section to any import function in your YAML config:

```yaml
yearly_instance:
  import_encounters:
    file_key: "ENCOUNTER"
    index_col: ["csn"]
    date_cols: ["hospital_admission_date_time"]
    # NEW: Column mappings for this dataset
    column_mappings:
      csn: "encounter_id"                    # Your dataset uses 'encounter_id' instead of 'csn'
      pat_id: "patient_id"                   # Your dataset uses 'patient_id' instead of 'pat_id'
      hospital_admission_date_time: "admit_datetime"  # Different datetime column name
```

## Mapping Format

The mapping format is: `standard_name: dataset_name`

- **standard_name**: The column name expected by sepyIMPORT
- **dataset_name**: The actual column name in your dataset

## Common Column Mappings

### Encounters
```yaml
column_mappings:
  csn: "encounter_id"
  pat_id: "patient_id" 
  hospital_admission_date_time: "admit_datetime"
  hospital_discharge_date_time: "discharge_datetime"
  encounter_type: "visit_type"
  age: "patient_age"
```

### Vitals
```yaml
column_mappings:
  csn: "encounter_id"
  recorded_time: "measurement_timestamp"
  heart_rate: "hr"
  blood_pressure_systolic: "sbp"
  blood_pressure_diastolic: "dbp"
  temperature: "temp"
  oxygen_saturation: "spo2"
```

### Labs
```yaml
column_mappings:
  csn: "encounter_id"
  pat_id: "patient_id"
  collection_time: "specimen_collection_datetime"
  lab_result_time: "result_available_datetime"
  component_id: "test_id"
  lab_result: "test_result"
```

### Medications
```yaml
column_mappings:
  csn: "encounter_id"
  pat_id: "patient_id"
  medication_id: "drug_id"
  med_order_time: "order_datetime"
  med_start: "infusion_start_time"
  med_stop: "infusion_end_time"
```

## Examples

### Example 1: Basic Usage

Your dataset has these columns:
- `encounter_id` instead of `csn`
- `patient_id` instead of `pat_id`

Configuration:
```yaml
import_encounters:
  file_key: "ENCOUNTER"
  index_col: ["csn"]  # Still use standard names in config
  column_mappings:
    csn: "encounter_id"     # Map to your dataset's name
    pat_id: "patient_id"    # Map to your dataset's name
```

### Example 2: Partial Mappings

Only some columns need mapping:
```yaml
import_vitals:
  file_key: "VITALS"
  index_col: ["csn"]
  date_cols: ["recorded_time"]
  column_mappings:
    recorded_time: "measurement_timestamp"  # Only this column needs mapping
    # Other columns like 'csn', 'heart_rate' use standard names
```

### Example 3: No Mappings Needed

If your dataset already uses standard column names:
```yaml
import_encounters:
  file_key: "ENCOUNTER"
  index_col: ["csn"]
  # No column_mappings section needed
```

## Complete Example Configuration

See `configurations/example_new_dataset_config.yaml` for a complete example showing column mappings for all data types.

## Backward Compatibility

- **Existing configurations continue to work** without any changes
- **Column mappings are optional** - only add them when needed
- **No breaking changes** to existing functionality

## Benefits

1. **Dataset Agnostic**: Easily adapt to new datasets
2. **No Code Changes**: Only configuration updates needed
3. **Maintainable**: Centralized mapping logic
4. **Flexible**: Map only the columns that need it
5. **Safe**: Extensive validation and error handling

## Testing

Run the test script to see the column mapping system in action:
```bash
python test_column_mapping.py
```

## Troubleshooting

### Common Issues

1. **Missing required columns**: Check that your mappings cover all required columns
2. **Typos in column names**: Verify dataset column names match exactly
3. **Case sensitivity**: Column names are case-sensitive

### Debug Tips

1. Check the logs for mapping information
2. Use the test script to verify your mappings
3. Start with a small subset of data to test mappings

## Technical Details

- **ColumnMapper class**: Handles all mapping logic
- **Bidirectional mapping**: Can map in both directions
- **Validation**: Ensures required columns are present
- **Performance**: Minimal overhead, only applied when needed

## Migration Guide

To migrate an existing dataset with different column names:

1. **Identify differences**: Compare your dataset columns to the expected standard names
2. **Create mappings**: Add `column_mappings` sections to your YAML config
3. **Test**: Use the test script or start with a small data subset
4. **Deploy**: Run your full data import with the new configuration

For questions or issues, refer to the test script examples or the complete example configuration file.

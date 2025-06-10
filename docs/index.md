# Sepy Data Processing Pipeline Documentation

Welcome to the documentation for the Sepy data processing pipeline, designed for processing clinical data from electronic health records (EHR). This documentation provides a comprehensive guide to understanding, implementing, and customizing the pipeline for various healthcare datasets.

## Documentation Sections

1. [**Data Pipeline Overview**](data_pipeline.md)
   - Description of the pipeline architecture
   - Required flat files and their expected columns
   - Data processing steps for each data type

2. [**Configuration Files**](configuration_files.md)
   - File dictionary configuration
   - Import configuration (sepyIMPORTConfigs)
   - Data configuration (dataConfig)
   - Customization options

3. [**Grouping Files**](grouping_files.md)
   - Medication groupings
   - Lab groupings
   - Bed labels
   - Fluid groupings
   - Comorbidity mappings
   - Creating custom grouping files

4. [**Usage Guide**](usage_guide.md)
   - Step-by-step implementation instructions
   - Troubleshooting common issues
   - Best practices
   - Next steps

## Quick Start

To quickly get started with the Sepy data processing pipeline:

1. Install required packages: `pip install pandas numpy pyyaml`
2. Prepare your clinical data files in CSV format
3. Create or adapt the necessary grouping files
4. Create the configuration files (file dictionary, import config, data config)
5. Initialize the pipeline and import your data
6. Access and validate the processed data

See the [Usage Guide](usage_guide.md) for detailed instructions.

## Key Features

- Standardized processing of diverse clinical data types
- Flexible configuration to adapt to different institutional data formats
- Comprehensive data cleaning and transformation
- Mapping of institution-specific codes to standardized terminology
- Efficient handling of large healthcare datasets

## Requirements

- Python 3.6+
- pandas
- numpy
- pyyaml

## Support and Contribution

For questions, issues, or contributions, please contact the project maintainers or submit issues and pull requests to the project repository.

## License

This project is licensed under [Your License] - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```
[Citation information for the Sepy project]
```

## Acknowledgments

- Christopher S. Josef, MD - Original author
- Jack F. Regan - Contributor
- Mehak Arora - Contributor
- [Other contributors as appropriate] 
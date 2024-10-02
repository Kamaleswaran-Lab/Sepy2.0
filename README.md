# Sepy

These scripts are used to generate super-tables from raw EMR data. 

##  Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Processing Pipeline

The raw EMR data used as input to the pipeline is assumed to be in a tabular format (stored as .dsv files). (Currently, the code does not handle other file formats but that is an easy functionality to add.) 

The following data is required, in the form of separate dsv files, to generate all the variables of the super-tables: 
1. Infusion Meds
2. Labs
3. Vitals
4. Ventilator Information
6. Demographics 
7. GCS
8. Encounters
9. Cultures
10. Bed Locations
11. Procedures
12. Diagnosis

The pipeline runs in two phases: 
1. Making a Yearly Pickle: Creating a single dictionary for each year, the items in which are dataframes containing relevant clinical features derived from the above files.  
2. Making a Encounter Dictionaries: Creating a single dictionary for each encounter, the items of which are dataframes containing relevant clinical features derived from the yearly pickle, including the final supertable. A full list and explaination of the items in this encounter-specific dictionary are provided next. 

### TODO:
continue documentation - list of clinical features in the encounter specific pickles
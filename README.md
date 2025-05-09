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

## Configuration File Setup

- `num_processes`
  - **Description**: Number of parallel processes to use for data processing.
  - **Recommended**: Set based on SLURM job CPU allocation.
- `processor_assignment`
  - **Description**: SLURM processor ID to assign this job to (if applicable).
- `make_pickle`
  - **Options**: `"yes"` or `"no"`
  - **Description**: Controls whether intermediate pickle files are created.
- `make_dictionary`
  - **Options**: `"yes"` or `"no"`
  - **Description**: Controls whether dictionaries for variables or groupings are created.

---

Define the absolute paths for all required input and output data:

- `data_path`: Path to the raw EMR dataset directory.
- `groupings_path`: Path to lab/medication groupings.
- `pickle_output_path`: Directory where pickle files should be saved.
- `dictionary_output_path`: Directory where processed dictionaries should be saved.
- `bed_unit_csv_fname`: CSV file mapping bed units to ICU types.
- `variable_bounds_csv_fname`: CSV specifying variable bounds for filtering.
- `dialysis_info_csn_fname`: File listing dialysis patients' CSNs.

---

- `unique_encounters_path`
  - **Description**: Full path to the `.dsv` file containing unique patient encounters.
- `dataset_identifier`
  - **Description**: Short tag identifying your dataset (e.g., `"gr_y"`).
- `years_to_scrape_data`
  - **Description**: List of years to scrape dsv data (as strings)

---

Control the scope of included patient encounters:

- `encounter_type`: `"Inpatient"`, `"Emergency"`, or `"all"`
- `age`: `"adult"`, `"pediatric"`, or `"all"`
- `unique_enc_filter`: `"yes"` or `"no"`
- `unique_enc`: *(Optional)* List of specific CSNs to include if filtering is `"yes"`.

---

Control how comorbidities and grouped variables are applied:

- `dictionary_paths`:
  - `comorbidity_types`: List of coding systems to summarize comorbidities.
  - `grouping_types`: Grouping categories for meds, labs, etc.
  - `year_types`: List of [internal label, source label] pairs for each table type.

---

Define how each table should be imported:

- For each `import_*` key:
  - `drop_cols`: Columns to exclude from the table.
  - `index_col`: Column(s) to use as row identifiers.
  - `date_cols`: Columns that should be parsed as dates.
  - `numeric_cols` (optional): Columns to treat as numeric.
  - `merge_cols` (optional): Columns to merge into one.
  - `group_cols` (for labs): Columns for grouping variable info.

---

Choose which comorbidity classification systems to summarize:

- For each `comorbidity_summary` key:
  - Choose a value to summarize

---

Set summary output directory and summary types:

- `sepsis_summary`: Folder name to store outputs.
- `sepsis_summary_types`: List of summary table types to generate.

## Slurm JOb

Set years (list) to analyze and configuration file (yaml file path) to access
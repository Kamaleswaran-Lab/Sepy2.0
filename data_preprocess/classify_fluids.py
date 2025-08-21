import pandas as pd
import re

def load_medication_mapping(mapping_file='../groupings/em_all_infusion_meds.csv'):
    """
    Load medication class mapping from the infusion meds file.
    
    Parameters:
    mapping_file: Path to the CSV file with medication classifications
    
    Returns:
    Dictionary mapping med_name to (med_class, med_subclass)
    """
    try:
        mapping_df = pd.read_csv(mapping_file)
        # Create mapping dictionary: med_name -> (med_class, med_subclass)
        med_mapping = {}
        for _, row in mapping_df.iterrows():
            med_name = row['med_name'].lower().strip() if pd.notna(row['med_name']) else ''
            med_class = row['med_class'] if pd.notna(row['med_class']) and row['med_class'] != 'NULL' else None
            med_subclass = row['super_table_col_name'] if pd.notna(row['super_table_col_name']) else None
            
            if med_name:
                med_mapping[med_name] = (med_class, med_subclass)
        
        return med_mapping
    except FileNotFoundError:
        print(f"Warning: Mapping file {mapping_file} not found. Proceeding without medication class mapping.")
        return {}

def classify_iv_fluids(df, mapping_file='../groupings/em_all_infusion_meds.csv'):
    """
    Classify medications as IV fluids vs medications with detailed fluid types,
    and add medication class information.
    
    Parameters:
    df: DataFrame with 'med_name' column
    mapping_file: Path to the CSV file with medication classifications
    
    Returns:
    DataFrame with added 'is_fluids', 'fluid_type', 'med_class', and 'med_subclass' columns
    """
    
    # Load medication class mapping
    med_mapping = load_medication_mapping(mapping_file)
    
    # Create copies for classification
    df = df.copy()
    df['med_name_lower'] = df['med_name'].str.lower().str.strip()
    df['is_fluids'] = False
    df['fluid_type'] = 'Medication'
    df['med_class'] = None
    df['med_subclass'] = None
    
    # Define classification functions
    def is_isotonic_crystalloid(name):
        patterns = [
            r'sodium chloride 0\.9%',
            r'normal saline',
            r'0\.9%.*sodium.*chloride',
            r'lactated.*ringer(?!.*dextrose)',
            r'ringer.*lactate(?!.*dextrose)',
            r'dextrose.*0\.9%.*nacl',
            r'dextrose.*normal saline'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_hypotonic_crystalloid(name):
        patterns = [
            r'sodium chloride 0\.45%',
            r'0\.45%.*sodium.*chloride',
            r'half.*saline',
            r'dextrose 5%.*water(?!.*nacl|.*saline|.*ringer)',
            r'd5w(?!.*nacl|.*saline|.*ringer)',
            r'dextrose.*0\.2%',
            r'dextrose.*0\.45%',
            r'dextrose 5%.*0\.2%',
            r'dextrose 5%.*0\.45%',
            r'd5w?\s+0\.45%ns',
            r'd5w?\s+0\.2%ns'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_hypertonic_crystalloid(name):
        patterns = [
            r'sodium chloride (3|5)%',
            r'(3|5)%.*sodium',
            r'dextrose (10|20|30|50|70)%',
            r'd(10|15|20|25|30|50|70)w',
            r'^mannitol$',
            r'd\d{2,}\.?\d*w',  # D10W, D12.5W, D15W, D20W, etc. (>=10% dextrose)
            r'd1[0-9]\.?\d*w',  # D10W through D19.xW
            r'd[2-9]\d\.?\d*w', # D20W, D25W, etc.
            r'd1[0-9]\.?\d*w?\s+0\.\d+%ns',  # D12.5W 0.2%NS, D20W 0.45%NS, etc.
            r'd1[0-9]\.?\d*%?w' # D10%W, D12.5%W, D15%W, etc.
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_mixed_crystalloid(name):
        patterns = [
            r'dextrose.*lactated.*ringer'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_natural_colloid(name):
        patterns = [
            r'^albumin',
            r'albumin human'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_synthetic_colloid(name):
        patterns = [
            r'^hetastarch$',
            r'^dextran$'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_potassium_solution(name):
        patterns = [
            r'^potassium chloride$',
            r'^kcl$',
            r'potassium chloride-sodium chloride',
            r'^potassium acetate$',
            r'^potassium phosphate$',
            r'potassium.*injection',
            r'kcl.*injection'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_calcium_solution(name):
        patterns = [
            r'^calcium chloride$',
            r'^cacl2$',
            r'^calcium gluconate$',
            r'^calcium glubionate$',
            r'calcium.*injection'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_magnesium_solution(name):
        patterns = [
            r'^magnesium sulfate$',
            r'^mgso4$'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_bicarbonate_solution(name):
        patterns = [
            r'^sodium bicarbonate$',
            r'^nahco3$'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_phosphate_solution(name):
        patterns = [
            r'^sodium phosphate',
            r'phosphate.*injection',
            r'sodium glycerophosphate',
            r'^potassium phosphate',
            r'phosphate.*sodium phosphate'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_other_electrolyte_solution(name):
        """Catch other electrolyte solutions not in specific categories"""
        patterns = [
            r'^sodium acetate$',
            r'^potassium acetate$',
            r'citric acid.*sodium citrate',
            r'sodium.*citrate$',
            r'magnesium.*citrate'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_premix_solution(name):
        """Identify premix solutions and diluents"""
        patterns = [
            r'^premix',
            r'premix diluent',
            r'premix.*ns',
            r'premix.*normal saline',
            r'premix.*dextrose',
            r'premix.*d5w',
            r'premix.*water',
            r'.*premix$'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_iv_line_fluid(name):
        """Identify IV line fluids and stabilizer solutions"""
        patterns = [
            r'.*line.*iv.*fluid',
            r'standard.*line.*fluid',
            r'iv.*stabilizer.*solution',
            r'pca.*iv.*syringe',
            r'.*continuous.*bladder.*irrigation',
            r'.*bladder.*irrigation.*\d+ml'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_sterile_water(name):
        """Identify sterile water solutions"""
        patterns = [
            r'^sterile water',
            r'water.*sterile',
            r'sterile.*water.*injection'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_glucose_solution(name):
        """Catch glucose solutions that aren't dextrose"""
        patterns = [
            r'glucose.*solution',
            r'glucose.*injection'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_electrolyte_solution_advanced(name):
        """Enhanced electrolyte solution detection"""
        patterns = [
            r'plasma.*lyte',
            r'electrolyte.*solution',
            r'intravenous.*electrolyte',
            r'ringer.*solution',
            r'hartmann.*solution',
            r'multiple.*electrolyte',
            r'balanced.*salt.*solution'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_parenteral_nutrition(name):
        """Identify parenteral nutrition solutions"""
        patterns = [
            r'amino.*acid.*injection',
            r'parenteral.*nutrition',
            r'hyperal',
            r'procalamine',
            r'clinisol',
            r'fat.*emulsion.*intravenous',
            r'amino acids.*glycerin.*electrolytes'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_dialysis_solution(name):
        """Identify dialysis solutions"""
        patterns = [
            r'dianeal',
            r'prismasol',
            r'prismasate',
            r'dialysate',
            r'peritoneal.*solution'
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    def is_excluded_medication(name):
        """Explicitly exclude these from fluid classification"""
        patterns = [
            r'iron dextran',
            r'amiodarone.*premix',  # This specific premix is a medication, not fluid
            r'oral',
            r'tablet',
            r'capsule',
            r'topical',
            r'ophthalmic',
            r'otic',
            r'nasal',
            r'inhaler',
            r'vaccine',
            r'desensitization.*protocol',
            r'dosing.*by.*pharmacist',
            r'anes.*epidural',  # Anesthetic epidural solutions
            r'anes.*drip',  # Anesthetic drips with medications
            r'insulin.*pump.*patient',  # Insulin pump settings
            r'study',  # Research study medications
        ]
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)
    
    for idx, row in df.iterrows():
        name = row['med_name_lower']
        original_name = row['med_name'].lower().strip()
        
        # Apply medication class mapping if available
        if original_name in med_mapping:
            med_class, med_subclass = med_mapping[original_name]
            df.at[idx, 'med_class'] = med_class
            df.at[idx, 'med_subclass'] = med_subclass
        
        # First check exclusions
        if is_excluded_medication(name):
            continue
            
        # Check fluid types in order of specificity
        if is_premix_solution(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Premix'
        elif is_isotonic_crystalloid(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Isotonic Crystalloid'
        elif is_hypotonic_crystalloid(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Hypotonic Crystalloid'
        elif is_hypertonic_crystalloid(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Hypertonic Crystalloid'
        elif is_mixed_crystalloid(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Mixed Crystalloid'
        elif is_natural_colloid(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Natural Colloid'
        elif is_synthetic_colloid(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Synthetic Colloid'
        elif is_electrolyte_solution_advanced(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Electrolyte Solution'
        elif is_potassium_solution(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Electrolyte Solution (Potassium)'
        elif is_calcium_solution(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Electrolyte Solution (Calcium)'
        elif is_magnesium_solution(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Electrolyte Solution (Magnesium)'
        elif is_bicarbonate_solution(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Electrolyte Solution (Bicarbonate)'
        elif is_phosphate_solution(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Electrolyte Solution (Phosphate)'
        elif is_other_electrolyte_solution(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Electrolyte Solution (Other)'
        elif is_parenteral_nutrition(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Parenteral Nutrition'
        elif is_dialysis_solution(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Dialysis Solution'
        elif is_iv_line_fluid(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'IV Line Fluid'
        elif is_glucose_solution(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Glucose Solution'
        elif is_sterile_water(name):
            df.at[idx, 'is_fluids'] = True
            df.at[idx, 'fluid_type'] = 'Sterile Water'
    
    # Clean up temporary column
    df = df.drop('med_name_lower', axis=1)
    
    return df

def main():
    df = pd.read_csv('em_unique_infusion_meds.csv')

    # Apply classification with medication mapping
    df_classified = classify_iv_fluids(df, '../groupings/em_all_infusion_meds.csv')

    # Display results
    print("Classification Summary:")
    print("Fluid Types:")
    print(df_classified['fluid_type'].value_counts())
    
    print("\nMedication Classes:")
    print(df_classified['med_class'].value_counts(dropna=False))
    
    print("\nMedication Subclasses:")
    print(df_classified['med_subclass'].value_counts(dropna=False))

    print(f"\nTotal Fluids: {df_classified['is_fluids'].sum()}")
    print(f"Total Medications: {(~df_classified['is_fluids']).sum()}")
    print(f"Medications with class mapping: {df_classified['med_class'].notna().sum()}")

    # Show all IV fluids
    fluids_df = df_classified[df_classified['is_fluids'] == True].copy()
    print(f"\nAll {len(fluids_df)} IV Fluids:")
    for idx, row in fluids_df.iterrows():
        med_class_info = f" (Class: {row['med_class']}, Subclass: {row['med_subclass']})" if pd.notna(row['med_class']) else ""
        print(f"{row.iloc[0]}: {row['med_name']} -> {row['fluid_type']}{med_class_info}")

    # Save results
    df_classified.to_csv('classified_medications.csv', index=False)

    # Just the fluids with all classification info
    fluids_only = df_classified[df_classified['is_fluids'] == True][['med_name', 'fluid_type', 'med_class', 'med_subclass']].copy()
    fluids_only.to_csv('iv_fluids_only.csv', index=False)
    
    # Save medication class summary
    med_class_summary = df_classified[['med_name', 'med_class', 'med_subclass']].copy()
    med_class_summary.to_csv('medication_classes.csv', index=False)

if __name__ == "__main__":
    main()
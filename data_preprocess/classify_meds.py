import pandas as pd
import re

def classify_medications(df):
    """
    Classify medications into therapeutic classes and subclasses.
    
    Parameters:
    df: DataFrame with 'med_name' column
    
    Returns:
    DataFrame with added 'med_class' and 'super_table_col_name' columns
    """
    
    df = df.copy()
    df['med_name_lower'] = df['med_name'].str.lower().str.strip()
    df['med_class'] = 'other'
    df['super_table_col_name'] = 'other'
    
    def classify_medication(name):
        """Return (med_class, super_table_col_name) for a medication name"""
        
        # ANTI-INFECTIVES
        # Aminoglycosides
        if re.search(r'gentamicin|tobramycin|amikacin|streptomycin|neomycin|kanamycin|netilmicin|plazomicin', name, re.IGNORECASE):
            return ('anti-infective', 'aminoglycosides')
        
        # Antifungals
        elif re.search(r'fluconazole|caspofungin|amphotericin|voriconazole|itraconazole|ketoconazole|micafungin|anidulafungin|posaconazole', name, re.IGNORECASE):
            return ('anti-infective', 'antifungals')
        
        # Antivirals
        elif re.search(r'acyclovir|ganciclovir|foscarnet|cidofovir|ribavirin|remdesivir|oseltamivir|valacyclovir|famciclovir', name, re.IGNORECASE):
            return ('anti-infective', 'antiviral')
        
        # Carbapenems
        elif re.search(r'meropenem|imipenem|ertapenem|doripenem|biapenem', name, re.IGNORECASE):
            return ('anti-infective', 'carbapenems')
        
        # Cephalosporins
        elif re.search(r'ceftriaxone|cefazolin|ceftaroline|cefepime|ceftolozane|cefiderocol|cephalexin|cefuroxime|ceftazidime|cefoxitin|cefdinir|cefpodoxime', name, re.IGNORECASE):
            return ('anti-infective', 'cephalosporins')
        
        # Fluoroquinolones
        elif re.search(r'ciprofloxacin|levofloxacin|moxifloxacin|gemifloxacin|delafloxacin|norfloxacin|ofloxacin', name, re.IGNORECASE):
            return ('anti-infective', 'fluoroquinolones')
        
        # Glycopeptides
        elif re.search(r'vancomycin|teicoplanin|telavancin|oritavancin|dalbavancin', name, re.IGNORECASE):
            return ('anti-infective', 'glycopeptide')
        
        # Lincosamides
        elif re.search(r'clindamycin|lincomycin', name, re.IGNORECASE):
            return ('anti-infective', 'lincosamides')
        
        # Lipopeptide
        elif re.search(r'daptomycin', name, re.IGNORECASE):
            return ('anti-infective', 'lipopeptide')
        
        # Macrolides
        elif re.search(r'azithromycin|clarithromycin|erythromycin|fidaxomicin|roxithromycin|spiramycin', name, re.IGNORECASE):
            return ('anti-infective', 'macrolide antibiotics')
        
        # Metronidazole
        elif re.search(r'^metronidazole$', name, re.IGNORECASE):
            return ('anti-infective', 'metronidazole')
        
        # Monobactams
        elif re.search(r'aztreonam', name, re.IGNORECASE):
            return ('anti-infective', 'monobactams')
        
        # Oxazolidinones
        elif re.search(r'linezolid|tedizolid', name, re.IGNORECASE):
            return ('anti-infective', 'oxazolidinones')
        
        # Penicillins
        elif re.search(r'ampicillin|penicillin|piperacillin|nafcillin|oxacillin|amoxicillin|ticarcillin|benzylpenicillin|phenoxymethylpenicillin', name, re.IGNORECASE):
            return ('anti-infective', 'penicillins')
        
        # Sulfonamides
        elif re.search(r'trimethoprim.*sulfamethoxazole|sulfadiazine|sulfisoxazole|co-trimoxazole|bactrim|septra', name, re.IGNORECASE):
            return ('anti-infective', 'sulphonamides')
        
        # Tetracyclines
        elif re.search(r'doxycycline|tetracycline|minocycline|tigecycline|demeclocycline', name, re.IGNORECASE):
            return ('anti-infective', 'tetracyclines')
        
        # Antimycobacterial
        elif re.search(r'rifampin|rifabutin|rifapentine|isoniazid|ethambutol|pyrazinamide', name, re.IGNORECASE):
            return ('anti-infective', 'antimycobacterial agents')
        
        # Antiparasitic
        elif re.search(r'pyrimethamine|sulfadiazine|pentamidine|atovaquone|primaquine|chloroquine|hydroxychloroquine|mefloquine|quinine', name, re.IGNORECASE):
            return ('anti-infective', 'antiparasitic')
        
        # VASOPRESSORS
        elif re.search(r'^norepinephrine$|levophed', name, re.IGNORECASE):
            return ('vasopressor', 'norepinephrine')
        elif re.search(r'^epinephrine$|adrenaline', name, re.IGNORECASE):
            return ('vasopressor', 'epinephrine')
        elif re.search(r'^dopamine$', name, re.IGNORECASE):
            return ('vasopressor', 'dopamine')
        elif re.search(r'^dobutamine$', name, re.IGNORECASE):
            return ('vasopressor', 'dobutamine')
        elif re.search(r'phenylephrine|neosynephrine', name, re.IGNORECASE):
            return ('vasopressor', 'phenylephrine')
        elif re.search(r'vasopressin', name, re.IGNORECASE):
            return ('vasopressor', 'vasopressin')
        elif re.search(r'milrinone|inamrinone', name, re.IGNORECASE):
            return ('vasopressor', 'other')
        
        # ANALGESICS
        elif re.search(r'morphine|fentanyl|hydromorphone|oxycodone|codeine|tramadol|nalbuphine|sufentanil|alfentanil|remifentanil|methadone|buprenorphine|hydrocodone', name, re.IGNORECASE):
            return ('analgesic', 'opioids')
        elif re.search(r'acetaminophen|tylenol|paracetamol', name, re.IGNORECASE):
            return ('analgesic', 'non-opioid')
        elif re.search(r'ketorolac|ibuprofen|naproxen|diclofenac|celecoxib|indomethacin|aspirin', name, re.IGNORECASE):
            return ('analgesic', 'nsaid')
        
        # SEDATIVES/ANESTHETICS
        elif re.search(r'propofol|etomidate|ketamine|dexmedetomidine|sevoflurane|isoflurane|desflurane', name, re.IGNORECASE):
            return ('sedative', 'anesthetic')
        elif re.search(r'midazolam|lorazepam|diazepam|alprazolam|clonazepam|temazepam|triazolam', name, re.IGNORECASE):
            return ('sedative', 'benzodiazepine')
        
        # NEUROMUSCULAR BLOCKING AGENTS
        elif re.search(r'rocuronium|vecuronium|succinylcholine|cisatracurium|atracurium|pancuronium|mivacurium', name, re.IGNORECASE):
            return ('neuromuscular_blocker', 'other')
        
        # ANTIEMETICS
        elif re.search(r'ondansetron|metoclopramide|promethazine|prochlorperazine|droperidol|granisetron|dolasetron|palonosetron', name, re.IGNORECASE):
            return ('antiemetic', 'other')
        
        # GASTROINTESTINAL
        elif re.search(r'famotidine|pantoprazole|omeprazole|lansoprazole|esomeprazole|ranitidine|cimetidine|nizatidine', name, re.IGNORECASE):
            return ('gastrointestinal', 'acid_suppressant')
        
        # ANTIHISTAMINES
        elif re.search(r'diphenhydramine|hydroxyzine|cetirizine|loratadine|fexofenadine|chlorpheniramine', name, re.IGNORECASE):
            return ('antihistamine', 'other')
        
        # VITAMINS/SUPPLEMENTS
        elif re.search(r'thiamine|cyanocobalamin|folic.*acid|pyridoxine|vitamin|multivitamin|ascorbic.*acid|phytonadione', name, re.IGNORECASE):
            return ('vitamin_supplement', 'other')
        
        # ANTIARRHYTHMICS
        elif re.search(r'amiodarone|lidocaine|procainamide|quinidine|flecainide|propafenone|sotalol', name, re.IGNORECASE):
            return ('antiarrhythmics', 'other')
        
        # BP LOWERING
        elif re.search(r'labetalol|esmolol|nicardipine|clevidipine|hydralazine|diltiazem|verapamil|metoprolol|propranolol|atenolol|amlodipine|nifedipine', name, re.IGNORECASE):
            return ('bp_lowering', 'other')
        
        # DIURETICS
        elif re.search(r'furosemide|hydrochlorothiazide|spironolactone|bumetanide|torsemide|amiloride|chlorothiazide|indapamide', name, re.IGNORECASE):
            return ('diuretic', 'other')
        
        # ENDOCRINE
        elif re.search(r'insulin|glucagon', name, re.IGNORECASE) or re.search(r'dextrose.*50%|d50', name, re.IGNORECASE):
            return ('endocrine', 'diabetes')
        elif re.search(r'methylprednisolone|hydrocortisone|prednisone|dexamethasone|prednisolone|triamcinolone', name, re.IGNORECASE):
            return ('endocrine', 'corticosteroids')
        
        # ANTICOAGULANTS
        elif re.search(r'heparin|enoxaparin|warfarin|rivaroxaban|apixaban|dabigatran|dalteparin|fondaparinux', name, re.IGNORECASE):
            return ('anticoagulant', 'other')
        
        # REVERSAL AGENTS
        elif re.search(r'naloxone|flumazenil|sugammadex|neostigmine|protamine|atropine|glycopyrrolate', name, re.IGNORECASE):
            return ('reversal_agent', 'other')
        
        # NEUROLOGIC
        elif re.search(r'phenytoin|levetiracetam|valproic.*acid|carbamazepine|lamotrigine|gabapentin|pregabalin|topiramate', name, re.IGNORECASE):
            return ('neurologic', 'anticonvulsant')
        
        # Skip IV fluids and electrolytes (should be classified separately)
        elif re.search(r'sodium chloride.*intravenous|dextrose.*intravenous|lactated.*ringer|albumin|hetastarch|dextran|mannitol|potassium chloride|calcium|magnesium sulfate|sodium bicarbonate', name, re.IGNORECASE):
            return ('fluid_electrolyte', 'other')
        
        # Default to other
        else:
            return ('other', 'other')
    
    # Apply classification
    for idx, row in df.iterrows():
        med_class, super_table_col_name = classify_medication(row['med_name_lower'])
        df.at[idx, 'med_class'] = med_class
        df.at[idx, 'super_table_col_name'] = super_table_col_name
    
    # Clean up temporary column
    df = df.drop('med_name_lower', axis=1)
    
    return df

def get_classification_summary(df):
    """Generate classification summary table"""
    # Group by class and subclass
    summary = df.groupby(['med_class', 'super_table_col_name']).size().reset_index(name='count')
    summary['medication_id'] = summary['count']  # Following your format
    
    # Sort by med_class and count
    summary = summary.sort_values(['med_class', 'count'], ascending=[True, False])
    
    return summary

if __name__ == "__main__":
    df = pd.read_csv('em_unique_infusion_meds.csv')
    
    # Apply classification
    df_classified = classify_medications(df)
    
    # Get summary
    summary = get_classification_summary(df_classified)
    
    # Display results
    print("Medication Classification Summary:")
    print("=" * 50)
    for _, row in summary.iterrows():
        print(f"{row['med_class']},{row['super_table_col_name']},{row['count']},{row['medication_id']}")
    
    # Show total counts by major class
    print("\nMajor Class Totals:")
    major_class_totals = df_classified.groupby('med_class').size().sort_values(ascending=False)
    for med_class, count in major_class_totals.items():
        print(f"{med_class}: {count}")
    
    # Save detailed results
    df_classified.to_csv('medications_classified_detailed.csv', index=False)
    
    # Save summary in your format
    summary.to_csv('medication_classification_summary.csv', index=False)
    
    print(f"\nTotal medications classified: {len(df_classified)}")
    print(f"Detailed results saved to: medications_classified_detailed.csv")
    print(f"Summary saved to: medication_classification_summary.csv")

    fluids_classified = pd.read_csv("classified_medications.csv")
    unclassified_meds = fluids_classified[fluids_classified["med_class"].isna()]["med_name"].tolist()
    for med in unclassified_meds:
        med_class = df_classified.loc[df_classified["med_name"] == med, "med_class"].values[0]
        med_subclass = df_classified.loc[df_classified["med_name"] == med, "super_table_col_name"].values[0]
        fluids_classified.loc[fluids_classified["med_name"] == med, "med_class"] = med_class
        fluids_classified.loc[fluids_classified["med_name"] == med, "med_subclass"] = med_subclass
    
    fluids_classified.to_csv("classified_medications_claude.csv", index=False)
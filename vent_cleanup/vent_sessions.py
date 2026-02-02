import pandas as pd
import numpy as np
from pathlib import Path
import os
import hashlib

def hash_value(value, hash_key = '123'):
    return hashlib.sha256((str(value) + hash_key).encode()).hexdigest()

def shift_date_unix(date):
        """
        Convert date to unix timestamp and drop the first digit
        """
        if pd.isnull(date):
            return date
        try:
            date_utc = pd.to_datetime(date).tz_localize('UTC')
            date_unix = date_utc.timestamp()
            date_unix_deid = float('0' + str(date_unix)[1:])
            return pd.to_datetime(date_unix_deid, unit = 's').strftime('%Y-%m-%d %H:%M:%S')
        except:
            return None 

def unshift_date_unix(deidentified_date):
    """
    Convert deidentified date back to original date
    """
    if pd.isnull(deidentified_date):
        return deidentified_date
    try:
        # Convert deidentified date to timestamp
        deid_timestamp = pd.to_datetime(deidentified_date).timestamp()
        
        # Remove the leading '0' and try each possible first digit (1-2)
        deid_str = str(deid_timestamp)
        if deid_str.startswith('0'):
            deid_str = deid_str[1:]  # Remove the '0'
        
        # Try first digit = 1 (for dates around 2001-2033)
        original_timestamp_1 = float('1' + deid_str)
        original_date_1 = pd.to_datetime(original_timestamp_1, unit='s')
        
        # Try first digit = 2 (for dates around 2033-2065)
        original_timestamp_2 = float('2' + deid_str)
        original_date_2 = pd.to_datetime(original_timestamp_2, unit='s')
        
        # Return the date that seems most reasonable (you may need context)
        # For now, return the one from the 2000s-2030s range
        return original_date_1.strftime('%Y-%m-%d %H:%M:%S')
        
    except:
        return None


def read_discharge_info(years, eroot):
    """Read discharge information from encounter files across multiple years."""
    discharge_info = []
    for year in years:
        encounter_file = eroot / str(year) / f"CJSEPSIS_ENCOUNTER_{year}.dsv"
        encounter_df = pd.read_csv(encounter_file, sep="|")
        encounter_df = encounter_df[['pat_id', 'csn', 'discharge_to']]
        discharge_info.append(encounter_df)
    
    return pd.concat(discharge_info, axis=0)


def read_beds_info(years, eroot):
    """Read bed location information and process ICU types across multiple years."""
    bed_labels = pd.read_csv(os.path.expandvars("$HOME/Sepy2.0/groupings/em_bed_labels.csv"))
    beds_info = []
    for year in years:
        bed_file = eroot / str(year) / f"CJSEPSIS_BEDLOCATION_{year}.dsv"
        bed_df = pd.read_csv(bed_file, sep="|")
        bed_df = bed_df[['pat_id', 'csn', 'bed_unit', 'bed_location_start', 'bed_location_end']]
        bed_df = bed_df.merge(
            bed_labels[['bed_units', 'icu_type']], 
            left_on='bed_unit', 
            right_on='bed_units', 
            how='left'
        )
        beds_info.append(bed_df)
    
    beds_info = pd.concat(beds_info, axis=0)
    
    beds_info["bed_location_start"] = pd.to_datetime(beds_info["bed_location_start"])
    mask = beds_info["icu_type"] == "sicu BEFORE 1/18/2018; cticu ON OR AFTER 1/18/2018"
    cutoff_date = pd.to_datetime("1986-05-11 22:13:20")
    
    beds_info.loc[mask, "icu_type"] = np.where(
        beds_info.loc[mask, "bed_location_start"] < cutoff_date,
        "sicu",
        "cticu"
    )
    
    mask = beds_info["icu_type"] == "cticu BEFORE 1/18/2018; micu ON OR AFTER 1/18/2018"
    beds_info.loc[mask, "icu_type"] = np.where(
        beds_info.loc[mask, "bed_location_start"] < cutoff_date,
        "cticu",
        "micu"
    )
    
    mask = beds_info["icu_type"] == "sicu BEFORE 1/18/2018"
    beds_info.loc[mask, "icu_type"] = np.where(
        beds_info.loc[mask, "bed_location_start"] < cutoff_date,
        "sicu",
        "other"
    )
    
    return beds_info


def read_vent_info(years, eroot):
    """Read ventilator information across multiple years."""
    vent_labels = pd.read_csv(os.path.expandvars("$HOME/Sepy2.0/groupings/em_vent_labels.csv"))
    
    vent_info = []
    for year in years:
        vent_file = eroot / str(year) / f"CJSEPSIS_VENT_{year}.dsv"
        vent_df = pd.read_csv(vent_file, sep="|")
        vent_df = vent_df.merge(
            vent_labels[['vent_name', 'vent_cat']], 
            left_on='vent_mode', 
            right_on='vent_name', 
            how='left'
        )
        vent_info.append(vent_df)
    
    vent_info = pd.concat(vent_info, axis=0)
    vent_info['recorded_time'] = pd.to_datetime(vent_info["recorded_time"])
    
    return vent_info


def read_o2_info(years, eroot):
    """Read oxygen flow rate information across multiple years."""
    device_labels = pd.read_csv(os.path.expandvars("$HOME/Sepy2.0/groupings/unique_o2_devices_mapping_Jan15.csv"))
    
    o2_info = []
    for year in years:
        print(year)
        o2_file = eroot / str(year) / f"VITALS_O2_FLOW_RATE_{year}.csv"
        o2_df = pd.read_csv(o2_file, usecols=['pat_id', 'csn', 'recorded_time',
                                               'unassisted_resp_rate', 'o2_device',
                                               'end_tidal_co2', 'oxygen_flow_rate'])
        o2_df['o2_device'] = o2_df.o2_device.map(dict(zip(device_labels['o2_device'], device_labels['mapping'])))
        o2_info.append(o2_df)
    
    o2_info = pd.concat(o2_info, axis=0)
    o2_info['recorded_time'] = pd.to_datetime(o2_info['recorded_time'])
    
    return o2_info


def read_cpt_info(years, eroot):
    """Read CPT procedure information (code 31500) across multiple years."""
    cpt_info = []
    for year in years:
        print(year)
        cpt_file = eroot / str(year) / f"CJSEPSIS_CPT_{year}.dsv"
        cpt_df = pd.read_csv(cpt_file, sep="|")
        cpt_df = cpt_df.loc[cpt_df.procedure_cpt_cd.astype("str") == "31500"]
        cpt_info.append(cpt_df)
    
    cpt_info = pd.concat(cpt_info, axis=0)
    cpt_info['procedure_dttm'] = pd.to_datetime(cpt_info['procedure_dttm'])
    
    return cpt_info


def read_icd_info(years, eroot):
    """Read ICD procedure information (endotracheal procedures) across multiple years."""
    icd_info = []
    for year in years:
        print(year)
        icd_file = eroot / str(year) / f"CJSEPSIS_ICDPROCEDURES_{year}.dsv"
        icd_df = pd.read_csv(icd_file, sep="|")
        icd_df = icd_df.loc[icd_df.procedure_desc.str.lower().str.contains("endotracheal")]
        icd_info.append(icd_df)
    
    icd_info = pd.concat(icd_info, axis=0)
    icd_info['procedure_date'] = pd.to_datetime(icd_info['procedure_date'])
    
    return icd_info


def read_notes_info(years, vent_csns=None):
    """Read clinical notes information across multiple years, optionally filtered by vent CSNs."""
    notes_path = Path("/data/irb/surgery/pro00114885/EmoryDataset/Notes/ProcessedNotes")
    notes_info = []
    
    for year in years:
        print(year)
        notes_file = notes_path / f"notes_{year}.pkl"
        notes_df = pd.read_pickle(notes_file)
        notes_df['csn_hashed'] = notes_df['CSN'].apply(hash_value)
        
        if vent_csns is not None:
            notes_df = notes_df.loc[notes_df.csn_hashed.isin(vent_csns)]
        
        notes_info.append(notes_df)
    
    notes_info = pd.concat(notes_info, axis=0)
    
    # Filter for specific document types
    notes_selected = notes_info.loc[notes_info.HNAM_DOCUMENT_CLINICAL_NM.isin([
        'Airway Intubation Procedure', 'Difficult Airway / Difficult Intubation', 
        'Endotracheal Intubation Procedure', 'Endotracheal Intubation Procedure *ED', 
        'Intubation Procedure', 'Otolaryngology Consult - Airway', 
        'Otolaryngology Consult - Tracheostomy', 'Tracheotomy Procedure'])]
    notes_selected['date_hashed'] = notes_selected["DAY_SERVICE_DESC2"].apply(shift_date_unix)
    
    return notes_selected
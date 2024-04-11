# -*- coding: utf-8 -*-
"""
@author: Elite Data Hacks
"""
import sys
import pandas as pd
import numpy as np
import sepyDICT as sd
import pickle
import time
from pathlib import Path
 
############################## File Paths ##############################
#### List of patient encounters (i.e. CSNs); now handled in BASH
# csn_list_file_name = '/home/cjosef/EliteDataHacks/pt_lists/ian_csn_year.psv'
# CSJ-PC csn_list_file_name = Path("C:/Users/DataSci/OneDrive - Emory University/CJ_Sepsis/1.pt_list/em_get_csn_list/em_pt_list_14_20.psv")

#### Location of pickles with yearly data; now handled in BASH
# pickle_path = '/home/cjosef/EliteDataHacks/pickles/'
# CSJ-PC pickle_path = Path('C:/Users/DataSci/OneDrive - Emory University/CJ_Sepsis/5.quality_check/')

#### Where the encounter dictionaries should be written; now handled in BASH
# output_path = '/home/cjosef/EliteDataHacks/IanFiles/'
# CSJ-PC output_path = Path('C:/Users/DataSci/OneDrive - Emory University/CJ_Sepsis/5.quality_check/')

def process_csn(csn, pickle_write_path):
    
    file_name = pickle_write_path / (str(csn) + '.pickle')
    
    #instantiate class for single encounter
    instance = sd.sepyDICT(yearly_instance, csn)
    #create encounter dictionary
    inst_dict = instance.encounter_dict
        
    #create a pickle file for encounter
    picklefile = open(file_name, 'wb')
    #pickle the encounter dictionary and write it to file
    pickle.dump(inst_dict, picklefile)
    #close the file
    picklefile.close()
    
    #return dictionary for summary report functions
    return(instance)

def sofa_summary(csn, instance):
    sofa_scores = instance.encounter_dict['sofa_scores'].reset_index().rename(columns = {'index':'time_stamp'})
    sofa_scores['csn']=csn #add csn to sofa_scores
    appended_sofa_scores.append(sofa_scores)

def sepsis3_summary(csn, instance):
    sep3_time = instance.encounter_dict['sep3_time']
    sep3_time['csn']=csn #add csn to sep3 time
    appended_sep3_time.append(sep3_time)   

def sirs_summary(csn, instance):
    sirs_scores = instance.encounter_dict['sirs_scores'].reset_index().rename(columns = {'index':'time_stamp'})
    sirs_scores['csn']=csn #add csn to sirs_scores
    appended_sirs_scores.append(sirs_scores)

def sepsis2_summary(csn, instance):
    sep2_time = instance.encounter_dict['sep2_time']
    sep2_time['csn']=csn #add csn to sep3 time
    appended_sep2_time.append(sep2_time) 

def enc_summary(csn, instance):
    enc_summary_dict = {**instance.flags, **instance.static_features, **instance.event_times}
    enc_summary_df = pd.DataFrame(enc_summary_dict, index=[0]).set_index(['csn'])
    appended_enc_summaries.append(enc_summary_df)

def comorbidity_summary(csn, instance):
    #ahrq_ICD9_dict[csn]=instance.ahrq_ICD9_PerCSN.icd_count
    #elix_ICD9_dict[csn]=instance.elix_ICD9_PerCSN.icd_count
    #quan_deyo_ICD9_dict[csn]=instance.quan_deyo_ICD9_PerCSN.icd_count
    #quan_elix_ICD9_dict[csn]=instance.quan_elix_ICD9_PerCSN.icd_count
    #ccs_ICD9_dict[csn]=instance.ccs_ICD9_PerCSN.icd_count

    #ICD10
    #ahrq_ICD10_dict[csn]=instance.ahrq_ICD10_PerCSN.icd_count
    #elix_ICD10_dict[csn]=instance.elix_ICD10_PerCSN.icd_count
    quan_deyo_ICD10_dict[csn]=instance.quan_deyo_ICD10_PerCSN.icd_count
    quan_elix_ICD10_dict[csn]=instance.quan_elix_ICD10_PerCSN.icd_count
    #ccs_ICD10_dict[csn]=instance.ccs_ICD10_PerCSN.icd_count
    #print(f'The csn for comorbids is: {csn}')
    
### initialize empty vars
#ICD9
#ahrq_ICD9_dict={}
#elix_ICD9_dict={}
#quan_deyo_ICD9_dict={}
#quan_elix_ICD9_dict={}
#ccs_ICD9_dict={}

#ICD10
#ahrq_ICD10_dict={}
#elix_ICD10_dict={}
quan_deyo_ICD10_dict={}
quan_elix_ICD10_dict={}
#ccs_ICD10_dict={}

# other summaries
appended_sofa_scores = []
appended_sep3_time = []
appended_sirs_scores = []
appended_sep2_time = []
appended_enc_summaries = []

start=time.perf_counter()

if __name__ == '__main__':
    #accepts the arguments from the command line
    try:
        print(sys.version)

        csn_list_file_name = Path(sys.argv[1]) #imports file location for CSN list
        print(f'MkDct- The CSN List location: {csn_list_file_name}')
        
        pickle_path = Path(sys.argv[2]) #sets directoy for yearly pickles
        print(f'MkDct- The pickle directory: {pickle_path}')
       
        output_path =  Path(sys.argv[3]) #sets directory for output
        print(f'MkDct- The output directory: {output_path}')
        
        num_processes = int(sys.argv[4]) # total number of tasks/processes
        print(f'MkDct- The tot num of processes: {num_processes}')

        processor_assignment = int(sys.argv[5]) # task number
        print(f'MkDct- This task is: {processor_assignment}')

        bash_year = int(sys.argv[6]) # year num provided in bash
        print(f'MkDct- The import year is: {bash_year}')
      
    except:
        print ('MkDct- There was an error importing one of the arguments.')
        print (f'MkDct- You are trying to load the following CSN list {sys.argv[1]}')
        print (f'MkDct- You are trying to use this many processors {sys.argv[2]}')  

    #####################################################################
    ########### Creat Encounter List Based on Processor Assignment
    #####################################################################
    # Cohort selector
    ed = 0 
    in_pt = 1 
    icu = 0 
    adult = 1
    vent_row = 0 
    vent_start = 0
    
    # reads the list of csns
    csn_df = pd.read_csv(csn_list_file_name, sep='|', header=0).iloc[1:200]

    #  only keep csns that meet specified year
    csn_df = csn_df[(csn_df.in_pt == in_pt) & (csn_df.adult == adult)]
    print(f'MkDct- A total of {csn_df.shape[0]} encounters were selected')
    csn_df = csn_df[['csn','year']]
    
    # drop duplicates
    csn_df = csn_df.drop_duplicates()
    csn_df = csn_df[csn_df.year == bash_year]
    total_num_enc = len(csn_df)
    print(f'MkDct- The year {bash_year} has {total_num_enc} encounters.')
    
    # breaks encounter list into chunks, selects correct chunk based on process num
    chunk_size = int(total_num_enc/num_processes)
    print(f'MkDct- The ~chunk size is {chunk_size}')

    #old way to split list
    #list_of_chunks = [csn_df.iloc[i:i + chunk_size,:] for i in range(0, total_num_enc, chunk_size)]
    #new way to split list
    list_of_chunks = np.array_split(csn_df, num_processes)
    print(f'MkDct- The list of chunks has {len(list_of_chunks)} unique dataframes.')
    
    #uses processor assignment to select correct chunk
    process_list = list_of_chunks[processor_assignment]['csn']
    print(f'MkDct- The process_list head:\n {process_list.head()}')
    
    #select correct pickle by year
    pickle_name = pickle_path / ('em_y' + str(bash_year) + '.pickle')
    print(f'MkDct- The following pickle is being read: {pickle_name}')
        
    try:
        # reads the IMPORT class instance (i.e.  1 year of patient data)
        pickle_load_time = time.perf_counter() #time to load pickle
        with open(pickle_name, 'rb') as handle:
            yearly_instance = pickle.load(handle)
            print(f'MkDct-Pickle from year {bash_year} was loaded in {time.perf_counter()-pickle_load_time}s.')

        print("-----------LOADED YEARLY PICKLE FILE!!!!---------------")
 
        # if success, make a dir for this year's encounters
        pickle_write_path = output_path / str(bash_year)
        Path.mkdir(pickle_write_path, exist_ok = True)
        print(f'MkDct-Directory for year {bash_year} was set to {pickle_write_path}')
        
        # make empty list to handle csn's with errors
        error_list = []
        start_csn_creation = time.perf_counter() #times calc's by year
            
        #################################################
        ############ Make Dicts by CSN
        #################################################
            
        count = 0
        for csn in process_list:
            count +=1 
            try:
                print(f'MkDct- The current pt csn is: {csn}, which is {count} of {chunk_size} for year {bash_year}')
                instance = process_csn(csn, pickle_write_path)
                print("MkDct- Instance created")
                try:
                    sofa_summary(csn, instance)
                except Exception as e: 
                    print('MkDct- Sofa Summary error')
                    print(e)
                try:
                    sepsis3_summary(csn, instance)
                except Exception as e: 
                    print('MkDct- Sepsis 3 error')
                    print(e)
                try:
                    sirs_summary(csn, instance)
                except Exception as e: 
                    print('MkDct- SIRS Summary error')
                    print(e)
                try:
                    sepsis2_summary(csn, instance)
                except Exception as e: 
                    print('MkDct- Sep2 Summary error')
                    print(e)
                try:
                    enc_summary(csn, instance)
                except Exception as e: 
                    print('MkDct- Encounter Summary error')
                    print(e)
                try:
                    comorbidity_summary(csn, instance)
                except Exception as e: 
                    print('MkDct- Comorbidity Summary error')
                    print(e)
                    
                print(f'MkDct- Enounter {count} of {chunk_size} is complete!')
                
            except Exception as e: 
                print(e)
                error_list.append([csn, e.args[0]])
                print(f'MkDct- The following csn had an error: {csn}')
        
        #################################################
        ############ Export Sepsis Summary
        #################################################
        
        # create sepsis_summary directory
        Path.mkdir(output_path / 'em_sepsis_summary', exist_ok = True)
        Path.mkdir(output_path / 'em_sepsis_summary' / 'sofa_summary', exist_ok = True)
        Path.mkdir(output_path / 'em_sepsis_summary' / 'sep3_summary', exist_ok = True)
        Path.mkdir(output_path / 'em_sepsis_summary' / 'sirs_summary', exist_ok = True)
        Path.mkdir(output_path / 'em_sepsis_summary' / 'sep2_summary', exist_ok = True)       
        Path.mkdir(output_path / 'em_sepsis_summary' / 'encounter_summary', exist_ok = True)       
        Path.mkdir(output_path / 'em_sepsis_summary' / 'error_summary', exist_ok = True)
        
        #ICD9 co-morbid        
        #Path.mkdir(output_path / 'em_sepsis_summary' / 'ahrq_ICD9_summary', exist_ok = True)
        #Path.mkdir(output_path / 'em_sepsis_summary' / 'elix_ICD9_summary', exist_ok = True)
        #Path.mkdir(output_path / 'em_sepsis_summary' / 'quan_deyo_ICD9_summary', exist_ok = True)
        #Path.mkdir(output_path / 'em_sepsis_summary' / 'quan_elix_ICD9_summary', exist_ok = True)
        #Path.mkdir(output_path / 'em_sepsis_summary' / 'ccs_ICD9_summary', exist_ok = True)
        
        #ICD10 co-morbid        
        #Path.mkdir(output_path / 'em_sepsis_summary' / 'ahrq_ICD10_summary', exist_ok = True)
        #Path.mkdir(output_path / 'em_sepsis_summary' / 'elix_ICD10_summary', exist_ok = True)
        Path.mkdir(output_path / 'em_sepsis_summary' / 'quan_deyo_ICD10_summary', exist_ok = True)
        Path.mkdir(output_path / 'em_sepsis_summary' / 'quan_elix_ICD10_summary', exist_ok = True)
        #Path.mkdir(output_path / 'em_sepsis_summary' / 'ccs_ICD10_summary', exist_ok = True)              
        
        # write general files
        unique_file_id = "%s_%s"%(processor_assignment,bash_year)
        pd.concat(appended_enc_summaries).to_csv(output_path   / 'em_sepsis_summary' / 'encounter_summary' /  ('encounters_summary_'+ unique_file_id +'.csv')  , index = True)      
        pd.DataFrame(error_list, columns = ['csn','error']).to_csv(output_path / 'em_sepsis_summary' / 'error_summary' / ('error_list_'+ unique_file_id +'.csv')  , index = False)

        # write sepsis files 
        pd.concat(appended_sofa_scores).to_csv(output_path / 'em_sepsis_summary' / 'sofa_summary' / ('sofa_summary_'+ unique_file_id +'.csv'), index = False)
        pd.concat(appended_sep3_time).to_csv(output_path / 'em_sepsis_summary' / 'sep3_summary' / ('sepsis3_summary_'+ unique_file_id +'.csv'), index = False)
        pd.concat(appended_sirs_scores).to_csv(output_path   / 'em_sepsis_summary' / 'sirs_summary' /  ('sirs_summary_'+ unique_file_id +'.csv'), index = False)
        pd.concat(appended_sep2_time).to_csv(output_path / 'em_sepsis_summary' / 'sep2_summary' / ('sepsis2_summary_'+ unique_file_id +'.csv'), index = False)
 
        #write comorbidity files
        #ICD9
        #pd.DataFrame.from_dict(ahrq_ICD9_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'ahrq_ICD9_summary' / ('ahrq_ICD9_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        #pd.DataFrame.from_dict(elix_ICD9_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'elix_ICD9_summary' / ('elix_ICD9_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        #pd.DataFrame.from_dict(quan_deyo_ICD9_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'quan_deyo_ICD9_summary' / ('quan_deyo_ICD9_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        #pd.DataFrame.from_dict(quan_elix_ICD9_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'quan_elix_ICD9_summary' / ('quan_elix_ICD9_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        #pd.DataFrame.from_dict(ccs_ICD9_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'ccs_ICD9_summary' / ('ccs_ICD9_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        #ICD10
        #pd.DataFrame.from_dict(ahrq_ICD10_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'ahrq_ICD10_summary' / ('ahrq_ICD10_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        #pd.DataFrame.from_dict(elix_ICD10_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'elix_ICD10_summary' / ('elix_ICD10_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        pd.DataFrame.from_dict(quan_deyo_ICD10_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'quan_deyo_ICD10_summary' / ('quan_deyo_ICD10_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        pd.DataFrame.from_dict(quan_elix_ICD10_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'quan_elix_ICD10_summary' / ('quan_elix_ICD10_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        #pd.DataFrame.from_dict(ccs_ICD10_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'ccs_ICD10_summary' / ('ccs_ICD10_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        #################################################
        
        print(f'MkDct- Time to create write encounter pickles for {bash_year} was {time.perf_counter()-start_csn_creation} (s)')

            
    except Exception as e: 
        print(e)
        print(f'MkDct- Could not find or open the pickle for year {bash_year}.')

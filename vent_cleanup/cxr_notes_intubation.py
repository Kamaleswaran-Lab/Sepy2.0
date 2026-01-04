from ast import arg
import os
import pandas as pd
from pathlib import Path
import regex as re
import time
from datetime import datetime
import numpy as np

from tqdm import tqdm
from openai import AzureOpenAI

AZURE_OPENAI_API_KEY = "3aab0a9f85bd477988b1a5123cd64f60"
AZURE_OPENAI_API_VERSION = "2024-05-01-preview"
AZURE_OPENAI_API_ENDPOINT = "https://dhd-akn30-oa05.openai.azure.com/"
AZURE_OPENAI_CLIENT = AzureOpenAI(api_version=AZURE_OPENAI_API_VERSION,
                                    azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
                                    api_key=AZURE_OPENAI_API_KEY,)
deployment = "gpt-4"

# Configuration
BATCH_SIZE = 10  # Number of reports to process in a single API call
MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 2000  # Save checkpoint every N processed reports


BATCH_PROMPT_TEMPLATE = """You are a medical AI assistant specialized in analyzing radiology reports. Your task is to extract what the report indicates as the status of an endotracheal tube.
That is, you are to check whether an endotracheal tube was 1. NEW 2. PRESENT 3. REMOVED 4. NOT MENTIONED

Rules:
1. If the report does not indicate whether the endotracheal tube was newly inserted or not, then just return "PRESENT" as your response.
2. If the report might refer to endotracheal tube as 
    "et tube", "ett", "tube in the trachea", "tracheostomy", "trach collar". 
   ALL these should be considered as referring to "endotracheal tubes". Please also return the exact sentence used in the report. 
3. Your response should be in this format:
    Endotracheal tube status: <NEW/PRESENT/REMOVED/NOT MENTIONED>
    Indicative phrase in the report: <>
4. Sometimes the mention can be in the "clinical indication" section of the report, referring to why the exam was done.

Examples:
- Report: REPORT EXAM: XR Chest 1 View Portable ESRC.4.6.1 CLINICAL INDICATION: Liver Transplant;Other COMPARISON: Multiple, most recently {{REDACTED-date}}. FINDINGS: Support Devices: The left IJ approach central venous catheter has been adjusted and now terminates in the superior cavoatrial junction. The right IJ approach central venous catheter has been adjusted and now terminates over the superior cavoatrial junction. Enteric tube is unchanged in position. The endotracheal tube terminates within the thoracic trachea. Lungs/pleura: No pneumothorax. The lungs are hypoventilated with mild interstitial crowding and linear atelectasis in the left lower lung. Trace right pleural effusion. Heart/mediastinum: Unchanged. Other: None IMPRESSION: 1.  Support apparatus as above. 2.  Minimal subsegmental atelectasis and trace right pleural effusion
  Endotracheal tube status: PRESENT
  Indicative phrase in the report: The endotracheal tube terminates within the thoracic trachea
  
- Report: 'REPORT EXAM: XR Chest 1 View Portable CLINICAL INDICATION: Abnormal finding on lung imaging;Abnormal finding on lung imaging. ESRC.4.6.1 COMPARISON: {{REDACTED-date}}. FINDINGS: Support Devices: ETT has been removed. Remainder of the support lines and tubes remain in stable position. Lungs/Pleura: No pneumothorax. Small bilateral dependent pleural effusions with associated atelectasis. Heart/Mediastinum: Widened vascular pedicle, likely postsurgical in nature. Other: None IMPRESSION: ET tube removed. No pneumothorax. Otherwise, stable exam.'
  Endotracheal tube status: REMOVED
  Indicative phrase in the report: ETT has been removed.

- Report: 'REPORT EXAM: XR Chest 1 View Portable CLINICAL INDICATION: Respiratory failure;Respiratory failure. ESRC.4.6.1 COMPARISON: Same day earlier. FINDINGS: Support Devices: Stable left IJ central venous line with tip deep within the right atrium. New ET tube with tip at the thoracic inlet. Nasogastric tube coursing inferiorly beyond the provided field-of-view at least within the stomach. Lungs/Pleura: Bilaterally reduced lung volumes with central bronchovascular crowding. Stable bilateral lung appearances with right dependent pleural effusion with underlying atelectasis/airspace opacities. Left basilar airspace opacities, stable. Heart/Mediastinum: Unchanged. Other: Left axillary vascular stent. IMPRESSION: Stable left IJ central venous line with tip deep within the right atrium. New ET tube with tip at the thoracic inlet. Nasogastric tube coursing inferiorly beyond the provided field-of-view at least within the stomach. No pneumothorax.'
  Endotracheal tube status: NEW
  Indicative phrase in the report: New ET tube with tip at the thoracic inlet.

Now, process the following {num_reports} reports. For each report, provide the response in the exact format above.

{reports_text}
"""


def create_batch_message(reports_list):
    """
    Create a prompt with multiple reports batched together
    
    Args:
        reports_list: List of tuples (idx, report_text)
    
    Returns:
        The message to send to the AI assistant
    """
    reports_text = ""
    for i, (idx, report) in enumerate(reports_list, 1):
        reports_text += f"\n--- REPORT {i} (ID: {idx}) ---\n{report}\n"
    
    prompt = BATCH_PROMPT_TEMPLATE.format(
        num_reports=len(reports_list),
        reports_text=reports_text
    )
    message_text = [{"role": "user", "content": prompt}]
    
    return message_text


def parse_batch_response(response_text, batch_reports):
    """
    Parse the response containing multiple report analyses
    
    Args:
        response_text: The response text from the AI assistant
        batch_reports: List of tuples (idx, report_text) to match IDs
    
    Returns:
        List of tuples (idx, parsed_dict) matched by ID
    """
    results = []
    
    # Split by report delimiter pattern
    sections = re.split(r'--- REPORT \d+ \(ID: .+?\) ---', response_text)
    # Also extract the IDs from the delimiters
    id_matches = re.findall(r'--- REPORT \d+ \(ID: (.+?)\) ---', response_text)
    
    # Remove empty first section (before first delimiter)
    sections = [s.strip() for s in sections if s.strip()]
    
    # Create a dict to store results by ID
    results_by_id = {}
    
    for section, report_id_str in zip(sections, id_matches):
        try:
            # Parse the status and phrase
            presence = section.split("Endotracheal tube status: ")[1].split("\n")[0].strip()
            indicative_phrase = section.split("Indicative phrase in the report: ")[1].split("\n")[0].strip()
            
            # Convert ID string to match the batch_reports index type
            results_by_id[str(report_id_str)] = {
                "presence": presence, 
                "indicative_phrase": indicative_phrase
            }
        except Exception as e:
            print(f"Error parsing section for ID {report_id_str}: {e}")
            results_by_id[str(report_id_str)] = {
                "presence": np.nan, 
                "indicative_phrase": np.nan
            }
    
    # Match results back to original batch order using IDs
    for idx, _ in batch_reports:
        if str(idx) in results_by_id:
            results.append((idx, results_by_id[str(idx)]))
        else:
            # ID not found in response
            print(f"Warning: No result found for report ID {idx}")
            results.append((idx, {"presence": np.nan, "indicative_phrase": np.nan}))
    
    return results


def process_batch(batch_reports):
    """
    Process a batch of reports with retry logic
    
    Args:
        batch_reports: List of tuples (idx, report_text)
        
    Returns:
        List of tuples (idx, result_dict)
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = AZURE_OPENAI_CLIENT.chat.completions.create(
                model=deployment,
                messages=create_batch_message(batch_reports),
                temperature=0.7,
                max_tokens=4000,  # Increased for batch processing
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            response_text = response.choices[0].message.content
            results = parse_batch_response(response_text, batch_reports)
            
            return results
            
        except Exception as e:
            print(f"\nError processing batch (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    # All retries failed - return NaN for all reports in batch
    print(f"\nFailed to process batch after {MAX_RETRIES} attempts. Setting all {len(batch_reports)} reports to NaN.")
    return [(idx, {"presence": np.nan, "indicative_phrase": np.nan}) for idx, _ in batch_reports]


def main():
    """Main function to process all reports"""
    notes = pd.read_pickle("/data/irb/surgery/pro00114885/EmoryDataset/radiology_notes_cxr_on_vent.pickle")
    
    # Initialize columns
    if 'intubation_status' not in notes.columns:
        notes['intubation_status'] = np.nan
    if 'intubation_indicative_phrase' not in notes.columns:
        notes['intubation_indicative_phrase'] = np.nan
    
    print(f"Loaded {len(notes)} reports to process")
    print(f"Batch size: {BATCH_SIZE} reports per API call")
    print(f"Expected number of API calls: {(len(notes) + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    # Create batches
    batches = []
    for i in tqdm(range(0, len(notes), BATCH_SIZE), desc="Creating batches"):
        batch = []
        for idx in range(i, min(i + BATCH_SIZE, len(notes))):
            report = notes.iloc[idx]['notes_deid']
            batch.append((notes.index[idx], report))
        batches.append(batch)
    
    # Process batches
    start_time = time.time()
    all_results = []
    
    for batch_num, batch in enumerate(tqdm(batches, desc="Processing batches")):
        results = process_batch(batch)
        all_results.extend(results)
        
        # Save checkpoint periodically
        if (batch_num + 1) * BATCH_SIZE % CHECKPOINT_INTERVAL == 0 or batch_num == len(batches) - 1:
            # Update dataframe with results so far
            for idx, parsed in all_results:
                notes.loc[idx, 'intubation_status'] = parsed['presence']
                notes.loc[idx, 'intubation_indicative_phrase'] = parsed['indicative_phrase']
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"/data/irb/surgery/pro00114885/EmoryDataset/notes_checkpoints/checkpoint_{timestamp}_n{len(all_results)}.pickle"
            notes.to_pickle(checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")
    
    # Update dataframe with all results
    for idx, parsed in all_results:
        notes.loc[idx, 'intubation_status'] = parsed['presence']
        notes.loc[idx, 'intubation_indicative_phrase'] = parsed['indicative_phrase']
    
    elapsed_time = time.time() - start_time
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/data/irb/surgery/pro00114885/EmoryDataset/notes_checkpoints/radiology_notes_cxr_intubation_processed_{timestamp}.pickle"
    notes.to_pickle(output_path)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total API calls made: {len(batches)}")
    print(f"Average time per API call: {elapsed_time/len(batches):.2f} seconds")
    print(f"Average time per report: {elapsed_time/len(notes):.2f} seconds")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total reports: {len(notes)}")
    print(f"Failed reports (NaN): {notes['intubation_status'].isna().sum()}")
    print("\nIntubation status distribution:")
    print(notes['intubation_status'].value_counts(dropna=False))


if __name__ == "__main__":
    main()
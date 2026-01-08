from ast import arg
import os
import pandas as pd
from pathlib import Path
import regex as re
import time
import asyncio
from datetime import datetime
import numpy as np
import glob

from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from openai import AsyncAzureOpenAI, RateLimitError

AZURE_OPENAI_API_KEY = "3aab0a9f85bd477988b1a5123cd64f60"
AZURE_OPENAI_API_VERSION = "2024-05-01-preview"
AZURE_OPENAI_API_ENDPOINT = "https://dhd-akn30-oa05.openai.azure.com/"
AZURE_OPENAI_CLIENT = AsyncAzureOpenAI(api_version=AZURE_OPENAI_API_VERSION,
                                        azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
                                        api_key=AZURE_OPENAI_API_KEY,)
deployment = "gpt-4"

# Configuration
BATCH_SIZE = 10  # Number of reports to process in a single API call
MAX_CONCURRENT_BATCHES = 3  # Number of batches to process concurrently
MAX_RETRIES = 3
CHECKPOINT_EVERY_N_BATCHES = 50  # Save checkpoint after every N batches


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


async def process_batch(batch_reports, semaphore):
    """
    Process a batch of reports with retry logic (async version)
    
    Args:
        batch_reports: List of tuples (idx, report_text)
        semaphore: Asyncio semaphore to limit concurrent requests
        
    Returns:
        List of tuples (idx, result_dict)
    """
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await AZURE_OPENAI_CLIENT.chat.completions.create(
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
                
            except RateLimitError as e:
                print(f"\nRate limit error on batch (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                print("Sleeping for 60 seconds...")
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"\nError processing batch (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed - return NaN for all reports in batch
        print(f"\nFailed to process batch after {MAX_RETRIES} attempts. Setting all {len(batch_reports)} reports to NaN.")
        return [(idx, {"presence": np.nan, "indicative_phrase": np.nan}) for idx, _ in batch_reports]


def find_latest_checkpoint():
    """
    Find the most recent checkpoint or final output file
    
    Returns:
        Path to the latest checkpoint file, or None if no checkpoint exists
    """
    checkpoint_dir = "/data/irb/surgery/pro00114885/EmoryDataset/notes_checkpoints"
    
    # Look for both checkpoint and final output files
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_async_*.pickle")
    output_pattern = os.path.join(checkpoint_dir, "radiology_notes_cxr_intubation_processed_async_*.pickle")
    
    checkpoint_files = glob.glob(checkpoint_pattern)
    output_files = glob.glob(output_pattern)
    
    all_files = checkpoint_files + output_files
    
    if not all_files:
        return None
    
    # Get the most recent file by modification time
    latest_file = max(all_files, key=os.path.getmtime)
    return latest_file


async def main():
    """Main async function to process all reports"""
    
    # Check for existing checkpoint
    checkpoint_file = find_latest_checkpoint()
    
    if checkpoint_file:
        print(f"Found existing checkpoint: {checkpoint_file}")
        print("Loading checkpoint to resume processing...")
        notes = pd.read_pickle(checkpoint_file)
        
        # Count how many are already processed
        if 'intubation_status' in notes.columns:
            processed_count = notes['intubation_status'].notna().sum()
            print(f"Already processed: {processed_count}/{len(notes)} reports")
        else:
            processed_count = 0
    else:
        print("No checkpoint found. Starting from scratch...")
        notes = pd.read_pickle("/data/irb/surgery/pro00114885/EmoryDataset/radiology_notes_cxr_on_vent.pickle")
        processed_count = 0
    
    # Initialize columns
    if 'intubation_status' not in notes.columns:
        notes['intubation_status'] = np.nan
    if 'intubation_indicative_phrase' not in notes.columns:
        notes['intubation_indicative_phrase'] = np.nan
    
    # Identify reports that need processing (intubation_status is NaN)
    unprocessed_mask = notes['intubation_status'].isna()
    unprocessed_indices = notes[unprocessed_mask].index.tolist()
    
    print(f"\nTotal reports: {len(notes)}")
    print(f"Already processed: {len(notes) - len(unprocessed_indices)}")
    print(f"Remaining to process: {len(unprocessed_indices)}")
    print(f"Batch size: {BATCH_SIZE} reports per API call")
    print(f"Max concurrent batches: {MAX_CONCURRENT_BATCHES}")
    print(f"Expected number of API calls: {(len(unprocessed_indices) + BATCH_SIZE - 1) // BATCH_SIZE}")
    if len(unprocessed_indices) == 0:
        print("\nAll reports already processed! Nothing to do.")
        return
    
    # Create batches only for unprocessed reports
    batches = []
    batch = []
    for idx in tqdm(unprocessed_indices, desc="Creating batches"):
        report = notes.loc[idx, 'notes_deid']
        batch.append((idx, report))
        
        if len(batch) == BATCH_SIZE:
            batches.append(batch)
            batch = []
    
    # Add remaining reports as final batch
    if batch:
        batches.append(batch)
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
    
    # Process batches concurrently
    start_time = time.time()
    print(f"\nProcessing {len(batches)} batches concurrently...")
    
    # Create all tasks
    tasks = [process_batch(batch, semaphore) for batch in batches]
    
    # Process with progress bar
    batches_completed = 0
    
    for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Processing batches"):
        results = await coro
        batches_completed += 1
        
        # Update dataframe with results immediately
        for idx, parsed in results:
            notes.loc[idx, 'intubation_status'] = parsed['presence']
            notes.loc[idx, 'intubation_indicative_phrase'] = parsed['indicative_phrase']
        
        # Save checkpoint every N batches
        if batches_completed % CHECKPOINT_EVERY_N_BATCHES == 0 or batches_completed == len(batches):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"/data/irb/surgery/pro00114885/EmoryDataset/notes_checkpoints/checkpoint_async_{timestamp}.pickle"
            notes.to_pickle(checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")
    
    elapsed_time = time.time() - start_time
    
    # Final stats
    total_processed_now = notes['intubation_status'].notna().sum()
    remaining = notes['intubation_status'].isna().sum()
    
    print(f"\n{'='*60}")
    print(f"Session complete!")
    print(f"Session time: {elapsed_time:.2f} seconds")
    print(f"Batches processed: {len(batches)}")
    print(f"Reports attempted: {len(unprocessed_indices)}")
    print(f"Average time per batch: {elapsed_time/len(batches):.2f} seconds")
    print(f"{'='*60}")
    
    print(f"\nProgress:")
    print(f"Total reports: {len(notes)}")
    print(f"Processed (non-NaN): {total_processed_now}")
    print(f"Remaining (NaN): {remaining}")
    print(f"Completion: {100*total_processed_now/len(notes):.1f}%")
    
    print(f"\nIntubation status distribution:")
    print(notes['intubation_status'].value_counts(dropna=True))
    
    if remaining > 0:
        print(f"\n⚠️  {remaining} reports remain unprocessed.")
        print("   Run this script again to resume from checkpoint.")


if __name__ == "__main__":
    asyncio.run(main())


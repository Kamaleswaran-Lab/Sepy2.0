#!/usr/bin/env python3
import os
import shutil
import hashlib
from pathlib import Path
from argparse import ArgumentParser

# Hash key for deidentification
hash_key = '123'

def hash_value(value, hash_key):
    """Hash a value using SHA256 with the provided hash key"""
    return hashlib.sha256((str(value) + hash_key).encode()).hexdigest()

def copy_and_deidentify_files(source_dir, dest_dir):
    """
    Copy files from supertables/<year>/<Supertables/ClinicalData>/<csn>.<file extension>
    to deid_supertables/<year>/<Supertables/ClinicalData>/<deid_csn>.<file extension>
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist")
        return
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    files_copied = 0
    
    # Walk through the source directory structure
    for root, dirs, files in os.walk(source_path):
        root_path = Path(root)
        
        # Calculate relative path from source
        rel_path = root_path.relative_to(source_path)
        
        # Create corresponding directory in destination
        dest_dir_path = dest_path / rel_path
        dest_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for file in files:
            source_file = root_path / file
            
            # Extract CSN and file extension
            file_parts = file.split('.')
            if len(file_parts) >= 2:
                csn = file_parts[0]
                file_extension = '.'.join(file_parts[1:])
                
                # Generate deidentified CSN
                deid_csn = hash_value(csn, hash_key)
                
                # Create new filename with deidentified CSN
                new_filename = f"{deid_csn}.{file_extension}"
                dest_file = dest_dir_path / new_filename
                
                # Copy the file
                try:
                    shutil.copy2(source_file, dest_file)
                    files_copied += 1
                    print(f"Copied: {source_file} -> {dest_file}")
                except Exception as e:
                    print(f"Error copying {source_file}: {e}")
            else:
                # If file doesn't have expected format, copy as-is
                dest_file = dest_dir_path / file
                try:
                    shutil.copy2(source_file, dest_file)
                    files_copied += 1
                    print(f"Copied (no CSN): {source_file} -> {dest_file}")
                except Exception as e:
                    print(f"Error copying {source_file}: {e}")
    
    print(f"\nTotal files copied: {files_copied}")

def main():
    parser = ArgumentParser(description="Deidentify CSNs in file names and copy files")
    parser.add_argument('--source', '-s', required=True, 
                       help='Source directory (e.g., supertables)')
    parser.add_argument('--dest', '-d', required=True,
                       help='Destination directory (e.g., deid_supertables)')
    parser.add_argument('--year', '-y', type=str,
                       help='Specific year to process (optional)')
    
    args = parser.parse_args()
    
    if args.year:
        # Process specific year
        source_year_path = Path(args.source) / args.year
        dest_year_path = Path(args.dest) / args.year
        
        if source_year_path.exists():
            print(f"Processing year {args.year}")
            copy_and_deidentify_files(source_year_path, dest_year_path)
        else:
            print(f"Year directory {source_year_path} does not exist")
    else:
        # Process all years found in source directory
        source_path = Path(args.source)
        if source_path.exists():
            for year_dir in source_path.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    print(f"\nProcessing year {year_dir.name}")
                    dest_year_path = Path(args.dest) / year_dir.name
                    copy_and_deidentify_files(year_dir, dest_year_path)
        else:
            print(f"Source directory {source_path} does not exist")

if __name__ == "__main__":
    main()

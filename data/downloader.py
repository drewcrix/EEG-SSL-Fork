"""
downloader.py

This script uses datalad to download a list of datasets for use with the EEG-SSL project.

For each dataset, first get the metadata, then find and download the EEG files.

Author: Leopold Ehrlich
"""

import datalad.api as dl
import os
import glob

# List of openneuro dataset ids to download
dataset_ids = [
    "ds004504",
    "ds003775"
    "ds007358"
    ]

base_path = os.path.abspath("./on")
os.makedirs(base_path, exist_ok=True)

for ds_id in dataset_ids:
    source_url = f"https://github.com/OpenNeuroDatasets/{ds_id}.git"
    dataset_path = os.path.join(base_path, ds_id)
    
    print(f"--- Processing {ds_id} ---")
    
    # clone metadata first
    print(f"Cloning {ds_id} metadata...")
    ds = dl.clone(source=source_url, path=dataset_path)
    

    # For datalad, hab
    print(f"Fetching EEG data for {ds_id}...")
    search_pattern = os.path.join(dataset_path, "**/eeg")
    found_eeg_dirs = glob.glob(search_pattern, recursive=True)
    
    try:
        ds.get(path=found_eeg_dirs)
        print(f"Successfully pulled EEG data for {ds_id}")
    except Exception as e:
        print(f"Could not find EEG folders in the expected BIDS format for {ds_id}: {e}")

    print("-" * 30)
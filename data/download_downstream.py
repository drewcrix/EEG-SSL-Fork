import argparse
from moabb.datasets import (
    PhysionetMI, 
    BNCI2014002, 
    BNCI2014009, 
    BNCI2014008, 
    BNCI2015004
)

def main():
    # 1. Map strings to the actual classes
    dataset_map = {
        "PhysionetMI": PhysionetMI,
        "BNCI2014002": BNCI2014002,
        "BNCI2014009": BNCI2014009,
        "BNCI2014008": BNCI2014008,
        "BNCI2015004": BNCI2015004,
    }

    # 2. Setup Argparse
    parser = argparse.ArgumentParser(description="Convert MOABB datasets to BIDS format.")
    
    
    parser.add_argument(
        "--path", 
        type=str, 
        default=".data/downstream/", 
        help="The destination path for BIDS conversion. Defaults to ~/mne_data."
    )
    
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        choices=list(dataset_map.keys()),
        default=list(dataset_map.keys()),
        help="List of datasets to convert. Space-separated. Default is all."
    )

    args = parser.parse_args()

    # 3. Execution Logic
    for ds_name in args.datasets:
        print(f"--- Converting {ds_name} ---")
        dataset_class = dataset_map[ds_name]
        
        # Instantiate and convert
        ds_instance = dataset_class()
        _ = ds_instance.get_data(cache_config=dict(path=args.path, save_raw=True))
        
    print("\nConversion process complete.")

if __name__ == "__main__":
    main()
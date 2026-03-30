import pandas as pd

def merge_datasets(unsw_path, cicids_path):
    """Merge cleaned UNSW-NB15 and CICIDS2017 into one dataset."""
    
    print("Loading cleaned datasets...")
    unsw   = pd.read_csv(unsw_path)
    cicids = pd.read_csv(cicids_path)
    
    print(f"UNSW shape:   {unsw.shape}")
    print(f"CICIDS shape: {cicids.shape}")
    
    # Print columns to see the mismatch
    print(f"\nUNSW columns:   {list(unsw.columns)}")
    print(f"CICIDS columns: {list(cicids.columns)}")
    
    # Add source column
    unsw["source"]   = 0
    cicids["source"] = 1
    
    # Align CICIDS columns to match UNSW column order
    cicids = cicids[unsw.columns]
    
    # Merge
    merged = pd.concat([unsw, cicids], ignore_index=True)
    print(f"\nMerged shape: {merged.shape}")
    
    # Shuffle rows
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Final label distribution
    print(f"\nFinal label distribution:")
    print(merged["Label"].value_counts())
    print(f"\nSource distribution:")
    print(merged["source"].value_counts())
    
    # Check missing values
    missing = merged.isnull().sum()[merged.isnull().sum() > 0]
    if len(missing) == 0:
        print(f"\nNo missing values — dataset is clean")
    else:
        print(f"\nMissing values found:\n{missing}")
    
    return merged


if __name__ == "__main__":
    merged = merge_datasets(
        "data/unsw_cleaned.csv",
        "data/cicids_cleaned.csv"
    )
    merged.to_csv("data/merged_dataset.csv", index=False)
    print(f"\nSaved to data/merged_dataset.csv")
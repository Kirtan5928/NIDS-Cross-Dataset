import pandas as pd
import numpy as np
import glob
import os

def load_cicids(data_folder):
    """Load all CICIDS2017 CSV files, clean and combine them."""
    
    # Find all CICIDS CSV files
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    # Exclude already cleaned files
    csv_files = [f for f in csv_files if "cleaned" not in f]
    print(f"Found {len(csv_files)} CICIDS files:")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, encoding='utf-8-sig')
        # Strip leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(f"Loaded {os.path.basename(f)}: {df.shape}")
    
    # Combine all files
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    
    # Clean the Label column
    print(f"\nLabel distribution before cleaning:\n{combined['Label'].value_counts()}")
    
    # Convert to binary — BENIGN = 0, everything else = 1
    combined["Label"] = combined["Label"].apply(
        lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
    )
    
    # Drop duplicate rows
    combined = combined.drop_duplicates()
    print(f"Shape after dropping duplicates: {combined.shape}")
    
    # Replace infinite values with NaN then drop
    combined = combined.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with missing values
    combined = combined.dropna()
    print(f"Shape after dropping NaN/inf: {combined.shape}")
    
    # Select matching features for Common Feature Schema
    # These CICIDS columns map to our UNSW features
    selected = {
        "Flow Duration"         : "dur",
        "Total Fwd Packets"     : "spkts",
        "Total Backward Packets": "dpkts",
        "Total Length of Fwd Packets": "sbytes",
        "Total Length of Bwd Packets": "dbytes",
        "Flow Packets/s"        : "rate",
        "Flow Bytes/s"          : "sload",
        "Bwd Packets/s"         : "dload",
        "Fwd IAT Mean"          : "sinpkt",
        "Bwd IAT Mean"          : "dinpkt",
        "Fwd Packet Length Std" : "sjit",
        "Bwd Packet Length Std" : "djit",
        "Init_Win_bytes_forward": "swin",
        "Init_Win_bytes_backward": "dwin",
        "Avg Fwd Segment Size"  : "smean",
        "Avg Bwd Segment Size"  : "dmean",
        "Label"                 : "Label"
    }
    
    # Keep only columns we need and rename them
    cols_to_keep = list(selected.keys())
    combined = combined[cols_to_keep]
    combined = combined.rename(columns=selected)
    
    # CICIDS doesn't have proto/service/state so fill with -1
    combined["proto"]   = -1
    combined["service"] = -1
    combined["state"]   = -1
    
    print(f"\nLabel distribution after cleaning:\n{combined['Label'].value_counts()}")
    print(f"Final CICIDS shape: {combined.shape}")
    
    return combined


if __name__ == "__main__":
    df = load_cicids("data")
    df.to_csv("data/cicids_cleaned.csv", index=False)
    print("\nSaved to data/cicids_cleaned.csv")
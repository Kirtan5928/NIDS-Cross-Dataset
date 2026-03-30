import pandas as pd

def load_unsw(train_path, test_path):
    """Load and clean UNSW-NB15 training and testing sets."""
    
    # Load both files
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    
    # Combine train and test into one dataframe
    df = pd.concat([train, test], ignore_index=True)
    print(f"UNSW combined shape: {df.shape}")
    
    # Drop columns we don't need
    # attack_cat is the attack category name — we only need binary label
    df = df.drop(columns=["attack_cat"])
    
    # Drop non-numeric columns that can't go into ML model directly
    # proto, service, state are categorical — we'll encode them
    df["proto"]   = df["proto"].astype("category").cat.codes
    df["service"] = df["service"].astype("category").cat.codes
    df["state"]   = df["state"].astype("category").cat.codes
    
    # Rename label column to standard name
    df = df.rename(columns={"label": "Label"})
    
    # Select the features we'll use in our Common Feature Schema
    selected = [
        "dur",        # Flow duration
        "spkts",      # Source packets
        "dpkts",      # Destination packets
        "sbytes",     # Source bytes
        "dbytes",     # Destination bytes
        "rate",       # Transfer rate
        "sload",      # Source load
        "dload",      # Destination load
        "sinpkt",     # Source inter-packet arrival time
        "dinpkt",     # Destination inter-packet arrival time
        "sjit",       # Source jitter
        "djit",       # Destination jitter
        "swin",       # Source TCP window size
        "dwin",       # Destination TCP window size
        "smean",      # Source mean packet size
        "dmean",      # Destination mean packet size
        "proto",
        "service",
        "state",
        "Label"
    ]
    
    df = df[selected]
    
    # Check for missing values after cleaning
    print(f"Missing values after cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Label distribution:\n{df['Label'].value_counts()}")
    print(f"Final UNSW shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    df = load_unsw(
        "data/UNSW_NB15_training-set.parquet",
        "data/UNSW_NB15_testing-set.parquet"
    )
    df.to_csv("data/unsw_cleaned.csv", index=False)
    print("\nSaved to data/unsw_cleaned.csv")
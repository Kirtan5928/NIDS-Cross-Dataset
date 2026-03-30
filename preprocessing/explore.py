import pandas as pd
import os

# ── UNSW-NB15 ──────────────────────────────────────────────────────────────
print("=" * 60)
print("UNSW-NB15 TRAINING SET")
print("=" * 60)

unsw_train = pd.read_parquet("data/UNSW_NB15_training-set.parquet")

print(f"Shape: {unsw_train.shape}")
print(f"\nColumns:\n{list(unsw_train.columns)}")
print(f"\nLabel column value counts:\n{unsw_train[unsw_train.columns[-1]].value_counts()}")
print(f"\nMissing values:\n{unsw_train.isnull().sum()[unsw_train.isnull().sum() > 0]}")

# ── CICIDS2017 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CICIDS2017 - FIRST FILE SAMPLE")
print("=" * 60)

cicids_file = "data/Monday-WorkingHours.pcap_ISCX.csv"
cicids = pd.read_csv(cicids_file, encoding='utf-8-sig')

print(f"Shape: {cicids.shape}")
print(f"\nColumns:\n{list(cicids.columns)}")
print(f"\nLabel column value counts:\n{cicids[cicids.columns[-1]].value_counts()}")
print(f"\nMissing values:\n{cicids.isnull().sum()[cicids.isnull().sum() > 0]}")
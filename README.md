# NIDS — Cross-Dataset Network Intrusion Detection System

A machine learning-based Network Intrusion Detection System trained on a unified corpus merging UNSW-NB15 and CICIDS2017 datasets. Features a live traffic stream simulator, threat analysis, and actionable remediation guidance.

---

## Results

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Aggregate Accuracy | **99.16%** | 98.69% |
| Attack Precision | 98% | 97% |
| Attack Recall | 98.3% | 98.8% |
| Attack F1 | 98% | 97% |
| Test Samples | 555,694 | 555,694 |

### Cross-Dataset Generalization

| Dataset | RF Accuracy | Notes |
|---------|------------|-------|
| UNSW-NB15 only | **97.28%** | Real network captures — harder |
| CICIDS2017 only | **99.91%** | Lab-generated traffic — separable |
| Generalization gap | 2.63% | Expected for merged corpus models |

> **Note:** High aggregate accuracy is partially attributed to CICIDS2017's lab-generated traffic being highly separable. UNSW-NB15 accuracy (97.28%) is the more honest real-world performance estimate.

---

## Architecture

```
Raw network flows (19 features)
        |
        v
Common Feature Schema
UNSW-NB15 + CICIDS2017 → merged_dataset.csv (2,778,471 samples)
        |
        v
Stratified 80/20 split
        |
        v
Random Forest ──┐
                ├── Evaluation + comparison
XGBoost ────────┘
        |
        v
Streamlit Dashboard
├── Overview        — confusion matrices, feature importance, cross-dataset breakdown
├── Live Stream     — simulated real-time classification + threat analysis + remediation
├── Manual Classifier — 19-feature input → ensemble verdict + remediation
└── Dataset Intel   — corpus statistics, correlation heatmap
```

---

## Features

- **Cross-dataset training** — Common Feature Schema merges UNSW-NB15 and CICIDS2017 into one unified corpus
- **Two models benchmarked** — Random Forest vs XGBoost, RF selected as final model
- **Live stream simulation** — real-time flow-by-flow classification with risk timeline and alert feed
- **Threat analysis** — infers likely attack type (Port Scan, DoS, Brute Force, Exfiltration) from flow features
- **Remediation guidance** — actionable SOC-style remediation steps per detected attack type
- **Feature explainability** — top feature contributions per attack category
- **Manual classifier** — enter any 19 flow features, get instant ensemble verdict

---

## Datasets

| Dataset | Samples | Type | Source |
|---------|---------|------|--------|
| UNSW-NB15 | 257,673 | Real network captures | UNSW Canberra |
| CICIDS2017 | 2,520,798 | Lab-generated traffic | University of New Brunswick |
| **Merged** | **2,778,471** | Common Feature Schema | This project |

---

## Features Used (19 flow-level)

| Feature | Description |
|---------|-------------|
| `dur` | Flow duration |
| `spkts` / `dpkts` | Source / destination packet count |
| `sbytes` / `dbytes` | Source / destination byte count |
| `rate` | Transfer rate |
| `sload` / `dload` | Source / destination load |
| `sinpkt` / `dinpkt` | Source / destination inter-packet time |
| `sjit` / `djit` | Source / destination jitter |
| `swin` / `dwin` | Source / destination TCP window size |
| `smean` / `dmean` | Source / destination mean packet size |
| `proto` / `service` / `state` | Protocol, service, connection state |

**Top features by importance:** `dwin` (11.3%) · `swin` (10.9%) · `djit` (9.9%) · `dmean` (9.3%) · `dbytes` (6.6%)

---

## Setup

### Prerequisites
- Python 3.10+
- ~5GB disk space for datasets

### Install

```bash
git clone https://github.com/Kirtan5928/NIDS-Cross-Dataset
cd NIDS-Cross-Dataset
pip install -r requirements.txt
```

### Run dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

### Retrain models

```bash
python models/train.py
```

---

## Project Structure

```
NIDS-Cross-Dataset/
├── dashboard/
│   └── app.py                  Streamlit dashboard (4 pages)
├── models/
│   ├── train.py                Training script — RF + XGBoost
│   ├── evaluate.py             Evaluation script
│   ├── random_forest.pkl       Trained RF model (~197MB)
│   └── xgboost.pkl             Trained XGBoost model
├── data/
│   ├── merged_dataset.csv      Unified 2.78M sample corpus
│   ├── unsw_cleaned.csv        Preprocessed UNSW-NB15
│   ├── cicids_cleaned.csv      Preprocessed CICIDS2017
│   └── UNSW_NB15_*.parquet     Raw UNSW files
└── requirements.txt
```

---

## Known Limitations

- **Binary classification only** — benign vs attack. No attack sub-type labels in merged corpus.
- **CICIDS2017 separability** — lab-generated traffic inflates aggregate accuracy. Real-world performance estimated at 97.28% (UNSW-NB15 holdout).
- **Flow-level features only** — adversary aware of top features (TCP window size, jitter) could craft evasive traffic.
- **No online learning** — model is static; distribution drift over time is not handled.

---

## Advanced Scope (Future Work)

- Multi-class attack classification using UNSW-NB15's 9 attack categories
- Adversarial robustness testing — craft flows to evade TCP window/jitter features
- Online learning with River library for distribution drift handling
- Graph-based detection for lateral movement patterns
- SHAP-based per-flow explainability

---

## Author

**Kirtan J Gowda** — B.E. CSE (Cybersecurity), MSRIT Bengaluru
[GitHub](https://github.com/Kirtan5928) · [LinkedIn](https://linkedin.com/in/kirtan-j-gowda-b0aa83360)
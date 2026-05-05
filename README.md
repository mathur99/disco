# Disco ML Take-Home: Click & Conversion Prediction

Two-model pipeline predicting CTR and CVR for Disco's post-purchase ad network.

## Models

- **CTR model**: `P(CLICKED=1)` — all impressions
- **CVR model**: `P(CONVERTED=1 | CLICKED=1)` — clicked impressions only

## Project structure

```
.
├── notebooks/          # EDA and modeling notebooks
├── src/                # Reusable modules (features, training, eval)
├── data/
│   ├── raw/            # Original CSVs (gitignored)
│   └── processed/      # Engineered features, splits
├── reports/            # Figures, metrics, writeup
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv assignment
source assignment/bin/activate
pip install -r requirements.txt
```

## Data

Place raw CSVs in `data/raw/`:
- `events_train.csv`
- `brand_metadata.csv`
- `user_metadata.csv`

## Results

_To be filled after modeling._

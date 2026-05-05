"""
Generate predictions on holdout data.

Loads model artifacts from data/processed/model_artifacts.joblib,
applies the same feature pipeline, and writes predictions_holdout.csv.

Usage:
    python src/predict.py --holdout data/raw/events_holdout.csv
    python src/predict.py --holdout data/raw/events_holdout.csv --output predictions_holdout.csv
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


ARTIFACT_PATH = Path("data/processed/model_artifacts.joblib")


def load_and_join(holdout_path: Path, brands_path: Path, users_path: Path) -> pd.DataFrame:
    events = pd.read_csv(holdout_path)
    brands = pd.read_csv(brands_path)
    users  = pd.read_csv(users_path)

    events["session_ts"] = pd.to_datetime(events["SESSION_START_AT_UTC"], format="mixed", utc=True)

    events = events.merge(
        brands[["BRAND_ID", "PRIMARY_CATEGORY"]].rename(columns={"PRIMARY_CATEGORY": "adv_category"}),
        on="BRAND_ID", how="left",
    )
    events = events.merge(
        brands[["PUBLISHER_UUID", "PRIMARY_CATEGORY"]].rename(columns={"PRIMARY_CATEGORY": "pub_category"}),
        on="PUBLISHER_UUID", how="left",
    )
    events = events.merge(users, on="IDENTITY_UUID", how="left")
    return events


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_anonymous"]  = df["IDENTITY_UUID"].isna().astype(int)
    df["is_bot"]        = df["OS_CLASS"].isin(["Cloud", "Hacker"]).astype(int)
    df["is_lead_gen"]   = df["WIDGET_TYPE"].isin(["LEAD_GEN", "SHOPIFY_NATIVE_LEAD_GEN"]).astype(int)
    df["same_category"] = (df["adv_category"] == df["pub_category"]).astype(int)
    return df


def apply_target_encode(col: pd.Series, lookup: dict) -> pd.Series:
    global_mean = lookup.get("__global_mean__", 0.0)
    return col.map(lookup).fillna(global_mean)


def build_features(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    df = df.copy()
    user_medians   = artifact["user_medians"]
    label_encoders = artifact["label_encoders"]
    ctr_lookups    = artifact["ctr_lookups"]
    cvr_lookups    = artifact["cvr_lookups"]
    user_cols      = artifact["user_cols"]
    cat_cols       = artifact["cat_cols"]

    for c in user_cols:
        df[c] = df[c].fillna(user_medians[c])
    df["log_ltv"] = np.log1p(df["LTV"])
    df["log_aov"] = np.log1p(df["AOV"])

    for col in cat_cols:
        le   = label_encoders[col]
        vals = df[col].fillna("unknown").astype(str)
        c2i  = {c: i for i, c in enumerate(le.classes_)}
        df[f"{col}_enc"] = vals.map(c2i).fillna(-1).astype(int)

    df["pub_ctr_enc"]   = apply_target_encode(df["PUBLISHER_UUID"], ctr_lookups["pub"])
    df["brand_ctr_enc"] = apply_target_encode(df["BRAND_ID"].astype(str), ctr_lookups["brand"])
    df["pub_cvr_enc"]   = apply_target_encode(df["PUBLISHER_UUID"], cvr_lookups["pub"])
    df["brand_cvr_enc"] = apply_target_encode(df["BRAND_ID"].astype(str), cvr_lookups["brand"])

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", required=True, help="Path to events_holdout.csv")
    parser.add_argument("--brands",  default="data/raw/brand_metadata.csv")
    parser.add_argument("--users",   default="data/raw/user_metadata.csv")
    parser.add_argument("--output",  default="predictions_holdout.csv")
    args = parser.parse_args()

    print(f"Loading artifacts from {ARTIFACT_PATH}...")
    artifact = joblib.load(ARTIFACT_PATH)

    ctr_model    = artifact["ctr_model"]
    cvr_model    = artifact["cvr_model"]
    features_ctr = artifact["features_ctr"]
    features_cvr = artifact["features_cvr"]

    print(f"Loading holdout data from {args.holdout}...")
    events = load_and_join(Path(args.holdout), Path(args.brands), Path(args.users))
    events = add_flags(events)
    events = build_features(events, artifact)

    print("Generating predictions...")
    events["pred_click"]   = ctr_model.predict_proba(events[features_ctr])[:, 1]
    events["pred_convert"] = cvr_model.predict_proba(events[features_cvr])[:, 1]

    out_cols = ["SESSION_ID", "pred_click", "pred_convert"]
    # Keep any original columns that exist
    for col in ["IDENTITY_UUID", "BRAND_ID", "PUBLISHER_UUID"]:
        if col in events.columns:
            out_cols.insert(1, col)

    out = events[out_cols].copy()
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out):,} predictions → {args.output}")
    print(f"pred_click   mean={out['pred_click'].mean():.4f}  min={out['pred_click'].min():.4f}  max={out['pred_click'].max():.4f}")
    print(f"pred_convert mean={out['pred_convert'].mean():.4f}  min={out['pred_convert'].min():.4f}  max={out['pred_convert'].max():.4f}")


if __name__ == "__main__":
    main()

"""
Generate predictions on holdout data.

Loads from models/latest.joblib by default. Override with --model.

Usage:
    python src/predict.py --holdout data/raw/events_holdout.csv
    python src/predict.py --holdout data/raw/events_holdout.csv --output predictions_holdout.csv
    python src/predict.py --holdout data/raw/events_holdout.csv --model models/artifacts_20260506_120000.joblib
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODELS_DIR = Path("models")


def resolve_model(model_arg: str | None) -> Path:
    if model_arg:
        return Path(model_arg)
    latest = MODELS_DIR / "latest.joblib"
    if latest.exists():
        return latest
    # Fallback: most recent artifact by filename
    artifacts = sorted(MODELS_DIR.glob("artifacts_*.joblib"))
    if not artifacts:
        raise FileNotFoundError("No model artifacts found in models/. Run src/train.py first.")
    return artifacts[-1]


def load_and_join(holdout_path, brands_path, users_path):
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


def add_flags(df):
    df = df.copy()
    df["is_anonymous"]  = df["IDENTITY_UUID"].isna().astype(int)
    df["is_bot"]        = df["OS_CLASS"].isin(["Cloud", "Hacker"]).astype(int)
    df["is_lead_gen"]   = df["WIDGET_TYPE"].isin(["LEAD_GEN", "SHOPIFY_NATIVE_LEAD_GEN"]).astype(int)
    df["same_category"] = (df["adv_category"] == df["pub_category"]).astype(int)
    return df


def apply_target_encode(col, lookup):
    return col.map(lookup).fillna(lookup.get("__global_mean__", 0.0))


def build_features(df, artifact):
    df             = df.copy()
    user_medians   = artifact["user_medians"]
    label_encoders = artifact["label_encoders"]
    ctr_lookups    = artifact["ctr_lookups"]
    cvr_lookups    = artifact["cvr_lookups"]

    for c in artifact["user_cols"]:
        df[c] = df[c].fillna(user_medians[c])
    df["log_ltv"] = np.log1p(df["LTV"])
    df["log_aov"] = np.log1p(df["AOV"])

    for col in artifact["cat_cols"]:
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
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--brands",  default="data/raw/brand_metadata.csv")
    parser.add_argument("--users",   default="data/raw/user_metadata.csv")
    parser.add_argument("--output",  default="predictions_holdout.csv")
    parser.add_argument("--model",   default=None, help="Path to specific artifact .joblib (default: models/latest.joblib)")
    args = parser.parse_args()

    model_path = resolve_model(args.model)
    print(f"Loading model: {model_path}")
    artifact = joblib.load(model_path)
    print(f"  Run timestamp: {artifact.get('run_ts', 'unknown')}")

    print(f"Loading holdout: {args.holdout}")
    events = load_and_join(Path(args.holdout), Path(args.brands), Path(args.users))
    events = add_flags(events)
    events = build_features(events, artifact)

    print("Generating predictions...")
    events["pred_click"]   = artifact["ctr_model"].predict_proba(events[artifact["features_ctr"]])[:, 1]
    events["pred_convert"] = artifact["cvr_model"].predict_proba(events[artifact["features_cvr"]])[:, 1]

    out_cols = ["SESSION_ID", "IDENTITY_UUID", "BRAND_ID", "PUBLISHER_UUID", "pred_click", "pred_convert"]
    out_cols = [c for c in out_cols if c in events.columns]
    events[out_cols].to_csv(args.output, index=False)

    print(f"Saved {len(events):,} predictions → {args.output}")
    print(f"pred_click   mean={events['pred_click'].mean():.4f}")
    print(f"pred_convert mean={events['pred_convert'].mean():.4f}")


if __name__ == "__main__":
    main()

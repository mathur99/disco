"""
Final training pipeline.

Trains LightGBM (CTR) and XGBoost (CVR) on the full events_train dataset,
saves models + encoders to data/processed/ for use by predict.py.

Usage:
    python src/train.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 42
DATA_DIR  = Path("data/raw")
OUT_DIR   = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_PATH = DATA_DIR / "events_train.csv"
BRANDS_PATH = DATA_DIR / "brand_metadata.csv"
USERS_PATH  = DATA_DIR / "user_metadata.csv"

CAT_COLS = [
    "PAGE_TYPE", "WIDGET_TYPE", "BRAND_DISPLAY_PLACEMENT",
    "OS_CLASS", "adv_category", "pub_category",
]
USER_COLS = ["LTV", "AOV", "NUM_PURCHASES", "BOUGHT_SUBSCRIPTION", "DISCOUNT_SHOPPER", "REFUNDED"]

FEATURES = [
    "PAGE_TYPE_enc", "WIDGET_TYPE_enc", "BRAND_DISPLAY_PLACEMENT_enc",
    "OS_CLASS_enc", "WIDGET_VERSION",
    "is_anonymous", "is_bot", "is_lead_gen", "same_category",
    "pub_ctr_enc", "brand_ctr_enc",
    "adv_category_enc", "pub_category_enc",
    "log_ltv", "log_aov",
    "NUM_PURCHASES", "BOUGHT_SUBSCRIPTION", "DISCOUNT_SHOPPER", "REFUNDED",
]

FEATURES_CVR = [f for f in FEATURES if f not in ("pub_ctr_enc", "brand_ctr_enc")] + [
    "pub_cvr_enc", "brand_cvr_enc"
]


def load_and_join():
    events = pd.read_csv(EVENTS_PATH)
    brands = pd.read_csv(BRANDS_PATH)
    users  = pd.read_csv(USERS_PATH)

    events["CLICKED"]    = events["CLICKED"].astype(int)
    events["CONVERTED"]  = events["CONVERTED"].astype(int)
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


def smooth_target_encode(train_col, target, k=30, fallback=None):
    """Returns (encoded series for train_col, lookup dict)."""
    global_mean = target.mean() if fallback is None else fallback
    agg = pd.DataFrame({"key": train_col.values, "y": target.values})
    agg = agg.groupby("key")["y"].agg(["sum", "count"])
    agg["enc"] = (agg["sum"] + k * global_mean) / (agg["count"] + k)
    lookup = agg["enc"].to_dict()
    lookup["__global_mean__"] = global_mean
    tr_enc = train_col.map(lookup).fillna(global_mean)
    return tr_enc, lookup


def apply_target_encode(col, lookup):
    global_mean = lookup.get("__global_mean__", 0.0)
    return col.map(lookup).fillna(global_mean)


def build_features(df, label_encoders, user_medians, ctr_lookups, cvr_lookups=None):
    df = df.copy()

    for c in USER_COLS:
        df[c] = df[c].fillna(user_medians[c])
    df["log_ltv"] = np.log1p(df["LTV"])
    df["log_aov"] = np.log1p(df["AOV"])

    for col in CAT_COLS:
        le = label_encoders[col]
        vals = df[col].fillna("unknown").astype(str)
        c2i = {c: i for i, c in enumerate(le.classes_)}
        df[f"{col}_enc"] = vals.map(c2i).fillna(-1).astype(int)

    df["pub_ctr_enc"]   = apply_target_encode(df["PUBLISHER_UUID"], ctr_lookups["pub"])
    df["brand_ctr_enc"] = apply_target_encode(df["BRAND_ID"].astype(str), ctr_lookups["brand"])

    if cvr_lookups is not None:
        df["pub_cvr_enc"]   = apply_target_encode(df["PUBLISHER_UUID"], cvr_lookups["pub"])
        df["brand_cvr_enc"] = apply_target_encode(df["BRAND_ID"].astype(str), cvr_lookups["brand"])

    return df


def evaluate(model, X_va, y_va):
    probs = model.predict_proba(X_va)[:, 1]
    return {
        "ROC-AUC":  round(roc_auc_score(y_va, probs), 4),
        "PR-AUC":   round(average_precision_score(y_va, probs), 4),
        "Log-loss": round(log_loss(y_va, probs), 4),
    }


def main():
    print("Loading data...")
    events = load_and_join()
    events = add_flags(events)

    # Time-based split for evaluation (last 14 days = val)
    cutoff = events["session_ts"].max() - pd.Timedelta(days=14)
    train  = events[events["session_ts"] <= cutoff].copy().reset_index(drop=True)
    val    = events[events["session_ts"] >  cutoff].copy().reset_index(drop=True)
    print(f"Train: {len(train):,}  |  Val: {len(val):,}  (cutoff {cutoff.date()})")

    # Fit label encoders on train
    label_encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(train[col].fillna("unknown").astype(str))
        label_encoders[col] = le

    # Fit user medians on train
    user_medians = train[USER_COLS].median()

    # CTR target encodings on train
    ctr_pub_enc,   ctr_pub_lookup   = smooth_target_encode(train["PUBLISHER_UUID"], train["CLICKED"])
    ctr_brand_enc, ctr_brand_lookup = smooth_target_encode(train["BRAND_ID"].astype(str), train["CLICKED"])
    ctr_lookups = {"pub": ctr_pub_lookup, "brand": ctr_brand_lookup}

    # CVR target encodings on clicked train subset
    train_clicked = train[train["CLICKED"] == 1].copy()
    cvr_pub_enc,   cvr_pub_lookup   = smooth_target_encode(train_clicked["PUBLISHER_UUID"], train_clicked["CONVERTED"])
    cvr_brand_enc, cvr_brand_lookup = smooth_target_encode(train_clicked["BRAND_ID"].astype(str), train_clicked["CONVERTED"])
    cvr_lookups = {"pub": cvr_pub_lookup, "brand": cvr_brand_lookup}

    # Build feature matrices (eval)
    train_fe = build_features(train, label_encoders, user_medians, ctr_lookups, cvr_lookups)
    val_fe   = build_features(val,   label_encoders, user_medians, ctr_lookups, cvr_lookups)

    X_tr = train_fe[FEATURES];        y_tr_ctr = train_fe["CLICKED"]
    X_va = val_fe[FEATURES];          y_va_ctr = val_fe["CLICKED"]

    train_cl_fe = train_fe[train_fe["CLICKED"] == 1]
    val_cl_fe   = val_fe[val_fe["CLICKED"] == 1]
    X_tr_cvr = train_cl_fe[FEATURES_CVR]; y_tr_cvr = train_cl_fe["CONVERTED"]
    X_va_cvr = val_cl_fe[FEATURES_CVR];   y_va_cvr = val_cl_fe["CONVERTED"]

    pos_weight_ctr = int((y_tr_ctr == 0).sum() / (y_tr_ctr == 1).sum())
    pos_weight_cvr = int((y_tr_cvr == 0).sum() / (y_tr_cvr == 1).sum())

    print(f"\n--- Validation evaluation ---")

    print("Training CTR model (LightGBM)...")
    ctr_model_eval = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        is_unbalance=True, random_state=SEED, verbose=-1,
    )
    ctr_model_eval.fit(X_tr, y_tr_ctr)
    ctr_metrics = evaluate(ctr_model_eval, X_va, y_va_ctr)
    print(f"  CTR val  ROC-AUC={ctr_metrics['ROC-AUC']}  PR-AUC={ctr_metrics['PR-AUC']}  LogLoss={ctr_metrics['Log-loss']}")

    print("Training CVR model (XGBoost)...")
    cvr_model_eval = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=pos_weight_cvr, random_state=SEED, verbosity=0, eval_metric="logloss",
    )
    cvr_model_eval.fit(X_tr_cvr, y_tr_cvr)
    cvr_metrics = evaluate(cvr_model_eval, X_va_cvr, y_va_cvr)
    print(f"  CVR val  ROC-AUC={cvr_metrics['ROC-AUC']}  PR-AUC={cvr_metrics['PR-AUC']}  LogLoss={cvr_metrics['Log-loss']}")

    # --- Retrain on ALL data for final model ---
    print("\n--- Retraining on full dataset ---")

    # Re-fit encoders on all data
    label_encoders_full = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(events[col].fillna("unknown").astype(str))
        label_encoders_full[col] = le

    user_medians_full = events[USER_COLS].median()

    _, ctr_pub_lookup_full   = smooth_target_encode(events["PUBLISHER_UUID"], events["CLICKED"])
    _, ctr_brand_lookup_full = smooth_target_encode(events["BRAND_ID"].astype(str), events["CLICKED"])
    ctr_lookups_full = {"pub": ctr_pub_lookup_full, "brand": ctr_brand_lookup_full}

    events_clicked = events[events["CLICKED"] == 1]
    _, cvr_pub_lookup_full   = smooth_target_encode(events_clicked["PUBLISHER_UUID"], events_clicked["CONVERTED"])
    _, cvr_brand_lookup_full = smooth_target_encode(events_clicked["BRAND_ID"].astype(str), events_clicked["CONVERTED"])
    cvr_lookups_full = {"pub": cvr_pub_lookup_full, "brand": cvr_brand_lookup_full}

    events_fe  = build_features(events, label_encoders_full, user_medians_full, ctr_lookups_full, cvr_lookups_full)
    events_cl  = events_fe[events_fe["CLICKED"] == 1]

    pos_full_ctr = int((events_fe["CLICKED"] == 0).sum() / (events_fe["CLICKED"] == 1).sum())
    pos_full_cvr = int((events_cl["CONVERTED"] == 0).sum() / (events_cl["CONVERTED"] == 1).sum())

    ctr_model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        is_unbalance=True, random_state=SEED, verbose=-1,
    )
    ctr_model.fit(events_fe[FEATURES], events_fe["CLICKED"])
    print("  CTR model trained on full data.")

    cvr_model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=pos_full_cvr, random_state=SEED, verbosity=0, eval_metric="logloss",
    )
    cvr_model.fit(events_cl[FEATURES_CVR], events_cl["CONVERTED"])
    print("  CVR model trained on full data.")

    # Save everything
    artifact = {
        "ctr_model": ctr_model,
        "cvr_model": cvr_model,
        "label_encoders": label_encoders_full,
        "user_medians": user_medians_full,
        "ctr_lookups": ctr_lookups_full,
        "cvr_lookups": cvr_lookups_full,
        "features_ctr": FEATURES,
        "features_cvr": FEATURES_CVR,
        "cat_cols": CAT_COLS,
        "user_cols": USER_COLS,
        "val_metrics": {"CTR": ctr_metrics, "CVR": cvr_metrics},
    }
    joblib.dump(artifact, OUT_DIR / "model_artifacts.joblib")
    print(f"\nArtifacts saved → {OUT_DIR / 'model_artifacts.joblib'}")

    metrics_path = OUT_DIR / "val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"CTR": ctr_metrics, "CVR": cvr_metrics}, f, indent=2)
    print(f"Val metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()

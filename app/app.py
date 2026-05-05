"""Flask app — CTR/CVR prediction UI with live simulation."""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, render_template, request, jsonify

ROOT          = Path(__file__).parent.parent
ARTIFACT_PATH = ROOT / "data/processed/model_artifacts.joblib"
BRANDS_PATH   = ROOT / "data/raw/brand_metadata.csv"
USERS_PATH    = ROOT / "data/raw/user_metadata.csv"

app = Flask(__name__)

# --- Load once at startup ---
artifact       = joblib.load(ARTIFACT_PATH)
brands_df      = pd.read_csv(BRANDS_PATH)
users_df       = pd.read_csv(USERS_PATH)

ctr_model      = artifact["ctr_model"]
cvr_model      = artifact["cvr_model"]
features_ctr   = artifact["features_ctr"]
features_cvr   = artifact["features_cvr"]
label_encoders = artifact["label_encoders"]
user_medians   = artifact["user_medians"]
ctr_lookups    = artifact["ctr_lookups"]
cvr_lookups    = artifact["cvr_lookups"]
cat_cols       = artifact["cat_cols"]
user_cols      = artifact["user_cols"]

publishers_list  = brands_df[["PUBLISHER_UUID", "BRAND_NAME", "PRIMARY_CATEGORY"]].sort_values("BRAND_NAME").to_dict("records")
advertisers_list = brands_df[["BRAND_ID", "BRAND_NAME", "PRIMARY_CATEGORY", "DESCRIPTION"]].sort_values("BRAND_NAME").to_dict("records")

# Maps for JS category auto-fill
pub_cat_map  = brands_df.set_index("PUBLISHER_UUID")["PRIMARY_CATEGORY"].to_dict()
adv_cat_map  = brands_df.set_index("BRAND_ID")["PRIMARY_CATEGORY"].to_dict()
adv_desc_map = brands_df.set_index("BRAND_ID")["DESCRIPTION"].to_dict()
adv_name_map = brands_df.set_index("BRAND_ID")["BRAND_NAME"].to_dict()
pub_name_map = brands_df.set_index("PUBLISHER_UUID")["BRAND_NAME"].to_dict()

default_ltv = round(float(users_df["LTV"].median()), 2)
default_aov = round(float(users_df["AOV"].median()), 2)
default_np  = int(users_df["NUM_PURCHASES"].median())
max_ltv     = round(float(users_df["LTV"].quantile(0.95)), 0)
max_aov     = round(float(users_df["AOV"].quantile(0.95)), 0)


def apply_target_encode(col: pd.Series, lookup: dict) -> pd.Series:
    return col.map(lookup).fillna(lookup.get("__global_mean__", 0.0))


def predict_single(d: dict) -> dict:
    pub_uuid  = d["publisher_uuid"]
    brand_id  = int(d["brand_id"])

    row = {
        "PUBLISHER_UUID":          pub_uuid,
        "BRAND_ID":                brand_id,
        "PAGE_TYPE":               d["page_type"],
        "WIDGET_TYPE":             d["widget_type"],
        "BRAND_DISPLAY_PLACEMENT": d["placement"],
        "OS_CLASS":                d["os_class"],
        "WIDGET_VERSION":          float(d.get("widget_version", 0.2)),
        "IDENTITY_UUID":           d.get("identity_uuid") or None,
        "LTV":                     float(d.get("ltv", default_ltv)),
        "AOV":                     float(d.get("aov", default_aov)),
        "NUM_PURCHASES":           int(d.get("num_purchases", default_np)),
        "BOUGHT_SUBSCRIPTION":     int(d.get("bought_subscription", 0)),
        "DISCOUNT_SHOPPER":        int(d.get("discount_shopper", 0)),
        "REFUNDED":                int(d.get("refunded", 0)),
        "adv_category":            adv_cat_map.get(brand_id, "unknown"),
        "pub_category":            pub_cat_map.get(pub_uuid, "unknown"),
    }

    df = pd.DataFrame([row])

    df["is_anonymous"]  = df["IDENTITY_UUID"].isna().astype(int)
    df["is_bot"]        = df["OS_CLASS"].isin(["Cloud", "Hacker"]).astype(int)
    df["is_lead_gen"]   = df["WIDGET_TYPE"].isin(["LEAD_GEN", "SHOPIFY_NATIVE_LEAD_GEN"]).astype(int)
    df["same_category"] = (df["adv_category"] == df["pub_category"]).astype(int)

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

    p_click   = float(ctr_model.predict_proba(df[features_ctr])[0, 1])
    p_convert = float(cvr_model.predict_proba(df[features_cvr])[0, 1])
    ecvr      = p_click * p_convert

    avg_ctr = ctr_lookups["pub"].get("__global_mean__", 0.0266)
    avg_cvr = cvr_lookups["pub"].get("__global_mean__", 0.034)

    return {
        "pred_click":    round(p_click,   4),
        "pred_convert":  round(p_convert, 4),
        "ecvr":          round(ecvr,      4),
        "click_1_in":    max(1, round(1 / p_click))   if p_click   > 0.001 else 999,
        "convert_1_in":  max(1, round(1 / p_convert)) if p_convert > 0.001 else 999,
        "avg_ctr":       round(avg_ctr,  4),
        "avg_cvr":       round(avg_cvr,  4),
        "vs_avg_ctr":    round((p_click   - avg_ctr) / avg_ctr * 100, 1),
        "vs_avg_cvr":    round((p_convert - avg_cvr) / avg_cvr * 100, 1),
        "brand_name":    adv_name_map.get(int(d["brand_id"]), "Brand"),
        "pub_name":      pub_name_map.get(d["publisher_uuid"], "Publisher"),
        "brand_desc":    adv_desc_map.get(int(d["brand_id"]), ""),
        "adv_category":  adv_cat_map.get(int(d["brand_id"]), "unknown"),
        "same_category": int(row["adv_category"] == row["pub_category"]),
        "placement":     d["placement"],
        "page_type":     d["page_type"],
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        publishers    = publishers_list,
        advertisers   = advertisers_list,
        pub_cat_map   = json.dumps(pub_cat_map),
        adv_cat_map   = json.dumps({str(k): v for k, v in adv_cat_map.items()}),
        adv_desc_map  = json.dumps({str(k): v for k, v in adv_desc_map.items()}),
        adv_name_map  = json.dumps({str(k): v for k, v in adv_name_map.items()}),
        pub_name_map  = json.dumps(pub_name_map),
        default_ltv   = default_ltv,
        default_aov   = default_aov,
        default_np    = default_np,
        max_ltv       = max_ltv,
        max_aov       = max_aov,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        result = predict_single(request.json)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5050)

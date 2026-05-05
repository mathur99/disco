# Disco — Click & Conversion Prediction

Two-model ML pipeline for Disco's post-purchase ad network: predict which shopper will click an ad, and which click will turn into a purchase.

---

## Quick start

```bash
python3.11 -m venv pyenv && source pyenv/bin/activate
pip install -r requirements.txt

python src/train.py                              # trains + saves models to data/processed/
python app/app.py                                # prediction UI at http://127.0.0.1:5050

# when holdout arrives:
python src/predict.py --holdout data/raw/events_holdout.csv
```

## Project layout

```
notebooks/          01_eda.ipynb, 02_modeling.ipynb
src/                train.py — full pipeline; predict.py — holdout inference
app/                Flask prediction UI with live simulation
data/raw/           events_train.csv, brand_metadata.csv, user_metadata.csv
data/processed/     model_artifacts.joblib, val_metrics.json
reports/            feature importance + EDA plots
```

---

## The problem

Each row in `events_train.csv` is one impression: a shopper on a publisher's post-purchase page being shown an advertiser brand inside Disco's widget.

**Two targets:**
- `CLICKED` — did the shopper click through to the brand? (`CTR = 2.66%`)
- `CONVERTED` — given a click, did they buy? (`CVR = 3.41%` on the clicked subset)

Both are heavily imbalanced (37:1 and 28:1). Accuracy is useless here — a model that predicts all-zeros scores 97% and is worthless for ranking.

---

## Data — what we found

| Table | Rows | Key finding |
|---|---|---|
| `events_train.csv` | 471,676 | 1 row = 1 impression; SESSION_ID is unique (no multi-brand sessions) |
| `brand_metadata.csv` | 342 | 100% advertiser + publisher category coverage |
| `user_metadata.csv` | 150,142 | Covers majority of shoppers; 0.56% anonymous (no IDENTITY_UUID) |

**Class imbalance**

| Target | Positives | Rate | Neg:Pos |
|---|---|---|---|
| CLICKED | 12,544 / 471,676 | 2.66% | 37:1 |
| CONVERTED (clicked only) | 428 / 12,544 | 3.41% | 28:1 |

**Signal by surface — CTR varies enormously**

| PAGE_TYPE | CTR | PLACEMENT | CTR | WIDGET_TYPE | CTR |
|---|---|---|---|---|---|
| THANK_YOU | 3.47% | FULLSCREEN | 6.25% | ESSENTIAL | 5.52% |
| ORDER_TRACKING | 2.52% | PULLUP | 2.67% | SHOPIFY_NATIVE | 3.43% |
| ORDER_STATUS | 1.23% | INLINE | 2.47% | APP_NATIVE | 2.37% |
| | | | | LEAD_GEN | ~0.00% |

> FULLSCREEN CTR is 3× INLINE. A model without placement as a feature will confuse surface quality with brand quality.

> LEAD_GEN widgets optimize for email capture, not click-through — their "conversion" means something different. Flagged separately.

> `Unknown` OS has the **highest CTR (5.56%)** — likely a specific SDK integration, not noise. Kept as its own category.

**Publisher & brand concentration** — long-tailed distribution. A handful of publishers drive most impressions. Smoothed target encoding handles this cleanly.

---

## Feature engineering

19 features total, built from joins across all three tables.

| Feature | Type | Why |
|---|---|---|
| `pub_ctr_enc` | smoothed target enc | Publisher's historical CTR — strongest signal in the whole model |
| `brand_ctr_enc` | smoothed target enc | Brand's historical CTR across all publishers |
| `pub_cvr_enc` / `brand_cvr_enc` | smoothed target enc | CVR variants for the CVR model (fitted on clicked subset only) |
| `PAGE_TYPE_enc` | label enc | 3.47% vs 1.23% CTR spread across values |
| `WIDGET_TYPE_enc` | label enc | LEAD_GEN vs ESSENTIAL — 0% vs 5.5% |
| `BRAND_DISPLAY_PLACEMENT_enc` | label enc | FULLSCREEN 3× INLINE |
| `OS_CLASS_enc` | label enc | Unknown SDK surface has highest CTR |
| `adv_category_enc`, `pub_category_enc` | label enc | Advertiser + publisher vertical |
| `same_category` | binary flag | Do publisher and advertiser share a category? Relevance signal |
| `is_lead_gen` | binary flag | Zero-CTR widgets need explicit flagging |
| `is_bot` | binary flag | Cloud/Hacker OS — 118 rows, kept but flagged |
| `is_anonymous` | binary flag | Null IDENTITY_UUID → no user features available |
| `log_ltv`, `log_aov` | log transform | LTV/AOV are right-skewed; log stabilises the scale |
| `NUM_PURCHASES`, `BOUGHT_SUBSCRIPTION`, `DISCOUNT_SHOPPER`, `REFUNDED` | raw/binary | Shopper behaviour from user_metadata |

### Smoothed target encoding — the key design choice

For `PUBLISHER_UUID` and `BRAND_ID` (high cardinality), we use:

```
enc = (sum_clicks + k × global_mean) / (count + k)    k = 30
```

A publisher with 2 impressions and 1 click doesn't get 50% CTR — it gets pulled toward the global mean. `k=30` means you need ~30 impressions before the publisher's own CTR dominates. This is the most impactful single feature.

---

## Modeling precautions

These are the places where it's easy to cheat without realising it.

**1. Time-based split, not random**

Split cutoff: last 14 days → validation. Random splitting leaks future publisher/user behaviour into training. A publisher that appeared only in the last week would be in both train and val, making the model look better than it is on unseen publishers.

```
Train: Jan 31 → Apr 9 (399,821 rows)   Val: Apr 9 → Apr 23 (71,855 rows)
```

**2. All encodings fitted on train only, applied to val**

Medians for user feature imputation, label encoder classes, and target encoding aggregates are computed on `train` then applied to `val`. If you fit encoders on the full dataset before splitting, the val set "knows" things it shouldn't.

**3. CVR model trained on clicked-only subset**

The CVR model only sees the 10,632 clicked training impressions. It re-encodes publisher and brand using **CVR** (not CTR) as the target — a publisher with high CTR doesn't necessarily have high CVR. Different signal, different encoding.

**4. No accuracy as a metric**

With 37:1 imbalance, a model predicting all zeros gets 97.3% accuracy. We use:
- **PR-AUC** (primary) — precision-recall. Penalises missing positives. Most sensitive to imbalance.
- **ROC-AUC** — good for ranking quality, less sensitive to threshold.
- **Log-loss** — penalises overconfident wrong predictions. Useful for calibration.

**5. Class imbalance handling**

- LightGBM: `is_unbalance=True` (reweights loss)
- XGBoost: `scale_pos_weight = neg/pos count` (~36 for CTR, ~29 for CVR)
- Shallower trees for CVR (`num_leaves=15`, `max_depth=4`) — 428 positives isn't enough to justify deep splits

**6. Unknown categories in val**

Val may contain publisher/brand IDs not seen in train (new brands launched during the hold-out window). Label encoder maps unseen values to `-1`; target encoder falls back to global mean. This is handled explicitly — not left to chance.

---

## Model comparison

Three families evaluated on validation. Winner chosen by PR-AUC.

**CTR** (all 471K impressions → 12,544 positives)

| Model | ROC-AUC | PR-AUC | Log-loss |
|---|---|---|---|
| **LightGBM** ✓ | **0.7319** | **0.0729** | 0.6046 |
| XGBoost | 0.7329 | 0.0727 | 0.5974 |
| Logistic Regression | 0.7078 | 0.0678 | 0.6775 |
| Random baseline | 0.5000 | 0.0266 | — |

**CVR** (clicked-only → 12,544 impressions, 428 positives)

| Model | ROC-AUC | PR-AUC | Log-loss |
|---|---|---|---|
| **XGBoost** ✓ | **0.6691** | **0.0714** | 0.6979 |
| Logistic Regression | 0.6578 | 0.0682 | 0.6656 |
| LightGBM | 0.5943 | 0.0576 | 0.5796 |
| Random baseline | 0.5000 | 0.0341 | — |

LightGBM lost on CVR because it overfits more aggressively on small data — even with `num_leaves=15`, 428 positives isn't enough. XGBoost's `max_depth=4` constraint held better.

LogReg gap vs trees (~0.005 PR-AUC) tells us non-linear interactions exist but aren't massive — the features are already doing a lot of the heavy lifting.

---

## Results — how good is it?

| Model | PR-AUC | vs random | ROC-AUC | vs random |
|---|---|---|---|---|
| CTR (LightGBM) | 0.0729 | **2.7× lift** | 0.7319 | +46% above 0.5 |
| CVR (XGBoost) | 0.0714 | **2.1× lift** | 0.6691 | +34% above 0.5 |

**What 2.7× lift actually means:** if you rank 1,000 impressions by model score and take the top 100, you'll get ~2.7× more clicks than if you picked 100 at random. That's real money — better ranking → fewer wasted impressions → higher publisher revenue and advertiser ROI.

**Top 5 CTR features** (LightGBM split importance):
1. `pub_ctr_enc` — publisher CTR history dominates everything else
2. `brand_ctr_enc` — brand CTR history
3. `WIDGET_TYPE_enc` — LEAD_GEN vs real widgets
4. `BRAND_DISPLAY_PLACEMENT_enc` — FULLSCREEN vs INLINE
5. `adv_category_enc` — Loyalty & Affiliates vs others

---

## What's missing / what we'd do next

**Short-term (highest ROI)**

- **Calibration** — gradient boosting `predict_proba` is not well-calibrated. Isotonic regression or Platt scaling on a holdout would fix log-loss and make `CTR × CVR = eCVR` reliably composable for ranking.
- **Subcategory Jaccard** — `SUBCATEGORIES` is a JSON list. Publisher–advertiser subcategory overlap as a continuous feature would sharpen `same_category` from binary to graded.
- **Separate LEAD_GEN model** — CVR semantics for LEAD_GEN (email capture) differ from purchase conversion. A sub-model trained only on LEAD_GEN clicks would be more accurate for that surface.

**Medium-term**

- **Brand description embeddings** — `DESCRIPTION` → sentence embedding (e.g. `all-MiniLM-L6-v2`) solves cold-start for new brands that have no CTR history yet. Smoothed encoding falls back to global mean; embeddings give a meaningful starting point.
- **User × brand affinity features** — cross-features between `DISCOUNT_SHOPPER`/`BOUGHT_SUBSCRIPTION` and `adv_category`. A discount shopper is probably more likely to click Loyalty & Affiliates brands specifically.

**Production concerns to raise**

| Concern | Problem | Mitigation |
|---|---|---|
| Label delay | `CONVERTED` lags clicks by hours/days — you don't know if a click converted yet | Fixed attribution window (e.g. 48h) or survival model |
| Distribution shift | New publishers/brands arrive without CTR history | Cold-start layer or Thompson sampling bandit for new entrants |
| Feature serving latency | `pub_ctr_enc` needs recent click aggregates in real-time | Feature store with sub-10ms lookups; not a batch join |
| Bot traffic | Current model flags but doesn't filter bots | Upstream filter — model shouldn't need to learn to ignore scrapers |
| Feedback loop | Model boosts high-CTR publishers → they get more impressions → CTR encoding becomes self-fulfilling | Periodic retraining with exploration budget |

---

## App

Interactive prediction UI + live simulation at `http://127.0.0.1:5050`

```bash
python app/app.py
```

Configure any publisher/brand/shopper combination, hit predict, see CTR + CVR + funnel. "Run Live Simulation" opens a fake post-purchase page and animates whether the shopper clicks and converts, driven by the model's probabilities.

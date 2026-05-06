# Disco — Click & Conversion Prediction

Two-model ML pipeline for Disco's post-purchase ad network: predict which shopper will click an ad, and which click will turn into a purchase.

---

## App *(not asked for, built anyway)*

Numbers on a page are hard to trust. A model can say "this impression has a 6.2% CTR" but that means nothing until you actually see what the shopper sees — the page, the widget, the brand ad — and ask yourself *"would I click this?"*

So I built a prediction UI with a live ad simulation. It wasn't part of the assignment. It exists because humans are bad at reasoning about probabilities in the abstract but very good at reasoning about things they can see.

```bash
python app/app.py   # → http://127.0.0.1:5050
```

**Prediction side** — pick any publisher, advertiser brand, page type, placement, and shopper profile. Hit predict. You get:
- CTR and CVR as percentages + plain-language ("1 in 40 shoppers will click")
- Gauges showing how this impression compares to the network average
- A funnel: out of 1,000 impressions shown, how many click, how many buy

**Simulation side** — hit "Run Live Simulation". A fake post-purchase order page loads. The Disco widget appears showing the advertiser's brand. An animated cursor moves across the screen, hovers over the ad, and then — based on the model's predicted probability — either clicks or scrolls away. If it clicks, you watch whether the shopper converts or bounces.

Every run is a fresh random draw against the model's probabilities. Run it ten times on the same configuration and you'll see the randomness — sometimes clicks, sometimes doesn't — which is exactly how probability works in real life. That's the point.

> The model says "3% CTR." The simulation lets you *watch* 3% happen. That's a different kind of understanding.

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
data/raw/           events_train.csv, brand_metadata.csv, user_metadata.csv  (gitignored)
data/processed/     val_metrics.json
models/             artifacts_YYYYMMDD_HHMMSS.joblib  (gitignored, binary)
                    metrics_YYYYMMDD_HHMMSS.json
                    latest.joblib  →  symlink to most recent artifact
reports/            feature importance + EDA plots
```

Each `src/train.py` run produces a timestamped artifact and updates `models/latest.joblib`. `predict.py` and the Flask app both resolve `latest.joblib` automatically, or accept `--model path/to/specific.joblib` for rollback.

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

The raw data has IDs, strings, skewed numbers, and JSON blobs — none of which a model can use directly. Below are the features we *created*, why each one exists, and a concrete example of how it's built.

---

### `pub_ctr_enc` and `brand_ctr_enc` — publisher/brand click history

**What it is:** Each publisher and brand gets a single number — their historical click rate — attached to every impression they appear in.

**Why:** The publisher a shopper lands on is the strongest predictor of whether they'll click. Some publishers drive highly engaged shoppers; others get low-intent traffic. Raw publisher ID (a UUID string) means nothing to a model — this encoding turns it into signal.

**How:**
```python
# For publisher e99c91ab with 1,000 impressions and 32 clicks:
global_ctr = 0.0266            # average across all publishers
k          = 30                # smoothing strength

enc = (32 + 30 × 0.0266) / (1000 + 30) = 0.0318   # close to their real 3.2% CTR

# For a new publisher with only 2 impressions and 1 click:
enc = (1 + 30 × 0.0266) / (2 + 30) = 0.0562       # pulled toward global mean, not 50%
```
The `k=30` term is the smoothing. Without it, a brand with 1 impression and 1 click looks like a 100% CTR brand. With it, you need meaningful volume before your own history dominates. **This is the #1 feature in the CTR model.**

`brand_ctr_enc` does the same thing at the brand level. `pub_cvr_enc` / `brand_cvr_enc` are identical but use conversion rate instead of CTR, fitted only on the clicked subset for the CVR model.

---

### `same_category` — publisher and advertiser in the same vertical

**What it is:** 1 if the publisher and the advertiser brand share the same `PRIMARY_CATEGORY`, 0 otherwise.

**Why:** A shopper who just bought running shoes on a sportswear site is more likely to click another sports/apparel brand than a finance app. Category match is a proxy for relevance.

**How:**
```python
# Publisher: Sirena Intimates → category = "Apparel & Accessories"
# Advertiser: CoinOne Shopping → category = "Loyalty & Affiliates"
same_category = (adv_category == pub_category)  →  0   # no match

# Publisher: PetCo → category = "Pet"
# Advertiser: BarkBox → category = "Pet"
same_category = (adv_category == pub_category)  →  1   # match
```
From EDA: same-category impressions had **4.3% CTR vs 2.6% average** — a meaningful lift.

---

### `is_lead_gen` — flag for email-capture widgets

**What it is:** 1 if `WIDGET_TYPE` is `LEAD_GEN` or `SHOPIFY_NATIVE_LEAD_GEN`, 0 otherwise.

**Why:** These widgets are designed for email capture, not click-through. Their CTR is effectively 0% — they're not competing for clicks, they're collecting leads. If the model sees these without a flag, it would confuse low CTR with a bad brand or bad placement.

**How:**
```python
is_lead_gen = WIDGET_TYPE.isin(["LEAD_GEN", "SHOPIFY_NATIVE_LEAD_GEN"])
# → 1 for those rows, 0 for everything else
```

---

### `is_bot` — flag for non-human traffic

**What it is:** 1 if `OS_CLASS` is `Cloud` or `Hacker`, 0 otherwise.

**Why:** Bots click at a different rate than humans (0.85% vs 2.66%). They're only 118 rows out of 471K so filtering them out makes no meaningful difference — but flagging them lets the model learn to discount those rows rather than treating them as noisy humans.

**How:**
```python
is_bot = OS_CLASS.isin(["Cloud", "Hacker"])
```

---

### `is_anonymous` — flag for shoppers with no identity

**What it is:** 1 if `IDENTITY_UUID` is null, 0 otherwise.

**Why:** 0.56% of shoppers have no identity — they're truly anonymous, no cross-session history. After the join with `user_metadata`, these rows get null values for LTV, AOV, etc. We fill those nulls with medians. But the model can't tell the difference between "LTV = median because they're average" and "LTV = median because we don't actually know." This flag makes that distinction explicit.

**How:**
```python
is_anonymous = IDENTITY_UUID.isna().astype(int)
```

---

### `log_ltv` and `log_aov` — log-transformed shopper value

**What it is:** `log(1 + LTV)` and `log(1 + AOV)` instead of raw dollar values.

**Why:** LTV and AOV are right-skewed — most shoppers have modest spend, but a few have LTV in the thousands. A raw value of 5,000 vs 50 creates a huge numeric range that can destabilise tree splits. Log-transform compresses the scale so a $50 vs $100 shopper and a $500 vs $1,000 shopper get treated with the same *relative* difference.

**How:**
```python
# Shopper A: LTV = $50   → log(1 + 50)   = 3.93
# Shopper B: LTV = $500  → log(1 + 500)  = 6.21
# Shopper C: LTV = $5000 → log(1 + 5000) = 8.52
# Without log: C is 100× A. With log: C is only 2.2× A.
log_ltv = np.log1p(LTV)
```

---

### `adv_category_enc` and `pub_category_enc` — brand verticals as numbers

**What it is:** The advertiser's and publisher's `PRIMARY_CATEGORY` encoded as integers (e.g. "Apparel & Accessories" → 0, "Beauty" → 1, ...).

**Why:** Category has strong CTR signal (Loyalty & Affiliates at 3.2% vs Pet at 0.6%). Raw strings can't go into a model — label encoding turns them into ordinal integers. Not perfect (implies ordering that doesn't exist) but tree models split on thresholds so it works fine in practice.

**How:**
```python
le = LabelEncoder()
le.fit(train["adv_category"].fillna("unknown"))
adv_category_enc = le.transform(val["adv_category"].fillna("unknown"))
# "Apparel & Accessories" → 0, "Automotive" → 1, "Baby & Toddler" → 2, ...
# Unseen category in val → -1
```

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

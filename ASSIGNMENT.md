# ML Take-Home: Click & Conversion Prediction

Welcome — this is a self-contained ML modeling assignment.

**Time:** plan on roughly **2 hours of focused work**. We care more about how
you think than how polished the final notebook is.

**Window:** you have **48 hours** from when we send this to return your
submission.

**AI tools:** use them. Cursor, Copilot, ChatGPT, Claude — whatever you'd use
on the job. Just be ready to defend your code and design choices in the
debrief (see below).

## About Disco

Disco is a **two-sided post-purchase ad network**.

- On one side are **publishers**: e-commerce brands (and the apps/order-tracking
  pages they run) that have just sold something to a shopper. After checkout,
  Disco's widget appears in their order-tracking / thank-you / order-status
  flows.
- On the other side are **advertisers**: brands that pay to be discovered by
  those high-intent post-purchase shoppers.
- A **brand** in Disco's network is usually both — a Disco publisher is also a
  Disco advertiser, which is why `BRAND_ID` and `PUBLISHER_UUID` share the same
  metadata table.

When a shopper finishes a purchase on Publisher A, Disco's widget shows them a
small set of recommended brands. The shopper may **click** through to one of
those brands and may later **convert** (sign up / purchase) on that advertiser's
site. Disco's job is picking which brands to show to which shoppers.

## How Disco maps to the data you have

| Concept                                    | Where it lives                                      |
| ------------------------------------------ | --------------------------------------------------- |
| Shopper just finished a purchase           | one row in `events_train.csv` (`PAGE_TYPE` etc.)    |
| Publisher whose page is hosting the widget | `PUBLISHER_UUID`, `PUBLISHER_NAME`                  |
| Brand being recommended in that slot       | `BRAND_ID`, `BRAND_NAME`                            |
| Shopper identity (across sessions)         | `IDENTITY_UUID` — joins to `user_metadata.csv`      |
| Brand context (category, description, ...) | `BRAND_ID` / `PUBLISHER_UUID` → `brand_metadata.csv`|
| Did the shopper click?                     | `CLICKED` (binary)                                  |
| Did the shopper convert at the advertiser? | `CONVERTED` (binary; only meaningful when CLICKED=1)|

A single shopping session can produce many rows — one per brand impression
shown in that session.

## Your tasks

Build **two** models on the data in this folder:

### 1. Click-Through-Rate (CTR) prediction
Given an impression (a row in events) — *publisher context, shopper context,
brand context, widget context* — predict `P(CLICKED = 1)`.

### 2. Click-to-Convert (CVR) prediction
Given that an impression was clicked, predict `P(CONVERTED = 1 | CLICKED = 1)`.

These are the two halves of the funnel Disco optimizes. We don't expect
state-of-the-art numbers — we want to see your modeling judgment.

## What we want to see

1. **Exploration** — quick EDA. What does the data look like? Class balance?
   Anything weird? What signals look promising?
2. **Feature engineering** — show the joins, show the feature design choices.
   Treat brand/user/publisher metadata as first-class — don't just throw IDs
   at a model.
3. **Modeling** — pick a model family (or two) and justify it. We're agnostic
   on framework; sklearn/lightgbm/xgboost/pytorch all fine. A simple model
   you understand beats a complex one you don't.
4. **Evaluation** — pick metrics appropriate to ranking/probability tasks
   (ROC-AUC, PR-AUC, log-loss, calibration, lift-at-K). Be explicit about
   train/validation splitting — random vs. time-based vs. user-based matters
   here, tell us what you chose and why.
5. **Discussion** — what would you do with more time? What's your model good
   at, what's it bad at, what production concerns would you raise?

## Deliverables

Share your work as either a **GitHub repo** or a **Google Drive folder** (make
sure it's accessible to us — link in your reply email).

Include:

1. **Your code** — notebook(s) or scripts, runnable end-to-end against the
   CSVs in this folder.
2. **A short writeup** (a README, doc, or markdown cells — whatever suits you).
   Keep it brief but cover:
   - What you did and why (design choices: features, model family, splitting,
     metrics).
   - Results — numbers + a sentence of interpretation each.
   - What you'd do next with more time, and any production concerns you'd
     raise.
3. **`predictions_holdout.csv`** — predictions on a held-out set we'll share
   separately on submission day, with two added columns: `pred_click` and
   `pred_convert`. *(We'll score these against ground truth on our side.)*

## Debrief

After we review your submission we'll book a **debrief session**. Expect:

- Walking us through your design choices and tradeoffs.
- Drill-down questions on anything in the writeup or code.
- A short **live-coding segment with new data** (you can use AI tools here
  too) — typically extending what you built or exploring a new angle on a
  fresh slice of events.

## Data files in this folder

- `events_train.csv` — impression-level events with `CLICKED` and `CONVERTED`
  labels. **This is your training data.**
- `brand_metadata.csv` — one row per brand/publisher with category, sub-categories,
  and a short description.
- `user_metadata.csv` — one row per shopper (joined on `IDENTITY_UUID`) with
  lifetime-value / order-history features.
- `GLOSSARY.md` — column-by-column definitions and Disco vocabulary.

A separate **holdout** events file (same schema, labels withheld) will be
provided when you're ready to submit predictions.

## Ground rules

- Any library or pretrained model is fair game.
- If something looks ambiguous in the data, make a reasonable assumption,
  document it, and keep moving. (You can also email us; we'll respond fast.)

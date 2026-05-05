# Glossary

A reference for Disco terminology and the columns in each table. Skim this
before EDA — some columns are non-obvious.

## Disco vocabulary

- **Publisher** — an e-commerce brand whose order-tracking / post-purchase
  pages host Disco's widget. After a shopper completes a purchase on the
  publisher, the widget shows recommended advertisers.
- **Advertiser** — a brand that pays to appear inside Disco's widget on
  publisher pages. The goal of an advertiser impression is a click and,
  ideally, a conversion on the advertiser's site.
- **Brand** — a single entity in Disco's network. Most brands are *both* a
  publisher (they host the widget on their order pages) and an advertiser
  (they buy placements on other publishers). That's why `BRAND_ID` and
  `PUBLISHER_UUID` both point at rows in `brand_metadata.csv`.
- **Impression** — a single (shopper, brand) pair shown inside the widget on
  a publisher page. One row in `events_train.csv` = one impression.
- **Session** — one visit by one shopper to one publisher's post-purchase
  page. A session typically yields multiple impressions (the widget shows
  several brands).
- **Click** — the shopper clicked through from the widget to the advertiser's
  site. Captured by `CLICKED`.
- **Conversion** — the shopper performed the advertiser's tracked action
  (account creation / purchase / lead form, depending on advertiser) after
  clicking. Captured by `CONVERTED`. Only meaningful when `CLICKED = 1`.
- **Post-purchase / two-sided marketplace** — Disco's defining shape: every
  impression happens *after* a known purchase, and the same brand pool plays
  both buyer and seller of attention.

## Funnel & metrics shorthand

- **CTR** (Click-Through Rate) — `P(CLICKED = 1)` over all impressions.
- **CVR** (Conversion Rate) — `P(CONVERTED = 1 | CLICKED = 1)` over clicks.
- **eCVR** (effective CVR) — `P(CONVERTED = 1)` over all impressions =
  CTR × CVR. Useful as a single ranking score.

---

## `events_train.csv` / `events_holdout.csv`

One row per impression.

| Column                    | Description |
| ------------------------- | ----------- |
| `SESSION_ID`              | UUID for the shopper's session on a publisher. Multiple impressions share a session id. |
| `SESSION_START_AT_UTC`    | Timestamp the session began (UTC). Useful for time-based splits. |
| `IDENTITY_UUID`           | Stable shopper identifier across sessions. Joins to `user_metadata.csv`. |
| `PUBLISHER_NAME`          | Display name of the publisher hosting the widget. |
| `PUBLISHER_UUID`          | Stable publisher identifier. Joins to `brand_metadata.csv` (`PUBLISHER_UUID`). |
| `BRAND_NAME`              | Display name of the advertiser shown in this impression. |
| `BRAND_ID`                | Advertiser identifier. Joins to `brand_metadata.csv` (`BRAND_ID`). |
| `PAGE_TYPE`               | The publisher page the impression was rendered on. See *Page types* below. |
| `WIDGET_TYPE`             | Which Disco surface delivered the impression. See *Widget types*. |
| `WIDGET_VERSION`          | Numeric version of the widget; older versions can have different behavior. |
| `OS_CLASS`                | Coarse device class: `Mobile`, `Desktop`, `Unknown`, `Cloud`, `Hacker`. The last two are likely bots/scrapers. |
| `BRAND_DISPLAY_PLACEMENT` | Where on the page the brand was shown. See *Display placements*. |
| `CUSTOM_METADATA`         | JSON blob of publisher-supplied context (shipping method, taxonomy tags, transaction type, etc.). Sparse and noisy — usable but optional. |
| `CLICKED`                 | `1` if the shopper clicked the impression. Label for the CTR model. |
| `CONVERTED`               | `1` if the click resulted in a tracked conversion at the advertiser. Label for the CVR model. |

### Page types

- **`THANK_YOU`** — the order confirmation page rendered immediately after
  checkout. Highest-intent moment.
- **`ORDER_STATUS`** — a static page showing current order state (often the
  page Shopify shows when a shopper revisits their order link).
- **`ORDER_TRACKING`** — the live shipment-tracking experience the shopper
  returns to multiple times between purchase and delivery. The largest
  single bucket of impressions.
- **`SUPPORT_CENTER`** — help / FAQ pages. Rare; lower intent.

### Widget types

- **`APP_NATIVE_ESSENTIAL`** — Disco's main offering inside a publisher's
  native mobile/web order-tracking app (e.g. an order-tracking platform's
  app embeds Disco directly).
- **`SHOPIFY_NATIVE_ESSENTIAL`** — the equivalent surface inside Shopify's
  native order-status / thank-you pages.
- **`ESSENTIAL`** — the generic web widget served on publisher pages that
  aren't part of an app integration.
- **`APP_NATIVE_BRAND_VISUAL`** — a richer brand-forward placement (more
  imagery, fewer brands per impression).
- **`LEAD_GEN` / `SHOPIFY_NATIVE_LEAD_GEN`** — placements optimized for
  email/lead capture rather than direct click-through. Conversion semantics
  may differ slightly here.

### Display placements

- **`INLINE`** — the widget renders within the natural flow of the page.
- **`PULLUP`** — a bottom-sheet / drawer that slides up from the page.
- **`FULLSCREEN`** — a modal / takeover surface.

Different placements have very different baseline CTRs — be careful not to
let placement leak the answer.

---

## `brand_metadata.csv`

One row per brand. Same row describes a brand whether it's acting as a
publisher or as an advertiser.

| Column             | Description |
| ------------------ | ----------- |
| `BRAND_ID`         | Brand identifier (joins to events on `BRAND_ID`). |
| `PUBLISHER_UUID`   | Publisher identifier for the same brand (joins to events on `PUBLISHER_UUID`). |
| `BRAND_NAME`       | Display name. |
| `PRIMARY_CATEGORY` | One of 19 top-level categories (Apparel & Accessories, Beauty & Personal Care, Food & Beverage, ...). |
| `SUBCATEGORIES`    | JSON-encoded list of finer-grained tags. Multi-valued. |
| `DESCRIPTION`      | Short marketing blurb. Useful for text features / embedding. |

A small number of brand_ids and publisher_uuids appear in events but not in
this table — handle that gracefully (impute / unknown bucket).

---

## `user_metadata.csv`

One row per shopper, joined on `IDENTITY_UUID`.

| Column                | Description |
| --------------------- | ----------- |
| `IDENTITY_UUID`       | Shopper id (joins to events). |
| `LTV`                 | Lifetime value across the publishers Disco has seen this shopper on (USD). |
| `AOV`                 | Average order value (USD). |
| `NUM_PURCHASES`       | Count of purchases observed for this shopper. |
| `BOUGHT_SUBSCRIPTION` | `1` if the shopper has ever purchased a subscription product. |
| `DISCOUNT_SHOPPER`    | `1` if the shopper tends to use discount/promo codes (heuristic). |
| `REFUNDED`            | `1` if at least one of their orders was refunded. |

These are aggregated *up to the impression's session*, so they're safe to use
as features (no time-leakage). Some shoppers in the events tables won't have
a row here — handle missingness.

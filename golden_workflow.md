## Data sanity checks

**Rationale:** Yield data has calendar gaps (weekends/holidays) and occasional missing maturities. Small preprocessing mistakes can create false predictability.

**Dataset and objective nuances:**

- Verify `test.csv` has **no** `target_bear_steepen_shock_10d`.
- Check base rates by `event_era` and by `slice_high_rate_vol`. The event should be rarer in calm regimes and more common when short-rate volatility is high.
- Inspect `missing_count`. Missingness can correlate with specific historical periods and can distort calibration if ignored.

## Validation strategy

**Rationale:** This is explicitly forward-looking and the split is time-based. Random splits inflate performance due to regime persistence.

**Dataset and objective nuances:**

- Prefer walk-forward CV or an era holdout:
  - train on earlier `event_era` buckets
  - validate on the newest era available in training
- Track the exact leaderboard components:
  - LogLoss (all)
  - AUPRC (all)
  - LogLoss on `slice_high_rate_vol`

## Preprocessing plan

**Rationale:** Features are already binned. The key is learning interactions and calibrating probabilities under shift.

**Dataset and objective nuances:**

- Treat `*_bin` columns as ordinal:
  - tree models can use them directly
  - linear models should add a small set of interactions (e.g., spread × momentum × vol)
- Use `missing_count` explicitly, and allow interactions with `event_era` (measurement regimes differ across decades).

## Baseline model (domain-justified)

**Rationale:** Bear-steepening shocks tend to occur when:

- the front end is moving (policy repricing), and
- the curve has room to steepen (interaction of level and slope), often under volatility.

**Dataset and objective nuances:**

- Start with a regularized gradient-boosted tree model:
  - allow interactions like `s_10y2y_bin × 2_yr_chg5_bin × 2_yr_std20_bin`
  - include `event_era` interactions (policy regimes differ)
- Apply post-hoc calibration:
  - the metric is LogLoss-heavy and punishes overconfidence
  - calibrate on an era-aware validation set

## Iteration plan

**Rationale:** Improve the composite score by addressing each term intentionally.

**Dataset and objective nuances:**

- Improve LogLoss first:
  - regularization / early stopping
  - post-hoc calibration
- Improve AUPRC next:
  - modestly increase capacity near the decision boundary
  - add a small set of interaction features if needed
- Finally focus on high-vol slice:
  - models often under-estimate tail risk during fast-moving front-end regimes

Calibration recipe:

- Fit your model on the training fold.
- Predict on the validation fold.
- Fit temperature scaling or Platt scaling on validation predictions.
- Apply calibration to test predictions and clip to \([1e-6, 1-1e-6]\).

## Error analysis

**Rationale:** Confident wrong predictions dominate LogLoss.

**Dataset and objective nuances:**

- Slice errors by:
  - `event_era`
  - `segment_id` (different curve segments behave differently across regimes)
  - `spread_bin` (current segment curve state)
  - `short_chg5_bin` (short-maturity momentum)
  - `short_std20_bin` (short-maturity volatility)
  - `slice_high_rate_vol`
- If errors cluster in one era, prefer era-conditional calibration over adding model size.

## Leaderboard safety

**Rationale:** The test set emphasizes newer regimes. A model that “memorizes” older patterns will fail out-of-sample.

**Dataset and objective nuances:**

- Do not use row-wise random CV.
- Prefer simpler models + calibration over large ensembles that overfit regime blocks.

## Submission checks

**Rationale:** Formatting errors are easy to avoid.

**Dataset and objective nuances:**

- Ensure `submission.csv` has exactly `row_id` and `pred_bear_steepen_shock_10d`.
- Ensure predictions are in \([0,1]\), contain no NaNs, and ids match `test.csv`.
- Run:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`


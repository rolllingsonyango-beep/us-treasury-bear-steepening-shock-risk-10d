## Objective

Predict whether the U.S. Treasury curve will experience a **bear-steepening shock** over the next 10 trading days.

Each row represents a **segment-day** at time \(t\): one trading day paired with a specific yield-curve segment (e.g., 10Y–2Y). Features describe the current curve state, momentum, and volatility for that segment. Define:

- For a segment with a “long” and “short” maturity, \(s(t)=y_\text{long}(t)-y_\text{short}(t)\)
- \(\Delta s_{10}(t) = s(t+10) - s(t)\)
- \(\Delta y_\text{short,10}(t) = y_\text{short}(t+10) - y_\text{short}(t)\)

The label is:

\[
\texttt{target\_bear\_steepen\_shock\_10d}=\mathbb{1}\left[\Delta s_{10}(t)\ge \tau_s(\text{segment}) \ \wedge\ \Delta y_{\text{short},10}(t)\ge \tau_{y,\text{short}}(\text{segment})\right]
\]

Where \(\tau_s(\text{segment})\) is the **training-row 85th percentile** of \(\Delta s_{10}\) for that segment, and \(\tau_{y,\text{short}}(\text{segment})\) is the **training-row 75th percentile** of \(\Delta y_{\text{short},10}\) for that segment. This makes the task about forecasting unusually large **steepening with rising short rates**, not just any spread move.

## Inputs

- **`train.csv`**: includes `row_id`, feature columns, and `target_bear_steepen_shock_10d`
- **`test.csv`**: includes `row_id` and the same feature columns, but **no target**

Anti-leakage design:

- Exact dates are not provided (only coarse calendar fields).
- Yield/spread/volatility statistics are released as **bins learned from training rows only**.
- The segment identity is provided as an anonymized `segment_id` (categorical).

## Output

Create a file named **`submission.csv`** with predicted probabilities for each row in `test.csv`.

## Submission format

Your `submission.csv` must contain exactly these columns:

- `row_id`
- `pred_bear_steepen_shock_10d`: probability that `target_bear_steepen_shock_10d = 1`

Predictions must be numeric and in **[0, 1]**.

## Metric (lower is better)

We use a deterministic composite metric:

\[
\text{Score} = 0.50\cdot \text{LogLoss}_{\text{all}}
             + 0.30\cdot \text{LogLoss}_{\text{high-vol slice}}
             + 0.20\cdot (1 - \text{AUPRC}_{\text{all}})
\]

- The **high-vol slice** emphasizes periods where short-rate volatility is high.
- Slice membership is provided as `slice_high_rate_vol` in both `train.csv` and `test.csv`.

Slice reproducibility (from visible features):

- With `n_bins=12`, `slice_high_rate_vol = 1` when `short_std20_bin >= 9` (top 3 volatility bins).

Deterministic scoring command:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`

## Constraints and leakage notes

- The train/test split is **time-based under the hood** (newer years emphasized in test), but exact dates are hidden.
- For honest offline evaluation, avoid random splits. Prefer an `event_era` holdout and walk-forward validation.


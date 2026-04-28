## Overview

This dataset is derived from the U.S. Treasury’s **Daily Treasury Par Yield Curve Rates**. It has been transformed into a coarsened forecasting task: given a **segment-day** snapshot of the curve (a day paired with a specific curve segment), predict whether that segment will experience a **bear-steepening shock** over the **next 10 trading days** (large steepening while the segment’s short maturity yield rises).

The task is designed to be challenging because it combines:

- **rare-event forecasting** (large bear-steepening moves are episodic),
- **regime shift** (policy eras differ materially),
- **nonlinear curve dynamics** (levels, slopes, curvature, momentum, volatility),
- and **strict anti-leakage constraints** (no exact dates; binned features learned on training only).

## Source

- **Upstream source**: U.S. Department of the Treasury – Daily Treasury Rate Archives
- **Archive page**: `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives`
- **CSV URL**: `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/par-yield-curve-rates-1990-2023.csv`

## License

Public-Domain-US-Gov

Rationale: U.S. Treasury datasets are U.S. government works.

## Features

All yield-related values are converted to **bins learned from training rows only**. Missing values are handled explicitly via `missing_count` and `-1` bin codes.

Core identifiers / time context:

- `row_id`: unique row identifier (integer)
- `segment_id`: anonymized segment identifier (categorical; each row corresponds to one segment on one day)
- `event_era`: coarse time bucket derived from the hidden year (older → newer)
- `month`: month of year (1–12)
- `dow`: day of week (0=Mon … 6=Sun)
- `missing_count`: number of missing raw curve/dynamics fields used for feature construction
- `slice_high_rate_vol`: 1 if short-maturity volatility is high (defined from `short_std20_bin >= 9`), else 0

Segment-level bins:

- `short_yield_bin`, `long_yield_bin`, `spread_bin`
- `short_chg1_bin`, `short_chg5_bin`, `long_chg1_bin`, `long_chg5_bin`
- `spread_chg1_bin`, `spread_chg5_bin`
- `short_std20_bin`, `spread_std20_bin`

Label (train/solution only):

- `target_bear_steepen_shock_10d`: 1 if the next-10-trading-day steepening and 2Y rise exceed training-derived thresholds, else 0
  - thresholds are computed **per segment** from training rows only

## Splitting & Leakage

- **Split strategy**: time-based regime shift. Older years are used for training; newer years are emphasized in test. A deterministic portion of bridge years is mixed into test.
- **Leakage mitigations**:
  - exact dates are excluded,
  - yields/spreads/dynamics are binned using training-derived quantile edges,
  - the label uses only future yields/spreads (not present in features) by construction.


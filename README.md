## U.S. Treasury Bear-Steepening Shock Risk (Next 10 Days)

Portfolio-ready, Kaggle-style prediction task built from **U.S. Treasury yield curve** time series.

### What you’re predicting
- **Task**: binary classification
- **Predict**: `pred_bear_steepen_shock_10d`
- **Target**: `target_bear_steepen_shock_10d` — a tail **bear-steepening** shock occurs over the next 10 trading days

### Data & evaluation highlights
- **Rows**: train **26,771**, test **6,200**
- **Split**: time-based regime shift (see `dataset_card.md`)
- **Metric**: composite scorer (LogLoss overall + slice LogLoss + (1 − AUPRC)); see `instruction.md`

### Repository contents
- `train.csv`, `test.csv`, `solution.csv`
- `sample_submission.csv`, `perfect_submission.csv`
- `build_dataset.py`, `build_meta.json`
- `score_submission.py`
- `instruction.md`, `golden_workflow.md`, `dataset_card.md`

### Quickstart

```bash
python build_dataset.py
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```


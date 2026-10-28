# 15 — Reproducibility & Evidence

## What is reproducible today `[New]`

| Stage | How | Determinism |
|---|---|---|
| Split | `src\dataset.py` reads `dataset\splits\{train,val}.csv`; regenerates w/ seed 42 after MD5 gate if missing | Identical every run (manifests persisted) |
| Train | `python src/train.py --model X --channel Y [--mode binary]` → `checkpoints\…`, `logs\training\…` | No fixed random seed set in-code — epochs differ per run (documented) |
| Evaluate | `python src/test.py --model X --channel Y [--mode binary]` → summary CSVs | Deterministic given the checkpoint |
| Verify auth | `web\app.py` `/api/identify|verify|mfa` against real checkpoints | Scores stable; cache-tolerant |

Evidence files that *pin* the story (all exist, all unmodified):
- Split manifests `dataset\splits\train.csv` (1,413 rows) / `val.csv` (606 rows).
- Result tables: the single consolidated `results_summary.csv` + `results_binary_summary.csv`
  (42 rows) + threshold tables (doc 10).
- 68 `.pt` checkpoints under `checkpoints\` (doc 11).
- 68 training-curve CSVs under `logs\training\` (doc 11).
- The leakage gate result (doc 05) and split-verification files from the old project
  (`dataset_v1\split_verification.txt`, `dataset\split_verification_v2.txt`).

## Honesty notes (do not claim beyond the evidence)
- No global training seed is fixed in `src\`; checkpoints/logs pin each run individually,
  not the sampler. If a reviewer reruns training, epoch-to-epoch numbers will differ slightly.
  `[New]`
- The old project (doc 06) is reconstructible only at the *results* level (CSV/JSON);
  its checkpoints are not mounted, and some grouped splits lived on external paths. `[Old]`
- Analysis scripts used during the write phase (correlation, per-scanner aggregation) were
  discarded after production of these docs; the only lemming leftovers are their outputs in
  the results CSVs. Anything not named in this doc or doc 11 was tooling, not evidence. `[New]`

## Verification protocol used for the numbers quoted in docs 01–17
1. Every number in docs 06–10 was re-derived from the named file (e.g. 5-fold macro averages
   recomputed from `binary_5fold\five_fold_summary.json`; per-scanner Dataset-1 counts
   re-scanned from `dataset\Dataset1\Fingerprint\`; manifest sizes counted from the CSVs).
2. The user-facing headline table was checked cell-by-cell against the single consolidated
   `results_summary.csv` (exact match on the shared rows; doc 10).
3. Conflicting legacy claims are labelled and never silently resolved.

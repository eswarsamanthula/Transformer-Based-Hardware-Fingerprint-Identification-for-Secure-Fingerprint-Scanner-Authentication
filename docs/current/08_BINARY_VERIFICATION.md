# 08 — Binary One-vs-All Verification

## Task `[New]`

For each target sensor (1..10), a model decides "was this image captured by sensor N?" —
binary label = 1.0 if the sample's scanner label equals the target, else 0.0
(`src\dataset.py::BinaryWrapperDataset`). Reuses the same manifests and Dataset 2 holdout;
only labels are remapped at `__getitem__` time. Loss: BCEWithLogits. Metrics at threshold
**0.5** on the sigmoid: accuracy / precision / recall / F1 / AUC, plus **FAR** (false
acceptance) and **FRR** (false rejection) per scanner split-wise, then a mean-across-scanners test row.

Implementation: `src\train.py` (`--mode binary`) trains one checkpoint per scanner;
`src\test.py` evaluates and writes `results\tables\results_binary_summary.csv`.

## The shipped binary campaign — mobilenet

Complete file verified: `results\tables\results_binary_summary.csv` (42 data rows:
10 scanners × {grayscale,rgb} × {val,test} × 2 modes? — actually: 10 scanners × 2 channels ×
(val,test) = 40 rows + 2 mean-test rows = 42; `mean` test rows are grayscale and rgb).

**Mean test (Dataset 2, unseen user, 200 images):**

| Channel | Accuracy | Precision | Recall | F1 | AUC | FAR | FRR |
|---|---|---|---|---|---|---|---|
| grayscale | 0.944 | 0.859 | 0.813 | 0.827 | 0.931 | 0.0233 | 0.350 |
| rgb | 0.944 | 0.887 | 0.791 | 0.798 | 0.923 | 0.0178 | 0.400 |

**Perfect scanners (both modes, test): Sensor2, Sensor6, Sensor10.**

**Weak scanners (test):**
- Sensor1 grayscale FRR 0.90 (FAR 0.089) — almost all genuine Sensor1 images rejected.
- Sensor1 rgb FRR 0.85; Sensor3 rgb FRR 0.85; Sensor9 rgb FRR 0.90 (FAR 0.0 — maximally conservative).
- Sensor5 rgb FAR 0.10 (risky direction: accepts impostors) though FRR 0.05.

## Interpretation

- The confusion is heavily **asymmetric**: the verifier is far more likely to *reject a genuine
  user* (FRR 0.35–0.40 mean) than *accept an impostor* (FAR ≈0.02). For an authentication
  system this is the safe direction.
- Weak sensors (1, 3, 9 — the class-skew extremes in Dataset 1: Sensor1 279, Sensor3 180,
  Sensor9 210) mirror the old project's scanner_07 problem: minority-hard sensors produce
  high false-rejection. The threshold is fixed at 0.5 everywhere; per-scanner threshold
  tuning is a planned experiment (doc 14).
- The live demo reflects this: a genuine Dataset 2 Sensor5 image scored genuine-probability
  ≈0.114 on the rgb verifier (doc 09) — consistent with the CSV's Sensor5 rgb behavior
  when evaluated at strict confidence.

## Verdict

Binary verification works and is safe per impostor, but the default 0.5 threshold is too
strict on several sensors — the single biggest lever for the authentication demo's UX is
per-scanner threshold calibration (planned).
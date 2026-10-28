# 17 — Madam Project Status

> Status summary. Legend: `[DONE]` implemented & evidenced; `[OLD]` historical evidence only;
> `[PARTIAL]` partially done; `[PLANNED]` not started; `[IDEA]` proposed.

## Per-area status

| Area | Status | Where |
|---|---|---|
| Base-paper protocol re-implemented on raw data | [DONE] `[New]` | `src\` + doc 02, 07 |
| Dataset 1 datasetFoundation (2019 imgs, folder = sensor) | [DONE] `[New]` | doc 03 |
| **Dataset 2 as final unseen-user test (200 imgs)** | [DONE] `[New]` | doc 03 |
| Seed-42 per-scanner 70/30 split manifests (1413/606) | [DONE] `[New]` | doc 04 |
| MD5 leakage gate + pixel-level investigation (0 matches) | [DONE] `[New]` | doc 05 |
| 6 paper CNNs trained & evaluated (val + unseen-user test) | [DONE] `[New]` | doc 07, 10 |
| 4 primary transformers — Colab-era run (values in the official file) | [DONE] `[New]` | doc 07, 10 |
| Earlier backbones (vit_b16/swin/deit/convnext) — consolidated into the main tree | [DONE] `[New]` | doc 07, 10, 11 |
| ConvNeXt grayscale collapse analysis | [DONE] `[New]` (results), fix [PLANNED] | doc 07, 14 |
| Binary one-vs-all verification w/ FAR/FRR (mobilenet) | [DONE] `[New]` | doc 08 |
| Binary one-vs-all verification w/ FAR/FRR (vgg16 + efficientformer_l1) | [DONE] `[New]` | doc 18, 19 |
| Per-scanner threshold calibration (validation-frozen) | [DONE] `[New]` | doc 19, `results/tables/binary_threshold_calibration.csv` |
| Final metrics packaging + graphs (`results/metrics`, `results/graphs`) | [DONE] `[New]` | doc 19 |
| `/results` final results page (reads metrics/tables/graphs) | [DONE] `[New]` | `web/templates/results.html`, doc 19 |
| HMAC-SHA256 MFA + live inference lab (merged `/auth` page) | [DONE, verified] `[New]` | doc 09 |
| Old cross-scanner 5-fold binary study (97.79% macro acc) | [DONE — OLD only] | doc 06 |
| Old project checkpoints / full reproducibility | [OLD / PARTIAL] | doc 06, 15, 16 |
| Seed averaging (error bars) | [PLANNED] | doc 14 n°4 |
| Unseen-hardware (cross-sensor) test | [PLANNED] | doc 14 n°6 |
| Spoofing / liveness evidence | [IDEA] | doc 14 n°7 |
| Latency/FLOPs benchmark (base: old `edge\benchmark.py`) | [PLANNED] | doc 14 n°8 |
| Consolidated checkpoint trees / test.py fallback | [PLANNED] | doc 14 n°11 |
| Research-paper manuscript (plan exists) | [PLANNED] | doc 13 |

## Documented discrepancy to always re-state
The widely-quoted table (incl. pit_ti grayscale 0.20) is exactly what the single
consolidated `results_summary.csv` holds. A later local rerun reported different transformer
values (pit_ti g 0.845, efficientformer_l1 g 0.91) but those rows are not part of the
official tree — quote the one file (doc 10).

## Open questions (tracked, none blocking)
- Where the old-era `.pt` files physically are (`[Not verified]`).
- Whether colab-era training used a fixed seed (`[Not verified]`).
- Docs layout: **`THFSA_v2\docs\current\`** (19 live docs, index = `00_INDEX.md`) +
  **`docs\archive\`** (5 historical docs: 01, 06, 16, 21, 23).

## Bottom line
The rebuild is complete and reproducible at the split/eval level; the raw-data experiments,
binary verification, and a **working HMAC-MFA demo** are all live and verified. The remaining
work is experimental depth (thresholds, seeds, unseen-hardware, latency) and the write-up —
no blocking issues.

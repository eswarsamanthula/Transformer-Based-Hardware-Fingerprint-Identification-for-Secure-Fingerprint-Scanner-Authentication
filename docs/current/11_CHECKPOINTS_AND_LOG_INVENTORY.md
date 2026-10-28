# 11 — Checkpoint & Log Inventory

## Current consolidated state `[Current]`

`checkpoints\` holds every trained artifact in a single tree. The earlier
fragmented layout (`transformer_colab\` and `new_result\` trees) no longer
exists; their checkpoints were consolidated under the main tree.

| Category | Count | Contents |
|---|---|---|
| `checkpoints\cnn\` | 42 `.pt` | 12 multiclass (6 CNNs × {grayscale,rgb}) + 30 binary one-vs-all (mobilenet grayscale + rgb × scanners 1–10, vgg16 grayscale × scanners 1–10) |
| `checkpoints\transformer\` | 26 `.pt` | 16 multiclass (8 backbones — tinyvit_5m, mobilevit_xs, efficientformer_l1, pit_ti, swin_tiny, deit_small, convnext_tiny, vit_b16 — × {grayscale,rgb}) + 10 binary one-vs-all (efficientformer_l1 rgb × scanners 1–10) |
| **Total** | **68 `.pt`** | 28 multiclass + 40 binary |

`logs\training\` holds **68 per-epoch CSVs** (`epoch,train_loss,train_acc,val_loss,val_acc`),
one per trained run, stem-matching the corresponding checkpoint.

## Repro mapping — which numbers come from which checkpoint set

| Result file | Checkpoint source |
|---|---|
| `results\tables\results_summary.csv` (multiclass, val + Dataset 2 test) | `checkpoints\cnn\` + `checkpoints\transformer\` |
| `results\tables\results_binary_summary.csv` (binary verifiers, Dataset 2 at frozen thresholds) | vgg16 grayscale + efficientformer_l1 rgb + mobilenet binary sets |
| `results\tables\binary_threshold_calibration.csv`, `binary_threshold_test_results.csv` | same binary checkpoints, val-fitted frozen thresholds |

## Provenance notes

- swin_tiny / deit_small / convnext_tiny / vit_b16 live in the main
  `checkpoints\transformer\` tree, so `src\test.py` evaluates them locally (no SKIPs).
- `logs\training\swin_tiny_rgb.csv` is **empty (header only)** although
  `swin_tiny_rgb_best.pt` exists — the swin RGB official number (0.765) comes from
  doc 07's corrected run, not from a populated training log.
- `convnext_tiny` grayscale is a genuine, loadable checkpoint that collapsed to one
  class (Dataset 2 test acc 0.10) — kept intentionally as the honest worst case.
- Training CSVs carry no LR column, so the run behind the swin grayscale 0.805 row
  cannot be proven from logs alone (see `results/audit/EXPERIMENT_AUDIT.md` §8).

## Regenerating anything

Every CSV above can be regenerated from the manifests with `src\{train,test}.py`
(no data was modified; models stay frozen). Reproducibility ratings in doc 15.
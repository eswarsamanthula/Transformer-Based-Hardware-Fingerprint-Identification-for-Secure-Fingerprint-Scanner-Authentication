# 10 — Results Summary & Leaderboard (how to cite)

## One official file, one rule

There is exactly **one** multiclass result file: `results\tables\results_summary.csv`
(56 rows = 14 models × {grayscale,rgb} × {val, test}). It is the merged, consolidated
table: the earlier separate files (`results_summary_colab.csv` and the `new_result\`
tree) no longer exist and their rows were folded into this single file. The dashboard
(`web\app.py` → `/api/multiclass`) reads this one file.

**Rule for any writeup: quote `results\tables\results_summary.csv`, test split,
Dataset 2 (unseen user), and say which split you used.**

## Combined multiclass leaderboards (test accuracy, Dataset 2 unseen user)

Values verbatim from `results_summary.csv` (`split=test`). Cells highlighted are the
canonical numbers to cite.

### grayscale
| Rank | Model | Acc |
|---|---|---|
| 1 | vgg16 | **0.945** |
| 2 | efficientnet_b3 | 0.89 |
| 3 | efficientformer_l1 | 0.83 |
| 4 | mobilevit_xs | 0.82 |
| 4 | resnet50 | 0.82 |
| 6 | swin_tiny (LR=0.0001) | **0.805** |
| 6 | vit_b16 | 0.805 |
| 8 | resnet152 | 0.75 |
| 8 | mobilenet | 0.75 |
| 10 | deit_small | 0.73 |
| 11 | tinyvit_5m | 0.72 |
| 12 | lenet5 | 0.58 |
| 13 | pit_ti | 0.20 (collapse, kept as-is) |
| 14 | convnext_tiny | **0.10** (collapse, kept as-is) |

### rgb
| Model | Acc |
|---|---|
| efficientformer_l1 | **0.88** (best transformer) |
| convnext_tiny | 0.855 |
| vgg16 | 0.845 |
| mobilevit_xs | 0.81 |
| mobilenet | 0.785 |
| resnet50 | 0.785 |
| resnet152 | 0.74 |
| tinyvit_5m | 0.69 |
| pit_ti | 0.675 |
| deit_small | 0.61 |
| vit_b16 | 0.61 |
| lenet5 | 0.435 |
| swin_tiny (LR=0.0001) | **0.765** |

## Integrity notes

- ConvNeXt-Tiny grayscale (0.10) and Pit-Ti grayscale (0.20) are genuine, loadable
  checkpoints that collapsed to one/two classes; they stay in the table as the honest
  worst cases and are never re-trained.
- Swin-Tiny rows are the corrected LR=0.0001 controlled runs (g 0.805 / rgb 0.765);
  their training logs carry no LR column (see doc 11 provenance).
- No row in this file was regenerated post-cleanup; the CSVs are the frozen,
  authoritative measured results.

## Binary summary

`results_binary_summary.csv`: the frozen one-vs-all binary verifiers evaluated on
Dataset 2 at val-calibrated thresholds. Lead rows: vgg16 grayscale acc 0.9395
(FAR 0.00 %, FRR 60.50 %), efficientformer_l1 rgb acc 0.9275 (FAR 1.22 %, FRR 61.50 %).
Full per-scanner rows + threshold calibration in docs 08 and 19.

## Rules for any future writeup

1. Quote the single file `results\tables\results_summary.csv`; name the split (val = Dataset 1, test = Dataset 2).
2. Give the source filename any time a number is quoted.
3. Keep [Old] 5-fold numbers (doc 06) in a separate, clearly-labelled section — different experiment, not comparable to the Dataset-2 test scores.
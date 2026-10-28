# 07 — New Raw-Data Experiments (project `THFSA_v2`)

## Rebuild rationale `[New]`

Per the project README: *"Rebuild after data loss. Only the R307 dataset survived."*
The old processed-artifact era was abandoned; the project was rebuilt to reproduce the
paper's protocol **on raw data only**: `dataset\Dataset1\Fingerprint\Sensor1..10` = train/val,
`dataset\Dataset2\Fingerprint_of_Test1_and_Test2` = final unseen-user test.

## Repository architecture (final rebuild) `[New]`

```
src/                5 files — config, dataset, models, train, test
dataset/            Dataset1 (2,019), Dataset2 (200), splits/ (manifests)
checkpoints/        cnn/ (42 .pt), transformer/ (26 .pt) — a single consolidated tree
logs/training/      68 per-epoch CSVs
results/tables/     results_summary.csv + results_binary_summary.csv +
                    binary_threshold_calibration.csv + binary_threshold_test_results.csv
web/                app.py + index/archive/results/auth templates + static
authentication/     HMAC-MFA modules (doc 09)
```

(An earlier fragmented layout — `transformer_colab\`, `new_result\`, `results_summary_colab.csv` —
no longer exists; all checkpoints/results were consolidated into the single tree above.)

## Model registry (config.py) `[New]`

| Group | Models | Notes |
|---|---|---|
| Paper CNNs | efficientnet_b3, mobilenet, resnet50, resnet152, vgg16, lenet5 | effnet/vgg16 use paper custom head; vgg16+lenet5 lr 0.0001; effnet 51ep, vgg 54ep, lenet 30ep, mobilenet/resnet50 40ep, resnet152 60ep |
| **Primary transformers** (new additions) | tinyvit_5m, mobilevit_xs, efficientformer_l1, pit_ti | bs 32, lr 0.001, max 60 epochs |
| Earlier backbones (comparison) | vit_b16 (bs8/70ep), swin_tiny, deit_small, convnext_tiny (bs32/50ep) | kept so results stay reproducible |

All `pretrained=False` (from scratch, per paper). Input sizes: effnet 300, mobilevit 256,
lenet 32, everything else 224. Binary = one-vs-all BCEWithLogits per target scanner
(doc 08).

Note (2026-09-15): the MobileViT-XS **grayscale** run was retrained with lr 3e-4 / 41 epochs
(log `logs\training\mobilevit_xs_grayscale.csv`); Dataset-2 test acc 0.82, val acc 0.9620.
The RGB row is unchanged.

## Training/Evaluation loop (train.py / test.py) `[New]`

- Adam, per-model lr; CE (multiclass) / BCE (binary); early stopping on val loss, patience 10,
  best weights restored; checkpoints saved as `<model>_<channel>[%_binary_scanner<n>]_best.pt`;
  per-epoch CSVs under `logs\training\`.
- test.py: macro P/R/F1, AUC (ovr macro; binary per-signal), **FAR/FRR at threshold 0.5**
  (binary), per-scanner mean row, writes the summary CSVs. Models whose checkpoint is missing
  are skipped with a `SKIP` line (doc 11 notes which live where).

## Result file — the single consolidated summary CSV `[New]`

Column layout: `model,channel,split(val|test),accuracy,precision,recall,f1,auc`.
Viewing note: test = Dataset 2 (unseen user, 200 images). Values below are **test** accuracies,
read verbatim from `results\tables\results_summary.csv` (56 rows = 14 models × {g,rgb} ×
{val,test}). The Colab-era and `new_result\` rows were folded into this one file.

| Model | grayscale | rgb |
|---|---|---|
| vgg16 | **0.945** | 0.845 |
| efficientnet_b3 | 0.89 | 0.825 |
| efficientformer_l1 | 0.83 | **0.88** |
| mobilevit_xs | 0.82 | 0.81 |
| resnet50 | 0.82 | 0.785 |
| swin_tiny (LR=0.0001) | **0.805** | **0.765** |
| vit_b16 | 0.805 | 0.61 |
| mobilenet | 0.75 | 0.785 |
| resnet152 | 0.75 | 0.74 |
| deit_small | 0.73 | 0.61 |
| tinyvit_5m | 0.72 | 0.69 |
| lenet5 | 0.58 | 0.435 |
| pit_ti | 0.20 (collapse) | 0.675 |
| convnext_tiny | 0.10 (collapse) | 0.855 |

## Reading the results correctly

1. **One file, one rule:** `results\tables\results_summary.csv`. The primary-transformer
   rows currently in the consolidated file are the Colab-era run values; a later local
   rerun of the same 4 models reported different numbers (e.g. pit_ti grayscale 0.845 and
   efficientformer_l1 grayscale 0.91), but those rows are not part of the official tree —
   quote this file only.
2. Extended-backbone rows: ViT-B/16 from scratch reaches 80.5% grayscale, DeiT-Small 73%;
   the corrected Swin-Tiny LR=0.0001 run reaches 80.5% grayscale and 76.5% RGB on
   Dataset 2 (unseen users). **ConvNeXt-Tiny grayscale collapsed to predicting one class**
   (test acc 0.10, recall 0.10) — exactly the from-scratch convergence risk flagged in
   `models.py`. ConvNeXt-Tiny RGB, however, reached 85.5%.
3. The dashboard (`web\app.py` → `/api/multiclass`) reads this single file, so the chart
   rows match the CSV exactly.

## Highlights & interpretation `[New]`

- Best overall on unseen-user test: **vgg16 grayscale 94.5%** (val 99.0%).
- Best transformer (on the official tree): **efficientformer_l1 rgb 88%** or
  **swin_tiny grayscale 80.5%** — pick your metric, cite the file.
- CNNs hold the top of the unseen-user test table; the slack for lightweight from-scratch
  transformers is most visible on RGB (larger hypothesis space, only ~1,414 training images —
  expect overfitting; this is the paper's own regime).
- CPU / short-budget training shows in the ConvNeXt/Pit collapses; see doc 14 for the planned
  fixes (fine-tuning, pretrained variants) that are deferred by design (models stay frozen).
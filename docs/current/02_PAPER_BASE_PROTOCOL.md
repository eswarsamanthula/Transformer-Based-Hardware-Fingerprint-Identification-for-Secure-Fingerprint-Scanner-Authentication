# 02 — Base Paper Protocol (what both projects implement)

> Tag: `[Paper]` throughout unless noted. This is the protocol both the old (`THFSA`) and
> new (`THFSA_v2`) projects implement. The new project's `src\config.py` docstring states
> explicitly that every value mirrors what the paper actually did.

## Paper

User B et al., *"Robust Authentication: Leveraging Hardware Fingerprints and AI to Enhance
Security Against Spoofing"*, IEEE Access (2025). `[Paper]`
Referenced by both project READMEs as the base protocol.

## Data protocol

- **Dataset 1** (train/val): 2,019 raw BMP fingerprint images in 10 scanner folders
  (`Sensor1..Sensor10` / `Sensor1..Sensor10` naming). Folder name = 10-class scanner label.
- **Dataset 2** (~200 images): users User B/User A, files named `Sensor<n>(Test<m>)<k>.bmp` — (Test1) = User A, (Test2) = User B
  (10 captures × 2 users × 10 scanners). Used **only** as the final unseen-user test set —
  never combined with train/val. `[Paper]`, confirmed by `[New]` `dataset\Dataset2\Fingerprint_of_Test1_and_Test2\` (200 files, renamed on disk).
- Holdout: 70/30 train/validation on Dataset 1. The new project implements it `[New]` as a
  per-scanner stratified, image-level split, seed 42, persisted to
  `dataset\splits\train.csv` (1413 rows) / `val.csv` (606 rows).

## Training protocol

- **No data augmentation.** Only model-required resizing and [0,1] tensor conversion.
  A crop/flip/rotate/jitter would alter the sensor-noise pattern the method depends on.
- **Trained from scratch** — `pretrained=False` for every model (the paper trains all
  models, CNN or otherwise, from random weights).
- Optimizer: Adam. `[Paper]`
- Loss: categorical cross-entropy (multiclass); binary cross-entropy (binary one-vs-all).
- Metric tracked: categorical/binary accuracy every epoch.
- Early stopping on validation loss, patience 10, **best weights restored**.
- Batch size 8 or 32 per model ("depending on the computational requirements of each model").
  The new project defaults 32 for lighter backbones, 8 for the largest (effnet_b3, vgg16, vit_b16).
- Two channel modes trained and reported **separately**: grayscale (1 ch) and RGB (3 ch).
- Learning rate 0.001; **0.0001 for VGG16 and LeNet5** (stated by the paper).

## Metrics

- Multiclass: accuracy, precision, recall, F1 (macro), AUC.
- Binary: same, plus **FAR** (false acceptance rate) and **FRR** (false rejection rate) — the
  new project scores FAR/FRR at threshold 0.5 on the sigmoid output. `[New]`

## What the projects keep vs change

- Keep: data, split philosophy, no augmentation, from-scratch, optimizer/loss/patience,
  batch size guidance, grayscale+RGB modes, VGG16/LeNet5 custom heads and learning rates.
- Change: **which models are trained.** The paper's 6 CNNs (EfficientNetB3, MobileNet,
  ResNet50, ResNet152, VGG16, LeNet5) are kept as baselines; the projects add
  Transformer / hybrid-Transformer backbones. Old project added ViT-B/16, Swin-Tiny,
  DeiT-Small, ConvNeXt-Tiny (doc 06). New project's *headline* additions are the 4
  lightweight backbones TinyViT-5M, MobileViT-XS, EfficientFormer-L1, PiT-Ti (doc 07).

## Implementation notes locked into the new project's code

- `src\config.py` — single source of truth; MODEL_INPUT_SIZE per model (effnet 300,
  mobilevit 256, lenet 32, rest 224).
- `src\models.py` — LeNet5 reimplemented per paper (modified FC section: hidden 1024 →
  hidden 512 → 10, ReLU, Dropout(0.5)); EfficientNetB3/VGG16 get the paper's custom head
  (GAP → 512 → ReLU → Dropout(0.2) → 10) with VGG16's 4096-FC block removed
  (`pre_logits = Identity`).
- `src\train.py` / `src\test.py` — implement the loop above 1:1 (doc 07).
# 03 — Datasets (both projects)

## New project — unused as delivered layout `[New]`

### Dataset 1 — `dataset\Dataset1\Fingerprint\Sensor1..Sensor10\`
2,019 raw `.bmp` images; folder name **is** the scanner label (10 classes).

Per-scanner counts (verified by directory scan):

| Sensor | Files |
|---|---|
| Sensor1 | 279 |
| Sensor2 | 290 |
| Sensor3 | 180 |
| Sensor4 | 170 |
| Sensor5 | 170 |
| Sensor6 | 170 |
| Sensor7 | 180 |
| Sensor8 | 170 |
| Sensor9 | 210 |
| Sensor10 | 200 |
| **Total** | **2,019** |

Filename semantics: `fingerprint0.bmp ...` — **no user id in filenames, and no user mapping
is attempted anywhere** (no metadata source exists; the paper's user-level holdout is handled
entirely by Dataset 2). Scanner folders are the only available label. `[New]`
Original captured resolution: 288×256 (H×W), per `config.IMG_ORIG_SIZE`.
### Dataset 2 — `dataset\Dataset2\Fingerprint_of_Test1_and_Test2\` — final mock test

200 `.bmp` files named `Sensor<n>(Test<m>)<k>.bmp` (10 sensors × 2 users × 10
captures), renamed from `Sensor<n>{User B|User A}<k>.bmp` — **(Test1) = User A,
(Test2) = User B**.
The scanner label is parsed from the filename. It is **exclusively the final unseen-user test set**:
never written into split manifests, never used for train/val. `[New]` → `src\dataset.py` enforces this.

These 200 files live only under Dataset 2 in the new project; an identical-sounding older
copy exists in the old project root as `Fingerprint of User B and User A\` (see below).

## Old project — historical layout `[Old]`

| Directory | Contents | Status |
|---|---|---|
| `dataset\R307` | `train\` + `val\` per scanner (pre-split processed copy), 2019 images + `.gitkeep` | Evidence of the old processed-data split |
| `dataset_v1` | `R307\` + `split_verification.txt` (2020 files) — original grouped-split source | Evidence of the first split attempt |
| `Dataset_1_Fold` | 5 folds of 2,019 files (group-aware 5-fold, seed 42) | Evidence of fold-based experiments |
| `dataset_grouped_split` | **partial copy** — 404 files (278 train + 126 val) | incomplete in-project snapshot; the full grouped split lived externally |
| `Fingerprint of User B and User A\` | 200 files `SensorN{User B|User A}k.bmp` (same naming as new Dataset 2) | source of the unseen-user test filenames |

Verification files that document external grouped splits:
- `dataset_v1\split_verification.txt` — grouped split, seed 42, ratio 0.7;
  1,626 same-finger groups (1,138 train / 488 val), cross-split groups = 0; totals 1,414 / 605.
- `dataset\split_verification_v2.txt` — phash-based v2 (`[external local dataset path]`),
  perceptual hash `phash`, hash_size=16 (256-bit), min pairwise distance 34, 7 image pairs ≤ 40.

Both refer to external paths under `[external local path]/` — those are the *complete*
grouped splits; only partial copies were committed into the old project.

## Key distinctions

1. `[New]` raw data, folders = scanners, label from folder, train/val = manifests, test = Dataset 2.
2. `[Old]` processed copies pre-dated the rebuild: pre-split directories, fold copies, grouped-split copies.
3. The new project does **not** use fold documents, grouped splits, or per-user pseudo-labels:
   the final protocol is single seed-42 70/30 split + Dataset 2 (doc 04).
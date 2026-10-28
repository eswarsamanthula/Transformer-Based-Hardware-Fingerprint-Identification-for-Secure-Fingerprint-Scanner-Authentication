# 04 — Split Protocol & Manifests

## New project (current protocol) `[New]`

### Design (implemented in `src\dataset.py`)
- **Per-scanner stratified 70/30 image-level split** of Dataset 1: each sensor's images are
  shuffled and split 70/30 independently, so every sensor keeps ~70/30 balance.
- Fixed seed `SPLIT_SEED = 42` → fully reproducible.
- The split is **persisted to disk** as two manifests:
  - `dataset\splits\train.csv` — 1,413 data rows (header + 1,413) `[New]`
  - `dataset\splits\val.csv` — 606 data rows `[New]`
  - Row format: `filepath,sensor_label`.
- If both manifests exist they are **read (identical split every run)**; if either is
  missing the split is regenerated, the MD5 leakage gate runs against Dataset 2 (aborts on
  any byte-identical match), and the manifests are rewritten.
- Dataset 2 is **never** part of the manifests — it is scanned separately as the final test set.

### Verified manifest statistics
- train + val = 1,413 + 606 = **2,019** = the full Dataset 1. `[New]` (directory count of each folder also totals 2,019 — doc 03).

### Why image-level, not user-level
Dataset 1 filenames carry no user id and no metadata source exists, so a user-level
(train on user A/B, test on user C/D) split is **impossible** on Dataset 1. The paper's
user-level generalization is handled entirely by Dataset 2 (unseen users User B/User A).
`[Paper]` protocol + `[New]` implementation.

## Old project — split history `[Old]` (documented for reference, not merged into the new protocol)

1. **Continuously pre-split processed copies** — `dataset\R307\train|val` per scanner.
2. **Grouped split by filename pairs** (`split_verification.txt`, seed 42, ratio 0.7):
   same-finger images grouped by matching `fingerprintN.bmp`/`fingerprintN(2).bmp`;
   1,626 groups, 1,138 train / 488 val, **cross-split groups = 0**, totals 1,414 / 605.
   This is the closest the old project came to separating same-finger captures.
3. **Group-aware 5-fold** (`create_folds.py`, seed 42, SRC `dataset\R307`,
   DST `dataset\Dataset_1_Fold`) — folds of 2,019 files used for the cross-scanner
   binary 5-fold campaign (doc 06).
4. **phash-based grouped split** (`Dataset_1_grouped_split_v2`, `split_verification_v2.txt`):
   `phash` hash_size=16 (256-bit), min pairwise phash distance 34, 7 image pairs ≤ 40 —
   a stricter near-duplicate grouping pass.

### Caveats recorded `[Old]`
- The in-project `dataset_grouped_split\` is a **partial** copy (404/2,019 files:
  278 train + 126 val). The complete grouped splits were only present under
  `[external local dataset path]` and `..._grouped_split_v2`.
- `[Not verified]` whether any old experiment actually trained from the complete external
  grouped sources; the old `results*` tables carry a `flag` column whose meaning changed
  across runs (doc 06).

## Decision & rationale
The new protocol uses the **seed-42 per-scanner stratified image-level split + Dataset 2
holdout**, exactly per the paper, persisted to manifests for reproducibility. User-level
separation is reported only through Dataset 2 scores.
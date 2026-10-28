# 14 — Missing / Planned Experiments

> Tags: `[Done]` (completed since original write-up), `[Planned]` / `[Proposed]`,
> `[Deferred]` (needs retraining; intentionally not done — the trained networks are frozen).

## Resolved (no longer pending)
2. **Binary runs beyond mobilenet** — **`[Done]`**: `results/tables/results_binary_summary.csv`
   now also covers **vgg16/grayscale** (acc 0.9395, F1 0.6828, AUC 0.7423) and
   **efficientformer_l1/rgb** (acc 0.9275, F1 0.6934, AUC 0.7902), test split; full
   per-model story in `docs/current/19_FINAL_RESULTS.md`.
5. **Per-scanner binary thresholds** — **`[Done]`**:
   `results/tables/binary_threshold_calibration.csv` (per-model/per-scanner/ per-channel
   thresholds fitted on Dataset 1 validation) + `binary_threshold_test_results.csv`
   (Dataset 2 at frozen thresholds). Consumed by `scanner_verification.py`; FAR/FRR curves
   reported in `results/metrics/*.csv` and `docs/current/19_FINAL_RESULTS.md`.

## Training/completion gaps (deferred — no retraining by design)
1. **ConvNeXt-Tiny grayscale collapse (test acc 0.10, one-class)** — retry with
   pretrained=True, a shorter warmup + lower LR, or label smoothing. **Deferred**: the task
   froze the trained networks; nothing is retrained. `[Deferred]`
3. **resnet152 checkpoints in `new_result\`** — `[Done]`: the `new_result\` tree no longer
   exists and resnet152 `.pt` are present under `checkpoints\cnn\`.
4. **Seed averaging** — single seed 42; rerun top-3 models with 3 seeds for error bars.
   `[Deferred]` (requires retraining.)

## Evaluations that do not exist yet
6. **Unseen-hardware (cross-sensor) test** — hold out one whole sensor for test. Properly
   different dataset configuration; the current Dataset-2 test is unseen-user-on-known-hardware
   only. Requires a data pass + a small dataset splitter variant. `[Planned]`
7. **Spoofing / liveness / presentation-attack evidence** — the paper claims spoofing
   defense; there is **no spoof dataset in this repo**. Any anti-spoofing claim stays unbacked. `[Proposed]`
8. **Inference latency / FLOPs / size table** — old `edge\benchmark.py` scaffolded this for
   the old stack; re-run for the final checkpoints (CPU), incl. per-model param counts
   (`models.py::count_params`). Feeds the paper's latency comparison. `[Planned]`
9. **Feature/saliency analysis** — evidence that decisions use sensor noise (e.g. Grad-CAM on
   genuine-vs-impostor pairs). Nice-to-have for the paper's "hardware fingerprint" claim. `[Proposed]`
10. **Calibration / threshold-aware reporting** — AUCs are high but Point-0.5 accuracy lags;
    report EER (equal-error-rate) per scanner as the canonical verification statistic. `[Planned]`

## Housekeeping
11. **Consolidate checkpoint trees** — `[Done]`: everything now lives under
    `checkpoints\cnn\` + `checkpoints\transformer\` (68 `.pt`); no `new_result\`/
    `transformer_colab\` trees remain and `src\test.py` finds every checkpoint (doc 11).
12. **Regenerate any missing log CSV parity** — `[Done]`: all 68 training logs regrouped
    under `logs\training\`; no `/colab` sub-tree remains.
13. **Timestamps on results** — consider a `runs\` provenance ledger (seed, epoch, file,
    hash) going forward. `[Proposed]`

## Explicit non-goals (do not start these)
- No user-level splitting of Dataset 1 (no user labels exist).
- No merging of old 5-fold numbers into the new leaderboard (different protocol — doc 16).

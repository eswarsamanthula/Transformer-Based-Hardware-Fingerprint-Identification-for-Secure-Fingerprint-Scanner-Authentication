# 12 — Project Contributions

## Contributions of the rebuild (`THFSA_v2`, `[New]` — all implemented and evidenced)

1. **Clean raw-data pipeline reproducing the paper protocol** — `src\` (config/dataset/
   models/train/test) implements no-augmentation, from-scratch, Adam, CE/BCE, patience-10-ES,
   best-weights-restore, batch 8/32, and grayscale+RGB — 1:1 with the paper; nothing cached,
   no processed copies needed. `[Paper]`+`[New]`.
2. **Persistent reproducible split** — seed-42 per-scanner stratified 70/30 manifests
   (`dataset\splits\train.csv` 1,413 / `val.csv` 606 / = 2,019) read back identically every run.
3. **Dataset 2 as a true unseen-user final test** kept fully separate from train/val —
   never written into manifests; the only place the paper's user-level holdout is realised.
4. **Byte-level MD5 leakage gate** in the splitter (aborts on any Dataset 1↔Dataset 2 match;
   run result: 0 matches) + write-era pixel-level near-duplicate analysis (max corr ≈0.835,
   closest pair ≈38% pixels differ) — the "no leakage" claim is evidenced, not asserted.
5. **The 4 lightweight primary Transformer/hybrid backbones** (TinyViT-5M, MobileViT-XS,
   EfficientFormer-L1, PiT-Ti) added to the paper's 6 CNNs — trained from scratch and
   evaluated on the unseen-user holdout, with both a Colab run and a local rerun on record.
6. **Binary one-vs-all verification service** with FAR/FRR at threshold 0.5 and a
   per-scanner × 2-channel × val/test table (mobilenet) — the piece the old project only
   scaffolded.
7. **Working HMAC-SHA256 multi-factor authentication** (`authentication\`, merged
   `/auth` page: register/sign-in with live gate trace + identify/verify/MFA lab,
   cached inference, validated sample picker) — verified end-to-end in this
   delivery period; supersedes the old `mfa\` scaffold.
8. **Honest failure reporting** — the ConvNeXt-Tiny grayscale from-scratch collapse is
   documented rather than hidden (models.py flags the expected risk; docs 07/10 cite the
   actual 0.10 value). Corrected Swin-Tiny LR=0.0001 results are 80.5% grayscale and 76.5% RGB.
9. **Documentation** — this `docs\` series + README/requirements in both projects, plus the
   old project's methodology.txt-era paperwork (`docs\methodology.md`, split audits/verification
   files).

## Historical contributions (`[Old]`, project `THFSA`)

- First end-to-end implementation and the **cross-scanner 5-fold binary study** (50 runs;
   verified macro averages acc 97.79%, F1 85.71%, FAR 1.01%, FRR 14.87%).
- Early suspicion-flagging heuristics and per-scanner metric JSONs.
- Tooling: group-aware 5-fold generator, grouped-split verification, edge scaffolding
  (benchmark/export/quantize), Phase II web dashboard, mfa scaffold.

## What is NOT yet a contribution (avoid claiming)

- ConvNeXt grayscale usefulness — currently collapsed; Swin-Tiny LR=0.0001 reaches 80.5%
  grayscale and 76.5% RGB, while rgb ConvNeXt reaches 0.855.
- Threshold-tuned binary verification, pretrained variants, unseen-hardware results,
  spoofing/liveness defense evidence — none exist yet (doc 14). `[Planned]`

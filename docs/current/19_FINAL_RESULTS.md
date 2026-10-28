# 19 — Final Experimental Results & Authentication MFA

> Final packaging of the measured results and the working authentication path.
> **No model training was performed in this phase** — every number below comes
> from the frozen artefacts already produced by the earlier experiments.

## 1. Non-training statement

All ML training was completed previously (see docs/07, docs/08). This
phase only: inspected results, derived presentation files from the existing
tables, generated graphs from those tables, and wired the results into the
web application. No training, inference, checkpoint replacement, split
change, threshold recalibration, or metric estimation was performed.

## 2. Final verification models

One-vs-all scanner **verification** models (frozen checkpoints, one binary
classifier per scanner 1–10):

| Model | Channel | Role |
|---|---|---|
| `vgg16` | grayscale | **CNN baseline** — 10 one-vs-all ScannerVerifiers |
| `efficientformer_l1` | rgb | **proposed lightweight Transformer/hybrid** — 10 one-vs-all ScannerVerifiers |

Scanner **identification** for the live demo additionally uses the same
frozen checkpoints. The binary models are one-vs-all scanners: they answer
"is this fingerprint from scanner N?" — they are **not** user-identity
classifiers.

## 3. Final metrics (Dataset 2 test · mean over scanners 1–10)

| Model | Channel | Accuracy | Precision | Recall | F1 | AUC | FAR | FRR |
|---|---|---|---|---|---|---|---|---|
| VGG16 | grayscale | 93.95% | 66.97% | 69.75% | 68.28% | 74.23% | 0.000% | 60.50% |
| EfficientFormer-L1 | rgb | 92.75% | 72.51% | 68.64% | 69.34% | 79.02% | 1.222% | 61.50% |

Source: `results/tables/results_binary_summary.csv` (`split=test`,
`scanner=mean`). Presentation copy: `results/metrics/final_model_comparison.csv`.

- Accuracy alone is not sufficient for authentication evaluation; FAR/FRR are
  the security-relevant metrics and AUC summarises discrimination.
- Neither model dominates: VGG16 leads on accuracy/FRR/FAR, EfficientFormer-L1
  leads on recall/F1/AUC. Report each metric as measured.

## 4. Scanner-wise results (Dataset 2 test)

| Scanner | Model | Channel | Accuracy | F1 | AUC | FAR | FRR |
|---|---|---|---|---|---|---|---|
| S1 | VGG16 | grayscale | 90.00% | 47.37% | 36.67% | 0.000% | 100.00% |
| S2 | VGG16 | grayscale | 100.00% | 100.00% | 100.00% | 0.000% | 0.00% |
| S3 | VGG16 | grayscale | 90.00% | 47.37% | 55.25% | 0.000% | 100.00% |
| S4 | VGG16 | grayscale | 99.50% | 98.58% | 100.00% | 0.000% | 5.00% |
| S5 | VGG16 | grayscale | 90.00% | 47.37% | 52.42% | 0.000% | 100.00% |
| S6 | VGG16 | grayscale | 100.00% | 100.00% | 100.00% | 0.000% | 0.00% |
| S7 | VGG16 | grayscale | 90.00% | 47.37% | 73.03% | 0.000% | 100.00% |
| S8 | VGG16 | grayscale | 90.00% | 47.37% | 57.06% | 0.000% | 100.00% |
| S9 | VGG16 | grayscale | 90.00% | 47.37% | 67.89% | 0.000% | 100.00% |
| S10 | VGG16 | grayscale | 100.00% | 100.00% | 100.00% | 0.000% | 0.00% |
| S1 | EfficientFormer-L1 | rgb | 90.00% | 47.37% | 66.53% | 0.000% | 100.00% |
| S2 | EfficientFormer-L1 | rgb | 93.50% | 75.83% | 95.28% | 0.556% | 60.00% |
| S3 | EfficientFormer-L1 | rgb | 93.00% | 73.12% | 94.69% | 0.556% | 65.00% |
| S4 | EfficientFormer-L1 | rgb | 93.00% | 79.65% | 92.69% | 3.333% | 40.00% |
| S5 | EfficientFormer-L1 | rgb | 89.50% | 47.23% | 70.78% | 0.556% | 100.00% |
| S6 | EfficientFormer-L1 | rgb | 100.00% | 100.00% | 100.00% | 0.000% | 0.00% |
| S7 | EfficientFormer-L1 | rgb | 91.00% | 72.53% | 90.08% | 3.889% | 55.00% |
| S8 | EfficientFormer-L1 | rgb | 90.00% | 47.37% | 24.69% | 0.000% | 100.00% |
| S9 | EfficientFormer-L1 | rgb | 87.50% | 50.35% | 55.47% | 3.333% | 95.00% |
| S10 | EfficientFormer-L1 | rgb | 100.00% | 100.00% | 100.00% | 0.000% | 0.00% |

Scanner-level variation is real and explicit: some sensors verify
near-perfectly (e.g. scanners 6 and 10 for both models), while several others
show FRR near 1.0 and near-chance AUC. This is exactly why per-scanner
threshold calibration exists and why averages alone are not a sufficient
report.

Source: `results/tables/results_binary_summary.csv`.
Presentation copy: `results/metrics/scanner_wise_results.csv`.

## 5. Threshold calibration (validation-frozen)

| Scanner | Model | Threshold | Val Acc | Val F1 | Val FAR | Val FRR | Test Acc | Test F1 | Test FAR | Test FRR |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | VGG16 | 0.17 | 24.59% | 26.88% | 87.548% | 0.00% | 22.00% | 19.59% | 86.111% | 5.00% |
| S2 | VGG16 | 0.11 | 100.00% | 100.00% | 0.000% | 0.00% | 100.00% | 100.00% | 0.000% | 0.00% |
| S3 | VGG16 | 0.10 | 35.15% | 20.93% | 70.833% | 3.70% | 40.00% | 20.00% | 63.889% | 25.00% |
| S4 | VGG16 | 0.05 | 100.00% | 100.00% | 0.000% | 0.00% | 100.00% | 100.00% | 0.000% | 0.00% |
| S5 | VGG16 | 0.08 | 8.58% | 15.55% | 99.820% | 0.00% | 10.00% | 18.18% | 100.000% | 0.00% |
| S6 | VGG16 | 0.05 | 100.00% | 100.00% | 0.000% | 0.00% | 100.00% | 100.00% | 0.000% | 0.00% |
| S7 | VGG16 | 0.09 | 55.12% | 22.73% | 46.739% | 25.93% | 14.50% | 18.96% | 95.000% | 0.00% |
| S8 | VGG16 | 0.08 | 81.52% | 24.32% | 14.234% | 64.71% | 89.50% | 0.00% | 0.556% | 100.00% |
| S9 | VGG16 | 0.05 | 99.83% | 99.21% | 0.184% | 0.00% | 90.00% | 0.00% | 0.000% | 100.00% |
| S10 | VGG16 | 0.05 | 100.00% | 100.00% | 0.000% | 0.00% | 100.00% | 100.00% | 0.000% | 0.00% |
| S1 | EfficientFormer-L1 | 0.28 | 78.22% | 36.54% | 16.475% | 54.76% | 77.50% | 21.05% | 17.222% | 70.00% |
| S2 | EfficientFormer-L1 | 0.53 | 89.93% | 59.60% | 3.661% | 48.28% | 93.50% | 55.17% | 0.556% | 60.00% |
| S3 | EfficientFormer-L1 | 0.19 | 94.72% | 72.41% | 3.623% | 22.22% | 94.00% | 64.71% | 1.667% | 45.00% |
| S4 | EfficientFormer-L1 | 0.27 | 92.24% | 51.55% | 3.784% | 50.98% | 80.50% | 46.58% | 20.000% | 15.00% |
| S5 | EfficientFormer-L1 | 0.19 | 82.84% | 35.00% | 14.595% | 45.10% | 77.50% | 26.23% | 18.333% | 60.00% |
| S6 | EfficientFormer-L1 | 0.18 | 100.00% | 100.00% | 0.000% | 0.00% | 100.00% | 100.00% | 0.000% | 0.00% |
| S7 | EfficientFormer-L1 | 0.65 | 96.37% | 78.85% | 1.630% | 24.07% | 91.50% | 45.16% | 2.222% | 65.00% |
| S8 | EfficientFormer-L1 | 0.06 | 80.86% | 29.27% | 16.036% | 52.94% | 89.50% | 0.00% | 0.556% | 100.00% |
| S9 | EfficientFormer-L1 | 0.15 | 95.21% | 78.20% | 3.315% | 17.46% | 85.00% | 6.25% | 6.111% | 95.00% |
| S10 | EfficientFormer-L1 | 0.38 | 99.67% | 98.31% | 0.000% | 3.33% | 100.00% | 100.00% | 0.000% | 0.00% |

**Thresholds were calibrated on the validation data and frozen before
Dataset 2 testing.** Dataset 2 was never used to select thresholds.

Source: `results/tables/binary_threshold_calibration.csv` (val) merged with
`results/tables/binary_threshold_test_results.csv` (test at the frozen
threshold). Presentation copy:
`results/metrics/threshold_summary.csv`.

## 6. Graphs

Generated from the same tables (`results/graphs/*.png`):

| File | Content |
|---|---|
| `model_accuracy_comparison.png` | VGG16 vs EfficientFormer-L1 acc |
| `model_f1_comparison.png` | F1 comparison |
| `model_auc_comparison.png` | AUC comparison |
| `model_far_comparison.png` | FAR comparison |
| `model_frr_comparison.png` | FRR comparison |
| `scanner_wise_accuracy.png` | Acc per scanner, both models |
| `scanner_wise_f1.png` | F1 per scanner |
| `scanner_wise_auc.png` | AUC per scanner |
| `scanner_wise_far_frr.png` | FAR and FRR per scanner |
| `scanner_thresholds.png` | Calibrated threshold per scanner, both models |

## 7. Dataset evaluation

| Dataset | Content | Role |
|---|---|---|
| Dataset 1 | 2,019 raw fingerprints, folder = scanner (Sensor 1–10) | training + validation (70/30 per-scanner stratified); threshold calibration source |
| Dataset 2 | 200 fingerprints from users **User B & User A** (`Sensor<n>{User B\|User A}<k>.bmp`) | independent test set, **unseen users**, **known scanner hardware** |

Dataset 2 is an **unseen-user** evaluation on **known scanner hardware**. It
is never described as unseen- or cross- hardware generalization.

## 8. Authentication architecture (HMAC-MFA, module level)

| # | Component | Method | Factor | Measured | Notes |
|---|---|---|---|---|---|
| 1 | User ID | `user_store (SQLite user lookup)` | identity handle | no | not a security factor by itself |
| 2 | Password | `bcrypt (checkpw)` | knowledge factor | no | protocol gate, no FRR benchmark measured |
| 3 | User fingerprint verification | `user_biometric derived 64-vector cosine template` | biometric factor (user identity) | no | academic prototype matcher, not production-grade biometric matching |
| 4 | Scanner hardware identification | `ScannerIdentifierBinary - frozen one-vs-all VGG16-grayscale + EfficientFormer-L1-RGB checkpoints` | acquired-device factor | yes | see final_model_comparison.csv / scanner_wise_results.csv |
| 5 | Scanner threshold verification | `per-scanner calibrated threshold (validation-frozen)` | acquired-device factor | yes | val metrics only; see threshold_summary.csv |
| 6 | TOTP second factor | `passkey.py RFC-6238 (pyotp)` | possession factor | no | no independent benchmark measured |
| 7 | HMAC-SHA256 mutual authentication | `K = HMAC(V, t_user || t_server)` | channel/session integrity | no | server_secret.bin; verified end-to-end by tests |
| 8 | Session key | `shared key after mutual auth` | session | no | established only after all gates pass |

### Registration

1. User ID
2. Password
3. Fingerprint enrollment
4. Claimed scanner
5. TOTP secret enrollment

→ stored: user biometric template (derived, JSON) · bcrypt password hash ·
TOTP secret · scanner binding.

### Login

1. User ID
2. Password
3. Fingerprint capture
4. Claimed scanner
5. TOTP code
6. User biometric verification (template)
7. Scanner hardware verification (VGG16 / EfficientFormer-L1 one-vs-all)
8. Scanner threshold verification (validation-frozen thresholds)
9. HMAC-SHA256 mutual authentication
10. Session key
11. Access granted

Every intermediate FALSE aborts. The shared session key is derived only after
the mutual HMAC exchange succeeds.

**Important:** VGG16 / EfficientFormer-L1 identify and verify the scanner
hardware fingerprint. They are *not* user-identity classifiers. User-level
fingerprint identity is provided by the separate biometric template module
(`authentication/user_biometric.py`, academic prototype).

## 9. Limitations

- TOTP is the current second factor; true WebAuthn/FIDO2 attestation is not
  implemented.
- QR rendering is not implemented; the registration flow exposes the TOTP
  secret / `otpauth://` URI directly in the overlay.
- No claim of live R307 hardware capture is made; enrolment/verification use
  uploaded images or Dataset 2 samples.
- The biometric template matcher is an academic prototype (derived 64-vector
  cosine similarity), not production-grade biometric recognition.
- Dataset 2 evaluates unseen users on known scanner hardware — not unseen
  hardware.
- Thresholds were calibrated on validation data and frozen before the final
  Dataset 2 test.

## 10. Sources of truth

- `results/tables/results_binary_summary.csv`
- `results/tables/binary_threshold_calibration.csv`
- `results/tables/binary_threshold_test_results.csv`
- `results/tables/results_summary.csv` (multiclass leaderboard, dashboard)
- `logs/training/` — per-epoch training curves (dashboard)
- `results/metrics/` — derived, presentation-ready copies
- `results/graphs/` — generated charts
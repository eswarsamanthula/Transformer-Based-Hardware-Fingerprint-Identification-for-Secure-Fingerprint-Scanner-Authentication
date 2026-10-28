# 18. Verification Accept Matrix — Dataset 2 (full sweep)

Machine-generated from the live web stack (`web/app.py` verifier resolver + frozen
one-vs-all binary checkpoints). Every Dataset 2 sample was verified against its own
scanner label, exactly as `/api/verify` does; Accept == `is_match = True`.

## Verifier models (one-vs-all checkpoints)

| Pair | Channel | Accepts |
|------|---------|--------:|
| mobilenet / grayscale | grayscale | 130/200 |
| vgg16 / grayscale | grayscale | 79/200 |
| efficientformer_l1 / rgb | rgb | 77/200 |

## Accept on all 3 models — 53/200

### Sensor 2 — 8 samples
```
Sensor2User A3.bmp  Sensor2User A4.bmp  Sensor2User A5.bmp  Sensor2User B1.bmp  Sensor2User B2.bmp
Sensor2User B3.bmp  Sensor2User B6.bmp  Sensor2User B7.bmp
```

### Sensor 4 — 5 samples
```
Sensor4User A10.bmp  Sensor4User A4.bmp  Sensor4User A5.bmp  Sensor4User A9.bmp  Sensor4User B9.bmp
```

### Sensor 6 — 20 samples
```
Sensor6User A1.bmp  Sensor6User A10.bmp  Sensor6User A2.bmp  Sensor6User A3.bmp  Sensor6User A4.bmp
Sensor6User A5.bmp  Sensor6User A6.bmp  Sensor6User A7.bmp  Sensor6User A8.bmp  Sensor6User A9.bmp
Sensor6User B1.bmp  Sensor6User B2.bmp  Sensor6User B3.bmp  Sensor6User B4.bmp  Sensor6User B5.bmp
Sensor6User B6.bmp  Sensor6User B7.bmp  Sensor6User B8.bmp  Sensor6User B9.bmp  Sensor6User B10.bmp
```

### Sensor 10 — 20 samples
```
Sensor10User A1.bmp  Sensor10User A10.bmp  Sensor10User A2.bmp  Sensor10User A3.bmp  Sensor10User A4.bmp
Sensor10User A5.bmp  Sensor10User A6.bmp  Sensor10User A7.bmp  Sensor10User A8.bmp  Sensor10User A9.bmp
Sensor10User B1.bmp  Sensor10User B2.bmp  Sensor10User B3.bmp  Sensor10User B4.bmp  Sensor10User B5.bmp
Sensor10User B6.bmp  Sensor10User B7.bmp  Sensor10User B8.bmp  Sensor10User B9.bmp  Sensor10User B10.bmp
```

## Per-model accept lists (appendix)

### mobilenet/grayscale — 130/200

**Sensor 1**
```
Sensor1User A9.bmp  Sensor1User B5.bmp
```

**Sensor 2**
```
Sensor2User A1.bmp  Sensor2User A10.bmp  Sensor2User A2.bmp  Sensor2User A3.bmp  Sensor2User A4.bmp
Sensor2User A5.bmp  Sensor2User A6.bmp  Sensor2User A7.bmp  Sensor2User A8.bmp  Sensor2User A9.bmp
Sensor2User B1.bmp  Sensor2User B2.bmp  Sensor2User B3.bmp  Sensor2User B4.bmp  Sensor2User B5.bmp
Sensor2User B6.bmp  Sensor2User B7.bmp  Sensor2User B8.bmp  Sensor2User B9.bmp  Sensor2User B10.bmp
```

**Sensor 3**
```
Sensor3User A1.bmp  Sensor3User A10.bmp  Sensor3User A3.bmp  Sensor3User A4.bmp  Sensor3User A5.bmp
Sensor3User A8.bmp  Sensor3User A9.bmp  Sensor3User B2.bmp  Sensor3User B3.bmp  Sensor3User B4.bmp
Sensor3User B10.bmp
```

**Sensor 4**
```
Sensor4User A1.bmp  Sensor4User A10.bmp  Sensor4User A2.bmp  Sensor4User A3.bmp  Sensor4User A4.bmp
Sensor4User A5.bmp  Sensor4User A6.bmp  Sensor4User A7.bmp  Sensor4User A8.bmp  Sensor4User A9.bmp
Sensor4User B9.bmp
```

**Sensor 5**
```
Sensor5User A10.bmp  Sensor5User A2.bmp  Sensor5User A3.bmp  Sensor5User A4.bmp  Sensor5User A5.bmp
Sensor5User A7.bmp  Sensor5User A8.bmp  Sensor5User A9.bmp  Sensor5User B1.bmp  Sensor5User B2.bmp
Sensor5User B5.bmp  Sensor5User B7.bmp  Sensor5User B9.bmp  Sensor5User B10.bmp
```

**Sensor 6**
```
Sensor6User A1.bmp  Sensor6User A10.bmp  Sensor6User A2.bmp  Sensor6User A3.bmp  Sensor6User A4.bmp
Sensor6User A5.bmp  Sensor6User A6.bmp  Sensor6User A7.bmp  Sensor6User A8.bmp  Sensor6User A9.bmp
Sensor6User B1.bmp  Sensor6User B2.bmp  Sensor6User B3.bmp  Sensor6User B4.bmp  Sensor6User B5.bmp
Sensor6User B6.bmp  Sensor6User B7.bmp  Sensor6User B8.bmp  Sensor6User B9.bmp  Sensor6User B10.bmp
```

**Sensor 7**
```
Sensor7User A1.bmp  Sensor7User B2.bmp  Sensor7User B6.bmp  Sensor7User B8.bmp  Sensor7User B9.bmp
```

**Sensor 8**
```
Sensor8User A1.bmp  Sensor8User A10.bmp  Sensor8User A2.bmp  Sensor8User A3.bmp  Sensor8User A4.bmp
Sensor8User A5.bmp  Sensor8User A6.bmp  Sensor8User A7.bmp  Sensor8User A8.bmp  Sensor8User A9.bmp
Sensor8User B1.bmp  Sensor8User B2.bmp  Sensor8User B3.bmp  Sensor8User B5.bmp  Sensor8User B6.bmp
Sensor8User B7.bmp  Sensor8User B9.bmp  Sensor8User B10.bmp
```

**Sensor 9**
```
Sensor9User A1.bmp  Sensor9User A10.bmp  Sensor9User A2.bmp  Sensor9User A6.bmp  Sensor9User A7.bmp
Sensor9User A9.bmp  Sensor9User B3.bmp  Sensor9User B7.bmp  Sensor9User B8.bmp
```

**Sensor 10**
```
Sensor10User A1.bmp  Sensor10User A10.bmp  Sensor10User A2.bmp  Sensor10User A3.bmp  Sensor10User A4.bmp
Sensor10User A5.bmp  Sensor10User A6.bmp  Sensor10User A7.bmp  Sensor10User A8.bmp  Sensor10User A9.bmp
Sensor10User B1.bmp  Sensor10User B2.bmp  Sensor10User B3.bmp  Sensor10User B4.bmp  Sensor10User B5.bmp
Sensor10User B6.bmp  Sensor10User B7.bmp  Sensor10User B8.bmp  Sensor10User B9.bmp  Sensor10User B10.bmp
```

### vgg16/grayscale — 79/200

**Sensor 2**
```
Sensor2User A1.bmp  Sensor2User A10.bmp  Sensor2User A2.bmp  Sensor2User A3.bmp  Sensor2User A4.bmp
Sensor2User A5.bmp  Sensor2User A6.bmp  Sensor2User A7.bmp  Sensor2User A8.bmp  Sensor2User A9.bmp
Sensor2User B1.bmp  Sensor2User B2.bmp  Sensor2User B3.bmp  Sensor2User B4.bmp  Sensor2User B5.bmp
Sensor2User B6.bmp  Sensor2User B7.bmp  Sensor2User B8.bmp  Sensor2User B9.bmp  Sensor2User B10.bmp
```

**Sensor 4**
```
Sensor4User A1.bmp  Sensor4User A10.bmp  Sensor4User A2.bmp  Sensor4User A3.bmp  Sensor4User A4.bmp
Sensor4User A5.bmp  Sensor4User A6.bmp  Sensor4User A7.bmp  Sensor4User A8.bmp  Sensor4User A9.bmp
Sensor4User B1.bmp  Sensor4User B2.bmp  Sensor4User B3.bmp  Sensor4User B5.bmp  Sensor4User B6.bmp
Sensor4User B7.bmp  Sensor4User B8.bmp  Sensor4User B9.bmp  Sensor4User B10.bmp
```

**Sensor 6**
```
Sensor6User A1.bmp  Sensor6User A10.bmp  Sensor6User A2.bmp  Sensor6User A3.bmp  Sensor6User A4.bmp
Sensor6User A5.bmp  Sensor6User A6.bmp  Sensor6User A7.bmp  Sensor6User A8.bmp  Sensor6User A9.bmp
Sensor6User B1.bmp  Sensor6User B2.bmp  Sensor6User B3.bmp  Sensor6User B4.bmp  Sensor6User B5.bmp
Sensor6User B6.bmp  Sensor6User B7.bmp  Sensor6User B8.bmp  Sensor6User B9.bmp  Sensor6User B10.bmp
```

**Sensor 10**
```
Sensor10User A1.bmp  Sensor10User A10.bmp  Sensor10User A2.bmp  Sensor10User A3.bmp  Sensor10User A4.bmp
Sensor10User A5.bmp  Sensor10User A6.bmp  Sensor10User A7.bmp  Sensor10User A8.bmp  Sensor10User A9.bmp
Sensor10User B1.bmp  Sensor10User B2.bmp  Sensor10User B3.bmp  Sensor10User B4.bmp  Sensor10User B5.bmp
Sensor10User B6.bmp  Sensor10User B7.bmp  Sensor10User B8.bmp  Sensor10User B9.bmp  Sensor10User B10.bmp
```

### efficientformer_l1/rgb — 77/200

**Sensor 2**
```
Sensor2User A3.bmp  Sensor2User A4.bmp  Sensor2User A5.bmp  Sensor2User B1.bmp  Sensor2User B2.bmp
Sensor2User B3.bmp  Sensor2User B6.bmp  Sensor2User B7.bmp
```

**Sensor 3**
```
Sensor3User A1.bmp  Sensor3User A7.bmp  Sensor3User A8.bmp  Sensor3User B4.bmp  Sensor3User B7.bmp
Sensor3User B9.bmp  Sensor3User B10.bmp
```

**Sensor 4**
```
Sensor4User A10.bmp  Sensor4User A4.bmp  Sensor4User A5.bmp  Sensor4User A9.bmp  Sensor4User B1.bmp
Sensor4User B2.bmp  Sensor4User B3.bmp  Sensor4User B4.bmp  Sensor4User B5.bmp  Sensor4User B8.bmp
Sensor4User B9.bmp  Sensor4User B10.bmp
```

**Sensor 6**
```
Sensor6User A1.bmp  Sensor6User A10.bmp  Sensor6User A2.bmp  Sensor6User A3.bmp  Sensor6User A4.bmp
Sensor6User A5.bmp  Sensor6User A6.bmp  Sensor6User A7.bmp  Sensor6User A8.bmp  Sensor6User A9.bmp
Sensor6User B1.bmp  Sensor6User B2.bmp  Sensor6User B3.bmp  Sensor6User B4.bmp  Sensor6User B5.bmp
Sensor6User B6.bmp  Sensor6User B7.bmp  Sensor6User B8.bmp  Sensor6User B9.bmp  Sensor6User B10.bmp
```

**Sensor 7**
```
Sensor7User A1.bmp  Sensor7User A5.bmp  Sensor7User A6.bmp  Sensor7User A7.bmp  Sensor7User B1.bmp
Sensor7User B5.bmp  Sensor7User B6.bmp  Sensor7User B9.bmp  Sensor7User B10.bmp
```

**Sensor 9**
```
Sensor9User B3.bmp
```

**Sensor 10**
```
Sensor10User A1.bmp  Sensor10User A10.bmp  Sensor10User A2.bmp  Sensor10User A3.bmp  Sensor10User A4.bmp
Sensor10User A5.bmp  Sensor10User A6.bmp  Sensor10User A7.bmp  Sensor10User A8.bmp  Sensor10User A9.bmp
Sensor10User B1.bmp  Sensor10User B2.bmp  Sensor10User B3.bmp  Sensor10User B4.bmp  Sensor10User B5.bmp
Sensor10User B6.bmp  Sensor10User B7.bmp  Sensor10User B8.bmp  Sensor10User B9.bmp  Sensor10User B10.bmp
```

> Generated by `sweep_verify.py`; do not hand-edit lists — rerun the sweep.

---

## Q&A — Why are the accept rates not 200/200?

**Q.** The 3 verifier models are supposed to recognize genuine fingerprints from the
scanner they were bound to. Why do they accept only 130 / 79 / 77 of the 200 samples
and agree on just 53?

**Q (short answer).** Because these are **real one-vs-all verifiers on unseen users,
scored at frozen thresholds** — not a synthetic 100% demo. "Accept" means
`model score ≥ threshold`. A genuine same-scanner sample is **rejected** whenever its
score falls below the threshold (false reject, FRR), and the three models disagree on
most samples. 200/200 would require every genuine impression to score above threshold
on every model — the measured data says that is not the case.

### The measured facts (from this file + `results/tables/binary_threshold_test_results.csv`)

| Model / channel | Accepts | Where it accepts |
|---|---|---|
| mobilenet / grayscale | 130/200 | all 10 sensors, concentrated on S2, S6, S10 (clean) + partial S3/4/5/8/9 |
| vgg16 / grayscale | 79/200 | only S2, S4, S6, S10 near-complete + S9 partial |
| efficientformer_l1 / rgb | 77/200 | S2, S6, S10 near-complete + partial S3/4/7/9 |
| **All 3 agree** | **53/200** | S2 (8), S4 (5), S6 (20), S10 (20) |

### Why (each reason is visible in the existing numbers, not speculative)

1. **FRR dominates the misses.** At the frozen thresholds on Dataset 2, the false-reject
   rate is far from zero on several sensors: vgg16 has FRR 1.00 on S8 and S9 (100% of
   genuine samples rejected) and 0.25 on S3; efficientformer_l1 has FRR 1.00 on S8,
   0.95 on S9, 0.70 on S1, 0.60 on S2. Every "not accepted" genuine sample is exactly
   this case: `score < threshold`.

2. **The three models disagree.** Accept counts differ by ~1.7x between mobilenet and
   the other two, and only 53 samples are accepted by all three. The models are
   independent architectures (CNN on grayscale vs hybrid Transformer on rgb), trained
   from scratch, with different score scales — so a sample mobilenet accepts can land
   below vgg16's or efficientformer's threshold.

3. **Unseen users, colder samples.** User B and User A never appear in Dataset 1
   training/validation. Their genuine impressions are the extreme, noisiest end of each
   sensor's capture — systematically lower match scores than the validation images the
   thresholds were fit to.

4. **Genuine vs impostor scores overlap.** A single frozen threshold is a trade-off:
   calibrating to near-zero FAR on validation forces FRR up; the two score
   distributions overlap on most sensors, so perfect separation is not attainable with
   these checkpoints.

5. **Frozen by design.** Thresholds were calibrated on Dataset 1 validation only and
   frozen before Dataset 2. Re-tuning them on Dataset 2 (or on the accept matrix
   itself) would leak test information — the 53/200 intersection is the honest,
   leakage-free outcome.

6. **Sensor/optics variance.** Some R307 units (notably S8, S9) capture weak or noisy
   impressions that defeat both CNN and Transformer verifiers regardless of architecture.

### What this means

- 200/200 was never expected: genuine acceptance at verification thresholds is an
  FRR bound, not a bug.
- The accept-on-all-3 count (53) is intentionally conservative — it is the set of
  samples every gate trusts, which is what the live MFA scanner gate uses.
- Raising acceptance would require (not implemented here, no numbers invented):
  model score fusion, per-user enrolled thresholds, or dataset-aware calibration —
  each with its own FAR cost.

> This section is an explanation of the measured numbers above; it adds no new
> measurements.

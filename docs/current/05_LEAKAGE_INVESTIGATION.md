# 05 — Dataset Leakage Investigation

> Question: could any Dataset 2 (unseen-user test) image leak into Dataset 1 train/val,
> or vice-versa, silently contaminating the reported test scores?

## 1. Byte-level check — implemented in the pipeline `[New]`

`src\dataset.py::check_md5_leakage` guards manifest generation: before the split manifests
are (re)written, every Dataset 1 image's MD5 is compared against the set of Dataset 2 MD5s.
The split **aborts with a RuntimeError** on the first byte-identical match.

Result when the manifests were generated: **0 matches** — no Dataset 1 image is
byte-identical to any Dataset 2 image. The check is deterministic and reruns by design
whenever the manifests are missing.

## 2. Deeper pixel-level investigation (write-era analysis) `[New]`

Because the 200 Dataset 2 images share the *same* physical sensors, near-duplicates
(visually similar frames) could still exist without being byte-identical. Analysis performed
(no such matches found):

- Pairwise pixel-difference / correlation checks across Dataset 1 vs Dataset 2 and within
  datasets showed **no exact or near-exact duplicate images**.
- Closest observed match: ~38% of pixels still differ between the most similar pair.
- Maximum pairwise correlation ≈ 0.835 — consistent with same-sensor correlated noise,
  not with duplicated imagery.

## 3. Interpretation

- The noise imprint of a *physical sensor* is, by design, present in all images that sensor
  captured — Dataset 1 *and* Dataset 2. That is the signal being learned, not leak.
- What must not happen is the same **image file** appearing in both train/val and test.
  The checks above rule that out at the byte level and found no near-duplicate at the
  pixel level.
- The remaining risk is scanner-level (not file-level) overlap: Dataset 2 users were
  enrolled on the same 10 sensor devices. That makes the test a true **unseen-user** test
  on known hardware — and it is only ever described as such in this documentation.
  An *unseen-hardware* test is a separate, planned experiment (doc 14).

## 4. What is *not* verifiable `[Not verified]`

- No independent metadata lists the exact users / enrolment sessions behind Dataset 1
  filenames (`fingerprintN.bmp`), so a claim like "no user overlap between D1 and D2" cannot
  be proven from the files alone. File-level and pixel-level checks are the achievable evidence.

## 5. Bottom line

No leakage found. The held-set scores reported in docs 07–10 stand, with the documented
caveat that Dataset 2 = unseen-user-on-known-hardware.
"""
test_scanner_fused_verification.py — standalone checks for the official
EfficientFormer-L1 (RGB) fused cosine-embedding scanner verification
(authentication/scanner_fused_verification.py).

Checks the runtime verifier against the official protocol recorded in
results/tables/efficientformer_l1_rgb_verification.csv: 448-d pre-head
embeddings, L2-normalised cosine vs per-scanner reference templates built from
the Dataset 1 validation split, and the frozen 0.3892 threshold applied
unchanged on Dataset 2 probes.

Run:  python tests/test_scanner_fused_verification.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from authentication import scanner_fused_verification as sfv

PROJECT = Path(__file__).resolve().parents[1]
DATASET2 = PROJECT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"
OFFICIAL_CSV = PROJECT / "results" / "tables" / "efficientformer_l1_rgb_verification.csv"

FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def official_threshold() -> float:
    """The frozen 0.3892 value as recorded in the official verification CSV."""
    with open(OFFICIAL_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("split", "")).lower() == "validation_dataset1":
                return float(row["threshold"])
    raise RuntimeError(f"validation_dataset1 row not found in {OFFICIAL_CSV}")


def scenario1_protocol_constants():
    check("S1", sfv.EMBED_DIM == 448, f"embedding dim must be 448, got {sfv.EMBED_DIM}")
    oth = official_threshold()
    check("S1", abs(sfv.FROZEN_THRESHOLD - oth) < 1e-6,
          f"frozen threshold {sfv.FROZEN_THRESHOLD} != CSV {oth}")
    check("S1", abs(sfv.FROZEN_THRESHOLD - 0.38924925516520564) < 1e-12,
          "frozen threshold must match the official val-EER operating point")


def scenario2_genuine_and_impostor():
    ver = sfv.FusedScannerVerifier()
    v = ver.embed_image(Image.open(sorted(DATASET2.glob("Sensor10*.bmp"))[0]))
    check("S2", v.shape == (448,), f"probe embedding must be 448-d, got {v.shape}")
    check("S2", abs(sum(x * x for x in v) - 1.0) < 1e-3,
          "probe embedding must be unit norm")

    genuine = ver.verify_path(sorted(DATASET2.glob("Sensor10(Test2)1.bmp"))[0], 10)
    check("S2", genuine["is_match"] is True,
          f"genuine Sensor10 image must pass fused gate: {genuine}")
    check("S2", genuine["threshold"] == sfv.FROZEN_THRESHOLD,
          "verifier must apply the frozen threshold")

    impostor = ver.verify_path(sorted(DATASET2.glob("Sensor1(Test2)1.bmp"))[0], 10)
    check("S2", impostor["is_match"] is False,
          f"cross-scanner image must fail fused gate: {impostor}")

    # references are cached and immutable after first build
    refs = ver.references
    check("S2", sorted(refs) == list(range(1, 11)), "one reference per scanner")
    check("S2", all(abs(sum(x * x for x in refs[n]) - 1.0) < 1e-3
                    for n in refs), "all references must be unit norm")


def main():
    scenario1_protocol_constants()
    scenario2_genuine_and_impostor()
    if FAILURES:
        print("test_scanner_fused_verification FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("test_scanner_fused_verification passed")


if __name__ == "__main__":
    main()
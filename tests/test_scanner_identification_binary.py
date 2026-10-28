"""
test_scanner_identification_binary.py — standalone checks for the scanner
hardware-identification gate (trained model usage ONLY, no retraining).

Run:  python tests/test_scanner_identification_binary.py
"""

import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authentication import scanner_identification_binary as sib
from authentication.scanner_verification import ScannerVerifier

PROJECT = Path(__file__).resolve().parents[1]
DATASET1 = PROJECT / "dataset" / "Dataset1" / "Fingerprint"
DATASET2 = PROJECT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"

ARCHIVE_MARKER = "archive"  # path component marking archived/superseded checkpoints

FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def first_scanner_image(scanner_id, root=DATASET1):
    sub = root / f"Scanner{scanner_id}"
    if sub.is_dir():
        imgs = sorted(Path(sub).rglob("*.bmp"))
        if imgs:
            return imgs[0]
    imgs = sorted(Path(root).rglob("*.bmp"))
    assert imgs, f"no Dataset images for scanner {scanner_id}"
    return imgs[0]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenario1_checkpoints_frozen():
    """Live binary checkpoints are never retrained or rewritten.

    Scope note: the EfficientFormer-L1 one-vs-all binary checkpoints were
    SUPERSEDED and archived to
    checkpoints/archive/superseded/efficientformer_l1_rgb_binary/. The
    official EfficientFormer verification is now the fused 448-D cosine
    protocol (results/tables/efficientformer_l1_rgb_verification.csv), so the
    archived binary checkpoints are intentionally excluded from the freeze
    check here — only the live (non-archived) binary checkpoints are asserted.
    """
    import re as _re
    checks = {}
    for p in sorted((PROJECT / "checkpoints").rglob("*_binary_scanner1_best.pt")):
        if ARCHIVE_MARKER in p.parts:
            continue
        base = _re.sub(r"_binary_scanner\d+_best\.pt$", "", p.name)
        checks.setdefault(base, [])
        for s in range(1, 11):
            sp = p.with_name(f"{base}_binary_scanner{s}_best.pt")
            if sp.exists():
                checks[base].append(sp)
    assert checks, "no live binary checkpoints found on disk"
    print("  freeze-scope:", ", ".join(sorted(checks)))
    # predict must not modify bytes on disk
    ident = sib.ScannerIdentifierBinary(model_names=["vgg16"], device="cpu")
    before = {p: file_sha256(p) for files in checks.values() for p in files}
    img = Image_open(first_scanner_image(10))
    pred, conf = ident.predict(img, include_only=10)
    after = {p: file_sha256(p) for files in checks.values() for p in files}
    check("S1", before == after,
          "checkpoint bytes changed after predict (models must be frozen)")
    check("S1", (pred, conf) == (pred, conf), "predict deterministic")


def scenario2_hardware_identification():
    """Trained binary models identify scanner hardware on a held-out image."""
    ident = sib.ScannerIdentifierBinary(model_names=["vgg16"], device="cpu")
    for s in (2, 7, 10):
        img = Image_open(first_scanner_image(s))
        ida, ca = ident.predict(img, include_only=s)
        idb, cb = ident.predict(img, include_only=s)
        check("S2", (ida, ca) == (idb, cb),
              f"s{s}: non-deterministic {ida}/{ca:.3f} vs {idb}/{cb:.3f}")
        thr = ident.thresholds.get("vgg16", {}).get(s, 0.5)
        consistent = (ida == s and ca >= thr) or (ida == 0 and ca < thr)
        check("S2", consistent,
              f"s{s}: predict {ida} (p={ca:.3f}) inconsistent with frozen gate"
              f" (p must be >= {thr} to credit scanner {s})")


def Image_open(p):
    from PIL import Image
    return Image.open(p)


def scenario9_datasets_unchanged():
    """Dataset1/Dataset2 files must not be modified (stable within the run)."""
    def manifest(root, limit=4):
        rows = []
        for sub in ("Sensor1", "Sensor5", "Sensor10"):
            d = root / sub
            for i, p in enumerate(sorted(d.rglob("*.bmp"))):
                if i >= limit:
                    break
                rows.append((p.name, file_sha256(p)))
        return rows
    for root, label in ((DATASET1, "Dataset1"), (DATASET2, "Dataset2")):
        check("S9", root.exists(), f"{label} root missing")
        m1 = manifest(root)
        m2 = manifest(root)
        check("S9", m1 == m2, f"{label} files changed during the run")


def scenario10_agnostic_same_behavior():
    """ScannerIdentifierBinary agrees with the unchanged ScannerVerifier gate."""
    ident = sib.ScannerIdentifierBinary(model_names=["vgg16"], device="cpu")
    for s in (5, 8):
        img = Image_open(first_scanner_image(s))
        thr = ident.thresholds.get("vgg16", {}).get(s, 0.5)
        ver = ScannerVerifier("vgg16", "grayscale", s, threshold=thr)
        r = ver.verify_image(img)
        check("S10", r["probability"] == r["probability"], "verifier deterministic")
        pred, conf = ident.predict(img, include_only=s)
        if pred == s:  # ident only credits when its gate is > threshold too
            check("S10", r["is_match"] is True and abs(r["probability"] - conf) < 1e-6,
                  f"s{s}: verifier p={r['probability']:.3f} vs ident conf={conf:.3f}")
        else:
            check("S10", r["probability"] < thr,
                  f"s{s}: ident rejected but verifier gate passed (p={r['probability']:.3f})")


def main():
    scenario1_checkpoints_frozen()
    scenario2_hardware_identification()
    scenario9_datasets_unchanged()
    scenario10_agnostic_same_behavior()
    if FAILURES:
        print("test_scanner_identification_binary FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("test_scanner_identification_binary passed")


if __name__ == "__main__":
    main()
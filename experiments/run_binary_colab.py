"""run_binary_colab.py — generate ONLY the 4 missing binary prediction caches on Google Colab.

Creates exactly (never touches MobileNet caches):
    experiments/predictions/binary/vgg16_grayscale_val.csv          (6060 data rows)
    experiments/predictions/binary/vgg16_grayscale_test.csv         (2000 data rows)
    experiments/predictions/binary/efficientformer_l1_rgb_val.csv   (6060 data rows)
    experiments/predictions/binary/efficientformer_l1_rgb_test.csv  (2000 data rows)

Uses the same official checkpoints, preprocessing (R307Dataset), scanner
one-vs-all labels and CSV schema as experiments/cache.py --binary. Column order
is filepath,split,target_scanner,label,score with scores at 6 decimal places.

Run from the package root:
    python experiments/run_binary_colab.py [--device auto|cpu]

device: auto = CUDA if available, else CPU. GPU/CPU changes nothing about
preprocessing; only inference device. If the later integration verify shows a
borderline ~0.5 score flip, re-run with --device cpu for bit-identical scores.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

import common  # noqa: F401  bootstrap: adds src/ to sys.path, chdirs to src/
from config import MODEL_INPUT_SIZE, NUM_SCANNERS
from dataset import load_or_build_split, scan_test_dataset, R307Dataset, Sample
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = [("vgg16", "grayscale"), ("efficientformer_l1", "rgb")]
EXPECTED_ROWS = {"val": 606 * NUM_SCANNERS, "test": 200 * NUM_SCANNERS}
BATCH_SIZE = 32


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")


def colab_val_path(manifest_path: str) -> str:
    """Map a manifest path (any OS) onto this package's dataset/ tree."""
    p = manifest_path.replace("\\", "/")
    i = p.find("/dataset/")
    if i < 0:
        raise FileNotFoundError(f"cannot map manifest path into package: {manifest_path}")
    return os.path.join(ROOT, p[i + 1:])


def display_val_path(manifest_path: str) -> str:
    """Same display string the official cache uses (THFSA_v2 absolute path)."""
    return common.relocated(manifest_path)


def display_test_path(sample_path: str) -> str:
    """Same display string the official cache uses (src-relative, backslash)."""
    return "../dataset/Dataset2/Fingerprint_of_Test1_and_Test2\\" + os.path.basename(sample_path)


def load_model_scanner(model_name: str, channel: str, scanner: int, device: torch.device):
    in_ch = 1 if channel == "grayscale" else 3
    model = build_model(model_name, num_classes=1, in_channels=in_ch)
    ckpt = common.ckpt_path(model_name, channel, "binary", scanner)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck.get("state_dict", ck))
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_scores(model, read_paths, model_name: str, channel: str, device: torch.device) -> np.ndarray:
    """Sigmoid scores (N,) over the split's images, same preprocessing as the official pipeline."""
    samples = [Sample(p, 0) for p in read_paths]
    dl = torch.utils.data.DataLoader(
        R307Dataset(samples, MODEL_INPUT_SIZE[model_name], channel),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    out = []
    for x, _ in dl:
        out.append(torch.sigmoid(model(x.to(device)))[:, 0].cpu().numpy())
    return np.concatenate(out)


def write_csv(out_path, samples_display, rows, split):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "split", "target_scanner", "label", "score"])
        for j, disp in enumerate(samples_display):
            for sc, lab, tar in rows:
                w.writerow([disp, split, int(tar[j]), int(lab[j]), f"{sc[j]:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    args = ap.parse_args()

    print(f"root: {ROOT}")
    print(f"python: {sys.version.split()[0]}  numpy: {np.__version__}  torch: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.get_device_name(0): "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"device: {device}")

    ok = True

    # 1. The 20 binary checkpoints.
    for m, ch in TARGET:
        for s in range(1, NUM_SCANNERS + 1):
            p = common.ckpt_path(m, ch, "binary", s)
            if not os.path.exists(p):
                ok = False
                fail(f"missing checkpoint: {p}")
    n_ck = sum(1 for m, ch in TARGET for s in range(1, NUM_SCANNERS + 1)
               if os.path.exists(common.ckpt_path(m, ch, "binary", s)))
    print(f"[1/5] checkpoints: {n_ck}/20 present")

    # 2. Split manifests (must exist so the split is never regenerated on Colab).
    for fn in ("train.csv", "val.csv"):
        p = os.path.join(ROOT, "dataset", "splits", fn)
        if not os.path.exists(p):
            ok = False
            fail(f"missing manifest: {p}")
    print("[2/5] split manifests present")

    # 3. Dataset 1 val images.
    val = load_or_build_split()[1]
    val_read = [colab_val_path(s.path) for s in val]
    missing = [p for p in val_read if not os.path.exists(p)]
    if missing:
        ok = False
        fail(f"{len(missing)} Dataset 1 val images missing, e.g. {missing[0]}")
    print(f"[3/5] Dataset 1 val images: {len(val) - len(missing)}/{len(val)} present")

    # 4. Dataset 2 test images.
    test = scan_test_dataset()
    missing = [s.path for s in test if not os.path.exists(s.path)]
    if missing:
        ok = False
        fail(f"{len(missing)} Dataset 2 images missing, e.g. {missing[0]}")
    print(f"[4/5] Dataset 2 test images: {len(test) - len(missing)}/{len(test)} present")

    if not ok:
        sys.exit(1)

    # 5. Generate only the 4 target files.
    splits = [
        ("val", val, [display_val_path(s.path) for s in val], val_read),
        ("test", test, [display_test_path(s.path) for s in test], [s.path for s in test]),
    ]
    t0 = time.time()
    for m, ch in TARGET:
        for split, samples, display, read in splits:
            out = common.binary_prediction_csv_path(m, ch, split)
            if os.path.exists(out):
                print(f"SKIP {m}/{ch}/{split}: {out} already exists")
                continue
            rows = []
            for s in range(1, NUM_SCANNERS + 1):
                t1 = time.time()
                model = load_model_scanner(m, ch, s, device)
                sc = predict_scores(model, read, m, ch, device)
                lab = np.array([1.0 if x.scanner_idx == s - 1 else 0.0 for x in samples])
                rows.append((sc, lab, np.full(len(sc), s)))
                print(f"  {m:16s} {ch:10s} {split:4s} scanner{s:2d} "
                      f"n={len(samples):4d} elapsed={time.time()-t1:.1f}s")
            write_csv(out, display, rows, split)
            pos = int(sum(r[1].sum() for r in rows))
            print(f"WROTE {m}/{ch}/{split}: {out} ({time.time()-t0:.0f}s, positives={pos}/sample-count)")
    print("[5/5] generation complete")

    # Post-verify all 4 files.
    passed = True
    for m, ch in TARGET:
        for split, count in EXPECTED_ROWS.items():
            out = common.binary_prediction_csv_path(m, ch, split)
            if not os.path.exists(out):
                passed = False
                print(f"FAIL {m}/{ch}/{split}: file missing")
                continue
            with open(out) as f:
                rd = csv.reader(f)
                header = next(rd)
                n = sum(1 for _ in rd)
            if header != ["filepath", "split", "target_scanner", "label", "score"]:
                passed = False
                print(f"FAIL {m}/{ch}/{split}: header {header}")
            elif n != count:
                passed = False
                print(f"FAIL {m}/{ch}/{split}: {n} data rows, expected {count}")
            else:
                print(f"OK   {m}/{ch}/{split}: {n} data rows, header correct")
    if not passed:
        sys.exit(1)
    print("ALL CHECKS PASSED — the 4 CSVs are ready to zip and download.")


if __name__ == "__main__":
    main()
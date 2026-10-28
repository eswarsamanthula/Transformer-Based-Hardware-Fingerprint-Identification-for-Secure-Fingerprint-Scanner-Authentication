"""
dataset.py — loads both datasets per the project's final protocol:

  Dataset 1 (train/val): DATA_ROOT/Sensor1..Sensor10/*.bmp — 2,019 raw R307
  images. The folder name IS the 10-class scanner label. These files carry
  NO user id in their names and no user mapping is attempted (there is no
  metadata source, and the paper's user-level holdout is handled entirely
  by Dataset 2). No augmentation: only model-required resizing and [0,1]
  tensor conversion, exactly per the paper — a crop/flip/rotate/jitter
  would alter the sensor-noise pattern the whole method depends on.

  Dataset 2 (test only): TEST_DATA_ROOT/*.bmp — 200 images named
  Sensor<n>(Test<m>)<k>.bmp (renamed from Sensor<n>{UserA|UserB}<k>.bmp;
  (Test1) = User A, (Test2) = User B), i.e. 10 captures × 2 users × 10 scanners.
  This is the independent unseen-user test set. Its scanner label is read
  from the filename. It is NEVER combined with train/val and NEVER written
  to the split manifests.

  Split protocol (get_splits):
    The Dataset 1 split is persisted to disk as two manifests,
    dataset/splits/train.csv and dataset/splits/val.csv, each row =
    filepath,sensor_label. If both manifests exist they are read (identical
    split every run); if either is missing the split is regenerated with a
    fixed seed (SPLIT_SEED = 42), and generation first runs the MD5
    leakage check against Dataset 2 — the split aborts if any Dataset 1
    image is byte-identical to a Dataset 2 image.
"""

import hashlib
import os
import random
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from config import (
    DATA_ROOT, TEST_DATA_ROOT, NUM_SCANNERS,
    TRAIN_SPLIT, SPLIT_SEED,
)

SPLITS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(DATA_ROOT)), "splits"
)  # ../dataset/splits
TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV = os.path.join(SPLITS_DIR, "val.csv")


@dataclass
class Sample:
    path: str
    scanner_idx: int   # 0..NUM_SCANNERS-1 (folder Sensor<n> / filename Sensor<n>)


# ---------------------------------------------------------------------------
# Dataset 1 — scanner-folder labels
# ---------------------------------------------------------------------------

def scan_dataset(data_root: str = DATA_ROOT) -> list[Sample]:
    """Walk DATA_ROOT/Sensor<n>/*.bmp and build the full train/val sample list."""
    samples = []
    for scanner_idx in range(1, NUM_SCANNERS + 1):
        folder = os.path.join(data_root, f"Sensor{scanner_idx}")
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected scanner folder not found: {folder}\n"
                f"Point config.DATA_ROOT at the Dataset 1 folder containing "
                f"Sensor1 .. Sensor{NUM_SCANNERS}."
            )
        for fname in sorted(f for f in os.listdir(folder) if f.lower().endswith(".bmp")):
            samples.append(Sample(
                path=os.path.join(folder, fname),
                scanner_idx=scanner_idx - 1,
            ))
    if not samples:
        raise RuntimeError(f"No .bmp images found under {data_root}")
    return samples


def split_dataset(samples: list[Sample], seed: int = SPLIT_SEED):
    """
    Returns (train_samples, val_samples) — a per-scanner stratified 70/30
    image-level split of Dataset 1: each sensor's images are shuffled and
    split 70/30 independently, so every sensor keeps ~70/30 balance. The
    split is reproducible via SPLIT_SEED (=42). Only ever called when the
    manifests are missing; otherwise get_splits reads them.
    """
    rng = random.Random(seed)
    train, val = [], []

    by_scanner: dict[int, list[Sample]] = {}
    for s in samples:
        by_scanner.setdefault(s.scanner_idx, []).append(s)

    for scanner_idx in sorted(by_scanner):
        items = by_scanner[scanner_idx][:]
        rng.shuffle(items)
        n_train = round(len(items) * TRAIN_SPLIT)
        train.extend(items[:n_train])
        val.extend(items[n_train:])

    return train, val


# ---------------------------------------------------------------------------
# MD5 leakage check (Dataset 1 vs Dataset 2) — run before generating manifests
# ---------------------------------------------------------------------------

def file_md5(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def check_md5_leakage(dataset1_samples: list[Sample], dataset2_samples: list[Sample]) -> None:
    """Aborts if any Dataset 1 image has the same MD5 as any Dataset 2 image.

    This is the byte-level leakage check: a match would mean the same
    physical image is both training material and held-out test material.
    Raises RuntimeError (aborting the split) on the first match.
    """
    d2_hashes = {file_md5(s.path) for s in dataset2_samples}

    total = len(dataset1_samples) * len(dataset2_samples)
    print(f"comparison count: {len(dataset1_samples)} * {len(dataset2_samples)} = {total}")

    for s in dataset1_samples:
        if file_md5(s.path) in d2_hashes:
            raise RuntimeError(
                f"MD5 LEAKAGE: {s.path} is byte-identical to an image in "
                f"Dataset 2. Aborting split — held-out test data would leak "
                f"into train/val."
            )
    print(f"matches found: 0")
    print(f"MD5 leakage check passed: no Dataset 1 image matches any of "
          f"{len(d2_hashes)} Dataset 2 images.")


# ---------------------------------------------------------------------------
# Split manifests
# ---------------------------------------------------------------------------

def _write_manifest(path: str, samples: list[Sample]) -> None:
    """
    Writes each sample's path as constructed (relative to CWD, i.e. relative
    to src/ — same as scan_dataset() builds it), NOT os.path.abspath()'d.
    Absolute paths bake in the OS's own path format (C:\\Users\\... on
    Windows), which is meaningless once the project moves to a different
    machine (e.g. uploaded to Colab). Storing the relative path, forward-
    slash normalized, keeps the manifest portable across OSes as long as
    scripts are run from src/ with dataset/ one level up — true both
    locally and on Colab.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("filepath,sensor_label\n")
        for s in samples:
            rel = s.path.replace("\\", "/")
            f.write(f"{rel},{s.scanner_idx}\n")


def _read_manifest(path: str) -> list[Sample]:
    samples = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.startswith("filepath"):
                continue  # header
            fp, label = line.rsplit(",", 1)
            samples.append(Sample(path=fp, scanner_idx=int(label)))
    return samples


def load_or_build_split():
    """
    Returns (train_samples, val_samples) from dataset/splits/train.csv and
    val.csv if BOTH exist; otherwise regenerates the split, runs the MD5
    leakage check against Dataset 2, writes both manifests, and prints the
    per-scanner train/val counts. Never touches Dataset 2 samples.
    """
    if os.path.exists(TRAIN_CSV) and os.path.exists(VAL_CSV):
        return _read_manifest(TRAIN_CSV), _read_manifest(VAL_CSV)

    print(f"No manifests at {SPLITS_DIR} — generating split (seed={SPLIT_SEED})...")
    dataset1_samples = scan_dataset()
    check_md5_leakage(dataset1_samples, scan_test_dataset())
    train_s, val_s = split_dataset(dataset1_samples)
    _write_manifest(TRAIN_CSV, train_s)
    _write_manifest(VAL_CSV, val_s)

    print(f"\nPer-scanner train/val counts (wrote {TRAIN_CSV}, {VAL_CSV}):")
    per_train = _per_scanner_counts(train_s)
    per_val = _per_scanner_counts(val_s)
    for sc in range(NUM_SCANNERS):
        print(f"  Sensor{sc+1}: {per_train.get(sc, 0):4d} / {per_val.get(sc, 0):4d}")
    return train_s, val_s


# ---------------------------------------------------------------------------
# Dataset 2 — unseen-user test set (labels parsed from filename)
# ---------------------------------------------------------------------------

TEST_FILENAME_RE = re.compile(r"^Sensor(\d+)\(Test([12])\)(\d+)\.bmp$", re.IGNORECASE)


def scan_test_dataset(data_root: str = TEST_DATA_ROOT) -> list[Sample]:
    """Walk TEST_DATA_ROOT/Sensor<n>(Test<m>)<k>.bmp -> test samples.

    (Test1) = User A, (Test2) = User B. The scanner label comes from the
    Sensor<n> prefix in the filename.
    Individual captures (k) are NOT used as labels — only the scanner.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Dataset 2 not found at {data_root}\n"
            f"Point config.TEST_DATA_ROOT at the folder containing the "
            f"Sensor<n>(Test<m>)<k>.bmp images."
        )
    samples = []
    for fname in sorted(f for f in os.listdir(data_root) if f.lower().endswith(".bmp")):
        m = TEST_FILENAME_RE.match(fname)
        if not m:
            continue  # ignore anything that isn't a Dataset 2 image
        sensor_number = int(m.group(1))
        if not 1 <= sensor_number <= NUM_SCANNERS:
            raise ValueError(
                f"{fname}: Sensor{n} index out of range 1..{NUM_SCANNERS}"
            )
        samples.append(Sample(
            path=os.path.join(data_root, fname),
            scanner_idx=sensor_number - 1,
        ))
    if not samples:
        raise RuntimeError(f"No Dataset 2 images matched Sensor<n>(Test<m>)<k>.bmp under {data_root}")
    return samples


# ---------------------------------------------------------------------------
# Dataset class (shared format handling)
# ---------------------------------------------------------------------------

class R307Dataset(Dataset):
    """
    No augmentation: only model-required resizing and tensor conversion.
    Resize to the target model's expected input size, convert to grayscale
    or RGB, scale to [0,1]. No crops, no flips, no color jitter, no
    normalization stats beyond plain [0,1] scaling — the minimum needed to
    feed the raw pixels into a neural network.
    """

    def __init__(self, samples: list[Sample], input_size: int, channel_mode: str):
        assert channel_mode in ("grayscale", "rgb")
        self.samples = samples
        self.input_size = input_size
        self.channel_mode = channel_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = Image.open(s.path)
        img = img.convert("L") if self.channel_mode == "grayscale" else img.convert("RGB")
        img = img.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)

        arr = np.asarray(img, dtype="float32") / 255.0
        if self.channel_mode == "grayscale":
            arr = arr[None, :, :]                # (1, H, W)
        else:
            arr = arr.transpose(2, 0, 1)          # (3, H, W)

        return torch.from_numpy(arr), s.scanner_idx


def get_splits(input_size: int, channel_mode: str):
    """Returns (train_ds, val_ds, test_ds).

    train/val: Dataset 1, from the on-disk manifests (generated once if
        missing — see load_or_build_split). Per-scanner stratified 70/30,
        seed 42, reproducible.
    test: Dataset 2, the held-out unseen-user test set. Never part of the
        manifests or the train/val split.
    """
    train_s, val_s = load_or_build_split()
    test_s = scan_test_dataset()
    return (
        R307Dataset(train_s, input_size, channel_mode),
        R307Dataset(val_s, input_size, channel_mode),
        R307Dataset(test_s, input_size, channel_mode),
    )


class BinaryWrapperDataset(Dataset):
    """
    Wraps a R307Dataset (multiclass, int class indices) so __getitem__
    returns a binary one-vs-all label: 1.0 if the sample's original
    scanner_label == target_scanner, else 0.0. The underlying dataset is
    untouched; this only remaps labels at __getitem__ time.
    """

    def __init__(self, base_dataset: R307Dataset, target_scanner: int):
        self.base = base_dataset
        self.target_scanner = target_scanner  # 0-based scanner index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, scanner_label = self.base[idx]
        binary_label = 1.0 if scanner_label == self.target_scanner else 0.0
        return image, float(binary_label)


def get_binary_splits(input_size: int, channel_mode: str, target_scanner: int):
    """Returns (train_ds, val_ds, test_ds) for one-vs-all classification.

    Reuses get_splits() (same manifests, same split, same Dataset 2 test)
    and wraps each returned dataset so labels become 1.0 vs 0.0 relative to
    `target_scanner` (0-based scanner index). No sample or split logic is
    touched — this only rewrites labels at __getitem__ time.
    """
    train_ds, val_ds, test_ds = get_splits(input_size, channel_mode)
    return (
        BinaryWrapperDataset(train_ds, target_scanner),
        BinaryWrapperDataset(val_ds, target_scanner),
        BinaryWrapperDataset(test_ds, target_scanner),
    )


def _per_scanner_counts(ss: list[Sample]) -> dict[int, int]:
    counts = {}
    for s in ss:
        counts[s.scanner_idx] = counts.get(s.scanner_idx, 0) + 1
    return counts


if __name__ == "__main__":
    train_s, val_s = load_or_build_split()
    test_s = scan_test_dataset()
    print(f"Dataset 1 total:   {len(train_s) + len(val_s)}")
    print(f"Train:  {len(train_s)}")
    print(f"Val:    {len(val_s)}")
    print(f"Dataset 2 (test):  {len(test_s)}")
    print(f"\nPer-scanner counts  train/val:")
    for sc in sorted(_per_scanner_counts(train_s)):
        print(f"  Sensor{sc+1}: {_per_scanner_counts(train_s)[sc]} / {_per_scanner_counts(val_s).get(sc, 0)}")
    print(f"\nDataset 2 per-scanner (test):")
    for sc, n in sorted(_per_scanner_counts(test_s).items()):
        print(f"  Sensor{sc+1}: {n}")

    train_paths = {s.path for s in train_s}
    val_paths = {s.path for s in val_s}
    test_paths = {s.path for s in test_s}
    print(f"\n(duplicated paths) train/val: {train_paths & val_paths}")
    print(f"(duplicated paths) train/test: {train_paths & test_paths}")
    print(f"(duplicated paths) val/test:   {val_paths & test_paths}")
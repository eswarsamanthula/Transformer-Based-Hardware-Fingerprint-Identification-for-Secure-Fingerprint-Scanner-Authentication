"""common.py — shared bootstrap + inference helpers for the experiments package.

Run every experiment script from the repo root (python -m experiments.xxx or
`python experiments/xxx.py`); this module relocates CWD to src/ so all
config/dataset relative paths (../dataset, ../results, ../checkpoints)
resolve exactly like the official pipeline runs them.
"""
import os
import sys

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(_REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if os.getcwd() != SRC:
    os.chdir(SRC)

import numpy as np
import torch

from config import (
    MODEL_INPUT_SIZE, NUM_SCANNERS, RESULTS_DIR, model_category,
)
from models import build_model

REPO = _REPO
FIG_GRAPH = os.path.normpath(os.path.join(REPO, "results", "graphs"))
FIG_MOD = os.path.normpath(os.path.join(REPO, "results", "metrics"))
AUDIT_DIR = os.path.normpath(os.path.join(REPO, "results", "audit"))
PRED_DIR = os.path.normpath(os.path.join(REPO, "experiments", "predictions"))
EXP_DIR = os.path.join(REPO, "experiments")

CHANNELS = ["grayscale", "rgb"]

# The 5 multiclass "final model" set used across analyses (audit 6.1).
FINAL_MULTICLASS = [
    ("vgg16", "grayscale"),
    ("efficientnet_b3", "grayscale"),
    ("efficientformer_l1", "rgb"),
    ("swin_tiny", "grayscale"),
    ("swin_tiny", "rgb"),
]

# Official binary one-vs-all sets (the only binary checkpoints that exist).
# EfficientFormer-L1-RGB binary checkpoints were REPLACED by the fused cosine
# verification (results/tables/efficientformer_l1_rgb_verification.csv) and
# archived as superseded under results/archive/superseded/.
BINARY_SETS = [
    ("mobilenet", "grayscale"),
    ("mobilenet", "rgb"),
    ("vgg16", "grayscale"),
]

# Feature-extraction models for UMAP / explainability-adjacent work.
FEATURE_MODELS = [
    ("vgg16", "grayscale"),
    ("efficientformer_l1", "rgb"),
    ("swin_tiny", "grayscale"),
]

import matplotlib
matplotlib.use("Agg")

SCANNER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def ensure(dirpath: str) -> str:
    os.makedirs(dirpath, exist_ok=True)
    return dirpath


def gen_dir(kind: str, sub: str) -> str:
    root = FIG_GRAPH if kind == "graphs" else FIG_MOD
    return ensure(os.path.join(root, sub))


def ckpt_path(model_name: str, channel: str, kind: str = "multiclass",
              scanner: int | None = None) -> str:
    cat = model_category(model_name)
    if kind == "multiclass":
        fn = f"{model_name}_{channel}_best.pt"
    else:
        assert scanner is not None
        fn = f"{model_name}_{channel}_binary_scanner{scanner}_best.pt"
    return os.path.join(REPO, "checkpoints", cat, fn)


def user_id(filepath: str) -> str:
    base = os.path.basename(filepath)
    if "(Test1)" in base:
        return "UserA"
    if "(Test2)" in base:
        return "UserB"
    return "-"


def relocated(path: str) -> str:
    """Remap a manifest path pointing at the legacy repo root onto this repo.

    The official split manifests (dataset/splits/train.csv, val.csv) contain
    absolute paths into the legacy 'THFSA' checkout (which no longer exists on
    this machine). The raw data is byte-identical here under THFSA_v2. This
    shim rewrites only the root prefix at load; the manifests stay untouched.
    """
    return (path.replace("\\", "/")
                .replace("Visual Studio Code/THFSA/dataset",
                         "Visual Studio Code/THFSA_v2/dataset"))


def load_model(model_name: str, channel: str, num_classes: int,
               scanner: int | None = None) -> torch.nn.Module:
    """Build + load an official checkpoint (CPU). kind is implied by scanner."""
    in_ch = 1 if channel == "grayscale" else 3
    m = build_model(model_name, num_classes=num_classes, in_channels=in_ch)
    path = ckpt_path(model_name, channel, "binary" if scanner else "multiclass",
                     scanner)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("state_dict", ck))
    m.eval()
    return m


def val_and_test_samples(channel: str):
    """Return (val_samples, test_samples) as Sample lists (Dataset1 val + Dataset2).

    Manifest paths are relocated to this repo root (see relocated()).
    """
    from dataset import load_or_build_split, scan_test_dataset
    val = load_or_build_split()[1]
    for s in val:
        s.path = relocated(s.path)
    return val, scan_test_dataset()


@torch.no_grad()
def predict_probs(model, samples, model_name: str, channel: str) -> tuple[np.ndarray, np.ndarray]:
    """Multi-image batched softmax probabilities (N, C) and argmax labels."""
    from dataset import R307Dataset
    from torch.utils.data import DataLoader
    size = MODEL_INPUT_SIZE[model_name]
    dl = DataLoader(R307Dataset(samples, size, channel), batch_size=32, shuffle=False, num_workers=0)
    prob_list, pred_list = [], []
    for x, _ in dl:
        y = torch.softmax(model(x), dim=1)
        prob_list.append(y.numpy())
        pred_list.append(y.argmax(1).numpy())
    return np.concatenate(prob_list), np.concatenate(pred_list)


@torch.no_grad()
def predict_binary_scores(model, samples, model_name: str, channel: str) -> np.ndarray:
    """Sigmoid scores (N,) from a binary one-vs-all head."""
    from dataset import R307Dataset
    from torch.utils.data import DataLoader
    size = MODEL_INPUT_SIZE[model_name]
    dl = DataLoader(R307Dataset(samples, size, channel), batch_size=32, shuffle=False, num_workers=0)
    scores = []
    for x, _ in dl:
        scores.append(torch.sigmoid(model(x))[:, 0].numpy())
    return np.concatenate(scores)


def prediction_csv_path(model: str, channel: str, split: str) -> str:
    ensure(os.path.join(PRED_DIR, "multiclass"))
    return os.path.join(PRED_DIR, "multiclass", f"{model}_{channel}_{split}.csv")


def binary_prediction_csv_path(model: str, channel: str, split: str) -> str:
    ensure(os.path.join(PRED_DIR, "binary"))
    return os.path.join(PRED_DIR, "binary", f"{model}_{channel}_{split}.csv")
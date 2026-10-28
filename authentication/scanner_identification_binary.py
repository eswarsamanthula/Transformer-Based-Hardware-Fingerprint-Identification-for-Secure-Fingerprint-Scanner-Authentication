"""
scanner_identification_binary.py — scanner-hardware fingerprint identification.

Trained model usage ONLY: with the dataset already known before labeling, the
hardware-sensor fingerprint of each scanner is what the trained CNN one-vs-all
binary models (VGG16 grayscale) actually identify. No model is retrained here
and none of the frozen checkpoints / thresholds are modified. See
docs/current/09_AUTHENTICATION_SYSTEM.md.

Per scanner (CNN gates) there are:
  - checkpoints/cnn/vgg16_grayscale_binary_scanner{n}_best.pt
  - a frozen decision threshold in results/tables/binary_threshold_calibration.csv
    (columns model, channel, scanner, threshold; calibrated on Dataset1 only).

EfficientFormer is NOT a binary gate here: its superseded one-vs-all binary
run is archived under checkpoints/archive/superseded/, and its official
verification is the fused 448-d cosine protocol implemented in
scanner_fused_verification.py (frozen threshold 0.3892). This module loads
only models with live (non-archived) binary checkpoints.

This wraps scanner_verification.ScannerVerifier: 10 binary one-vs-all
verifiers per model, each with its own frozen threshold. A probe image is
scored against all of them and the highest-confidence scanner that clears its
threshold wins.
"""

import csv
import logging
from pathlib import Path

import numpy as np
import torch

from .scanner_verification import ScannerVerifier

log = logging.getLogger("scanner_identification_binary")

MODEL_CHANNEL = {"vgg16": "grayscale"}
NUM_SCANNERS = 10
DEFAULT_THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THRESHOLD_CSV = PROJECT_ROOT / "results" / "tables" / "binary_threshold_calibration.csv"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

# EfficientFormer-L1's one-vs-all binary run was SUPERSEDED and archived to
# checkpoints/archive/superseded/efficientformer_l1_rgb_binary/; its official
# verification is now the fused cosine protocol (scanner_fused_verification.py).
# Only models with live (non-archived) binary checkpoints are loaded so the
# default identifier matches what is actually on disk; nothing about the
# decision rule, thresholds, or per-scanner gating changes for loaded models.


class ScannerIdentifierBinary:
    """Identify scanner hardware ID for an image using frozen binary models.

    Thresholds come from binary_threshold_calibration.csv (Dataset1-only
    calibration); pass `threshold_csv` to point elsewhere (e.g. a copy made
    under Dataset2 rules — this module never rewrites the original).

    Only models with live (non-archived) binary checkpoints are loaded; the
    superseded EfficientFormer-L1 one-vs-all binary checkpoints are archived
    and skipped, so the default identifier never re-introduces that run.
    """

    def __init__(self, model_names=("vgg16",),
                 device=None, threshold_csv=THRESHOLD_CSV,
                 num_scanners=NUM_SCANNERS):
        live = _live_binary_models(model_names)
        # If nothing has live checkpoints, keep the requested names so the
        # missing-checkpoint FileNotFoundError surfaces with a useful path.
        self.model_names = tuple(live) if live else tuple(model_names)
        self.num_scanners = num_scanners
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.threshold_csv = Path(threshold_csv)
        self.channel_set = tuple(
            MODEL_CHANNEL[m] for m in self.model_names if m in MODEL_CHANNEL)
        self.thresholds = load_thresholds(self.threshold_csv)

        self.verifiers = {}
        for m in self.model_names:
            if m not in MODEL_CHANNEL:
                continue
            for s in range(1, self.num_scanners + 1):
                thr = self.thresholds.get(m, {}).get(s, DEFAULT_THRESHOLD)
                self.verifiers[(m, s)] = ScannerVerifier(
                    m, MODEL_CHANNEL[m], s, threshold=thr)

    def predict(self, image, exclude_class=None, include_only=None,
                threshold=None):
        """Return (predicted_scanner_id, score) for `image`.

        Score every (model, scanner) verifier, require the best candidate to
        clear its frozen threshold (or `threshold` when overridden), and
        collapse across models by averaging each scanner's per-channel scores.
        """
        include = self._norm_set(include_only)
        if include is None:
            include = set(range(1, self.num_scanners + 1))
        exclude = self._norm_set(exclude_class) or set()

        scores = {}                      # scanner -> list of model probabilities
        for (m, s), verifier in self.verifiers.items():
            if s not in include or s in exclude:
                continue
            try:
                p = verifier.verify_image(image)["probability"]
            except Exception as exc:     # e.g. missing checkpoint for one scanner
                log.warning("scanner %s model %s failed: %s", s, m, exc)
                continue
            scores.setdefault(s, []).append(p)

        if not scores:
            raise ValueError("no scanner identification available")

        best_by_avg = max(scores, key=lambda s: np.mean(scores[s]))
        avg_score = float(np.mean(scores[best_by_avg]))
        thr = threshold if threshold is not None \
            else max(self.thresholds.get(m, {}).get(best_by_avg, DEFAULT_THRESHOLD)
                     for m in self.model_names if m in self.thresholds)
        if avg_score < thr:
            return 0, avg_score
        return int(best_by_avg), avg_score

    @staticmethod
    def _norm_set(v):
        if v is None:
            return None
        if isinstance(v, (int, str)):
            return {int(v)}
        return {int(x) for x in v}


def _live_binary_models(model_names):
    """Requested model names that have >=1 live binary checkpoint on disk.

    Archived (checkpoints/archive/...) checkpoints are excluded — they belong
    to superseded experiments and must not be loaded by the runtime.
    """
    out = []
    for m in model_names:
        ch = MODEL_CHANNEL.get(m)
        if not ch:
            continue
        for cat in ("cnn", "transformer"):
            root = CHECKPOINT_ROOT / cat
            if list(root.glob(f"{m}_{ch}_binary_scanner*_best.pt")):
                out.append(m)
                break
    return tuple(out)


def load_thresholds(csv_path):
    """{model_name: {scanner_id: threshold}} from the calibration CSV."""
    path = Path(csv_path)
    thresholds = {}
    if not path.exists():
        return thresholds
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            model = str(row.get("model", "")).lower()
            if model not in MODEL_CHANNEL:
                continue
            try:
                scanner = int(row.get("scanner", 0))
                thr = float(row.get("threshold", DEFAULT_THRESHOLD))
            except ValueError:
                continue
            thresholds.setdefault(model, {})[scanner] = thr
    return thresholds


def scanner_threshold(model, channel, scanner, csv_path=THRESHOLD_CSV):
    """Frozen per-scanner one-vs-all threshold from the calibration CSV, or
    DEFAULT_THRESHOLD (0.5) when the model/channel has no calibration rows
    (e.g. the mobilenet gates). Never rewrites the CSV."""
    path = Path(csv_path)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (str(row.get("model", "")).lower() == str(model).lower()
                        and str(row.get("channel", "")).lower() == str(channel).lower()
                        and str(row.get("scanner", "")).strip() == str(scanner)):
                    try:
                        return float(row["threshold"])
                    except (ValueError, KeyError):
                        break
    return DEFAULT_THRESHOLD


def one_vs_all_gate(model, channel, scanner, threshold=None):
    """One-vs-all binary gate ('is this image from scanner N?') backed by
    ScannerVerifier, using the frozen per-scanner threshold. This is the
    model-dependent binary method for Gate 4 (vgg16 / mobilenet)."""
    return ScannerVerifier(
        model, channel, scanner,
        threshold=threshold if threshold is not None
        else scanner_threshold(model, channel, scanner),
    )
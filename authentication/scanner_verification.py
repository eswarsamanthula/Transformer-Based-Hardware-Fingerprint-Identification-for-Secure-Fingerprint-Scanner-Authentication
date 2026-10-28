"""
scanner_verification.py — binary one-vs-all scanner verification.

"is this image from scanner N, yes/no" — the sigmoid twin of
scanner_identification.py. One checkpoint per target scanner:

    checkpoints/<cnn|transformer>/<model>_<channel>_binary_scanner<n>_best.pt
    -> {"mode": "binary", "state_dict", ...}

Used for FAR/FRR-style gating inside authentication_flow.py: given a
claimed scanner and a captured fingerprint, returns the match probability
(sigmoid) and the accept/reject verdict at a chosen threshold (default
0.5, matching test.py's reporting threshold).

Usage:
    from authentication import scanner_verification as sv
    ver = sv.ScannerVerifier("mobilenet", "grayscale")
    result = ver.verify_path(3, "some_fingerprint.bmp")   # Sensor3 match?
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
os.environ.setdefault(
    "THFSA_DATA_ROOT", str(_PROJECT_ROOT / "dataset" / "Dataset1" / "Fingerprint"
))
os.environ.setdefault(
    "THFSA_TEST_DATA_ROOT",
    str(_PROJECT_ROOT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"),
)

import config  # noqa: E402
from models import build_model  # noqa: E402

try:
    from authentication.scanner_identification import CHECKPOINTS_ROOT, _has_lazy, _preprocess
except ModuleNotFoundError:
    from scanner_identification import CHECKPOINTS_ROOT, _has_lazy, _preprocess  # noqa: E402


class ScannerVerifier:
    """Cached one-vs-all model for one target scanner."""

    DEFAULT_THRESHOLD = 0.5  # matches test.py's FAR/FRR reporting threshold

    def __init__(self, model_name: str, channel_mode: str, scanner_n: int,
                 checkpoints_root: Path = CHECKPOINTS_ROOT,
                 threshold: float = DEFAULT_THRESHOLD):
        assert channel_mode in ("grayscale", "rgb")
        assert 1 <= scanner_n <= config.NUM_SCANNERS
        self.model_name = model_name
        self.channel_mode = channel_mode
        self.scanner_n = scanner_n
        self.threshold = threshold
        self.in_channels = 1 if channel_mode == "grayscale" else 3
        self.input_size = config.MODEL_INPUT_SIZE[model_name]
        self.device = "cpu"

        ckpt_path = (
            checkpoints_root / config.model_category(model_name)
            / f"{model_name}_{channel_mode}_binary_scanner{scanner_n}_best.pt"
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no binary checkpoint at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if ckpt.get("mode", "binary") != "binary":
            raise ValueError(f"{ckpt_path} is not a binary checkpoint")
        self.model = build_model(
            model_name, num_classes=1, in_channels=self.in_channels
        ).to(self.device)
        if _has_lazy(self.model):
            self.model(torch.zeros(1, self.in_channels, self.input_size, self.input_size))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def _probability(self, image: Image.Image) -> float:
        x = _preprocess(image, self.channel_mode, self.input_size).to(self.device)
        with torch.no_grad():
            logit = self.model(x)
            p = torch.sigmoid(logit.view(-1))[0].item()
        return float(p)

    def verify_image(self, image: Image.Image) -> dict:
        p = self._probability(image)
        return {
            "model": self.model_name,
            "channel": self.channel_mode,
            "scanner": self.scanner_n,
            "threshold": self.threshold,
            "probability": p,
            "is_match": p >= self.threshold,
        }

    def verify_path(self, path: str | Path) -> dict:
        return self.verify_image(Image.open(path))

    def verify_bytes(self, data: bytes) -> dict:
        return self.verify_image(Image.open(io.BytesIO(data)))


if __name__ == "__main__":
    # self-check: a genuine image claims the right to its own scanner, and
    # the same verifier also runs the FAR-style negative check.
    test = _PROJECT_ROOT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"
    positive = sorted(test.glob("Sensor5*.bmp"))[0]
    ver = ScannerVerifier("mobilenet", "grayscale", scanner_n=5)
    r = ver.verify_path(positive)
    print(f"genuine {positive.name}: p={r['probability']:.3f} match={r['is_match']}")
    negative = sorted(test.glob("Sensor3*.bmp"))[0]
    n = ver.verify_path(negative)
    print(f"impostor {negative.name}: p={n['probability']:.3f} match={n['is_match']}")
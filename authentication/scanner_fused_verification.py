"""
scanner_fused_verification.py — official fused cosine-embedding scanner
verification (EfficientFormer-L1 · RGB).

"is this image from scanner N, yes/no" via the project's official
verification protocol (results/tables/efficientformer_l1_rgb_verification.csv,
reproduced by experiments/make_verification_graphs.py):

  - Frozen multiclass checkpoint:
      checkpoints/transformer/efficientformer_l1_rgb_best.pt
  - Pre-head (penultimate) 448-d embedding per image.
  - Embeddings are L2-normalised; similarity = cosine.
  - Reference template per scanner = mean of the L2-normalised embeddings of
    that scanner from the Dataset 1 VALIDATION split (the same split the
    threshold was calibrated on), re-normalised. Built lazily on first use
    and cached in memory.
  - Frozen decision threshold = validation-EER point 0.389249..., applied
    unchanged at runtime.

This is the live counterpart of the fused protocol that replaced the
superseded EfficientFormer one-vs-all binary run (archived under
checkpoints/archive/superseded/efficientformer_l1_rgb_binary/). The CNN
binary one-vs-all gates (vgg16/mobilenet) remain separate —
scanner_verification.py / scanner_identification_binary.py.

Usage:
    from authentication import scanner_fused_verification as sfv
    ver = sfv.FusedScannerVerifier()
    result = ver.verify_path(3, "some_fingerprint.bmp")   # Sensor3 match?
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
os.environ.setdefault(
    "THFSA_DATA_ROOT", str(_PROJECT_ROOT / "dataset" / "Dataset1" / "Fingerprint")
)
os.environ.setdefault(
    "THFSA_TEST_DATA_ROOT",
    str(_PROJECT_ROOT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"),
)

import config  # noqa: E402
from dataset import R307Dataset, load_or_build_split  # noqa: E402
from models import build_model  # noqa: E402

try:
    from authentication.scanner_identification import CHECKPOINTS_ROOT, _has_lazy, _preprocess
except ModuleNotFoundError:
    from scanner_identification import CHECKPOINTS_ROOT, _has_lazy, _preprocess  # noqa: E402

MODEL_NAME = "efficientformer_l1"
CHANNEL = "rgb"
EMBED_DIM = 448
# Validation-EER operating point (frozen), from
# results/tables/efficientformer_l1_rgb_verification.csv
FROZEN_THRESHOLD = 0.38924925516520564


def _relocated(path):
    """Map manifest paths written under the old THFSA checkout to THFSA_v2."""
    return (path.replace("\\", "/")
                 .replace("Visual Studio Code/THFSA/dataset",
                          "Visual Studio Code/THFSA_v2/dataset"))


class FusedScannerVerifier:
    """Cached EfficientFormer-L1 (RGB) fused cosine verifier."""

    def __init__(self, checkpoints_root: Path = CHECKPOINTS_ROOT,
                 threshold: float = FROZEN_THRESHOLD):
        self.model_name = MODEL_NAME
        self.channel_mode = CHANNEL
        self.threshold = float(threshold)
        self.in_channels = 3
        self.input_size = config.MODEL_INPUT_SIZE[self.model_name]
        self.device = "cpu"

        ckpt_path = checkpoints_root / "transformer" / f"{self.model_name}_rgb_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no multiclass checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if ckpt.get("mode", "multiclass") != "multiclass":
            raise ValueError(f"{ckpt_path} is not a multiclass checkpoint")
        self.model = build_model(
            self.model_name, num_classes=config.NUM_SCANNERS,
            in_channels=self.in_channels
        ).to(self.device)
        if _has_lazy(self.model):
            self.model(torch.zeros(1, self.in_channels, self.input_size, self.input_size))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self._references = None

    # ---- embeddings ------------------------------------------------------

    def _forward_embed(self, x: torch.Tensor) -> np.ndarray:
        """Pre-head (penultimate) embedding for a batch, L2-normalised per
        image — identical capture to experiments/make_verification_graphs.py."""
        feats = []

        def hook(module, inp, out):
            t = inp[0].detach()
            if t.ndim > 2:
                t = torch.nn.functional.adaptive_avg_pool2d(t, 1).flatten(1)
            feats.append(t)

        h = self.model.head.register_forward_hook(hook)
        with torch.no_grad():
            self.model(x)
        h.remove()
        E = torch.cat(feats, 0).cpu().numpy()
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        return En

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Unit-norm 448-d embedding for a single probe image."""
        x = _preprocess(image, self.channel_mode, self.input_size).to(self.device)
        return self._forward_embed(x)[0].astype("float64")

    # ---- reference templates (lazy, from the calibration split) ----------

    @property
    def references(self) -> dict[int, np.ndarray]:
        """{scanner_id 1..10: unit-norm 448-d reference template}, built once
        from the Dataset 1 validation split (same split the threshold was
        frozen on). In-memory cache only; the underlying split and model are
        never modified."""
        if self._references is None:
            _, val = load_or_build_split()
            for s in val:
                s.path = _relocated(s.path)
            samples_by_scanner = {
                n: [s for s in val if s.scanner_idx == n - 1]
                for n in range(1, config.NUM_SCANNERS + 1)
            }
            refs = {}
            for n in range(1, config.NUM_SCANNERS + 1):
                dl = DataLoader(R307Dataset(samples_by_scanner[n],
                                            self.input_size, self.channel_mode),
                                batch_size=32, shuffle=False, num_workers=0)
                feats = []
                with torch.no_grad():
                    for x, _ in dl:
                        feats.append(self._forward_embed(x.to(self.device)))
                E = np.concatenate(feats)
                En = E / np.linalg.norm(E, axis=1, keepdims=True)
                ref = En.mean(axis=0).astype("float64")
                refs[n] = ref / np.linalg.norm(ref)
            self._references = refs
        return self._references

    # ---- verification API ------------------------------------------------

    def verify_image(self, image: Image.Image, scanner_n: int) -> dict:
        """Cosine of the probe against the scanner reference; match if the
        frozen 0.3892 threshold is met."""
        try:
            ref = self.references[int(scanner_n)]
        except KeyError:
            raise ValueError(f"no fused template for scanner {scanner_n}")
        score = float(np.dot(self.embed_image(image), ref))
        return {
            "model": self.model_name,
            "channel": self.channel_mode,
            "scanner": int(scanner_n),
            "threshold": self.threshold,
            "score": round(score, 6),
            "is_match": bool(score >= self.threshold),
        }

    def verify_path(self, path: str | Path, scanner_n: int) -> dict:
        return self.verify_image(Image.open(path), scanner_n)

    def verify_bytes(self, data: bytes, scanner_n: int) -> dict:
        return self.verify_image(Image.open(io.BytesIO(data)), scanner_n)


if __name__ == "__main__":
    # self-check: a genuine Dataset 2 image claims its own scanner, the
    # embedding is 448-d and unit-norm; an impostor score is printed (not
    # asserted — a frozen 0.3892 threshold leaves ~3% FRR by design).
    test = _PROJECT_ROOT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"
    ver = FusedScannerVerifier()
    v = ver.embed_image(Image.open(sorted(test.glob("Sensor10*.bmp"))[0]))
    assert v.shape == (EMBED_DIM,), f"expected {EMBED_DIM}-d embedding, got {v.shape}"
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3, "embedding must be unit norm"
    genuine = ver.verify_path(sorted(test.glob("Sensor10*.bmp"))[0], 10)
    print(f"genuine Sensor10: score={genuine['score']:.4f} "
          f"(th {genuine['threshold']:.4f}) match={genuine['is_match']}")
    assert genuine["is_match"], "genuine same-scanner image must pass"
    impostor = ver.verify_path(sorted(test.glob("Sensor1*.bmp"))[0], 10)
    print(f"impostor  Sensor1 (claimed 10): score={impostor['score']:.4f} "
          f"match={impostor['is_match']}")
    print("scanner_fused_verification self-check passed")
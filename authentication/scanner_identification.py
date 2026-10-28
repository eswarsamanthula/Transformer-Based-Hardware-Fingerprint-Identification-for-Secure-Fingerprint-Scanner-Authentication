"""
scanner_identification.py — multiclass scanner identification.

Loads a trained multiclass checkpoint and runs inference on one fingerprint
image, returning the full softmax distribution over the 10 scanners plus a
top-k ranking. This is the module authentication_flow.py and the web
dashboard use for live model inference.

Checkpoint layout (matches src/test.py):
    checkpoints/<cnn|transformer>/<model>_<channel>_best.pt
    -> {"model_name", "channel_mode", "mode": "multiclass", "state_dict", ...}

Image preprocessing mirrors R307Dataset in src/dataset.py exactly: PIL open,
convert L/RGB, BILINEAR resize to the model's input size, scale to [0,1].
No augmentation, no EXIF rotation — identical to how the models were fed.

Usage:
    from authentication import scanner_identification as si
    ids = si.ScannerIdentifier("vgg16", "grayscale")
    result = ids.predict_path("some_fingerprint.bmp")
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
    "THFSA_DATA_ROOT", str(_PROJECT_ROOT / "dataset" / "Dataset1" / "Fingerprint")
)
os.environ.setdefault(
    "THFSA_TEST_DATA_ROOT",
    str(_PROJECT_ROOT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"),
)

import config  # noqa: E402  (after sys.path setup, as train.py/test.py do)
from models import build_model  # noqa: E402

CHECKPOINTS_ROOT = _PROJECT_ROOT / "checkpoints"


def available_models(root: Path = CHECKPOINTS_ROOT) -> list[dict]:
    """Multiclass (model, channel) pairs that have a checkpoint on disk."""
    seen, found = set(), []
    for p in sorted(root.rglob("*_best.pt")):
        if "_binary_" in p.name:
            continue
        base = p.name[: -len("_best.pt")]
        for ch in ("grayscale", "rgb"):
            if base.endswith("_" + ch):
                model = base[: -(len(ch) + 1)]
                key = (model, ch)
                if key not in seen:
                    seen.add(key)
                    found.append({"model": model, "channel": ch})
                break
    found.sort(key=lambda d: (d["model"], d["channel"]))
    return found


def available_verification_models(root: Path = CHECKPOINTS_ROOT) -> list[str]:
    """Models that have per-scanner one-vs-all checkpoints (e.g. mobilenet)."""
    models, seen = [], set()
    for p in sorted(root.rglob("*_binary_scanner1_best.pt")):
        base = p.name[: -len("_binary_scanner1_best.pt")]
        for ch in ("grayscale", "rgb"):
            if base.endswith("_" + ch):
                model = base[: -(len(ch) + 1)]
                if model not in seen:
                    seen.add(model)
                    models.append(model)
                break
    return models


def _has_lazy(module: torch.nn.Module) -> bool:
    return any(type(m).__name__.startswith("Lazy") for m in module.modules())


def _preprocess(image: Image.Image, channel_mode: str, input_size: int) -> torch.Tensor:
    img = image.convert("L" if channel_mode == "grayscale" else "RGB")
    img = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype="float32") / 255.0
    if channel_mode == "grayscale":
        arr = arr[None, :, :]          # (1, H, W)
    else:
        arr = arr.transpose(2, 0, 1)   # (3, H, W)
    return torch.from_numpy(arr).unsqueeze(0)


class ScannerIdentifier:
    """Cached multiclass model over one (model, channel) checkpoint."""

    def __init__(self, model_name: str, channel_mode: str,
                 checkpoints_root: Path = CHECKPOINTS_ROOT):
        assert channel_mode in ("grayscale", "rgb")
        self.model_name = model_name
        self.channel_mode = channel_mode
        self.in_channels = 1 if channel_mode == "grayscale" else 3
        self.input_size = config.MODEL_INPUT_SIZE[model_name]
        self.device = "cpu"

        ckpt_path = (
            checkpoints_root / config.model_category(model_name)
            / f"{model_name}_{channel_mode}_best.pt"
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no multiclass checkpoint at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if ckpt.get("mode", "multiclass") != "multiclass":
            raise ValueError(f"{ckpt_path} is a binary checkpoint")
        self.model = build_model(
            model_name, num_classes=config.NUM_SCANNERS, in_channels=self.in_channels
        ).to(self.device)
        if _has_lazy(self.model):
            self.model(torch.zeros(1, self.in_channels, self.input_size, self.input_size))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def _probabilities(self, image: Image.Image) -> np.ndarray:
        x = _preprocess(image, self.channel_mode, self.input_size).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return probs

    def predict_image(self, image: Image.Image, topk: int = 5) -> dict:
        probs = self._probabilities(image)
        order = probs.argsort()[::-1]
        return {
            "model": self.model_name,
            "channel": self.channel_mode,
            "probabilities": [float(p) for p in probs],
            "top": [
                {"scanner": int(s + 1), "probability": float(probs[s])}
                for s in order[:topk]
            ],
            "predicted_scanner": int(order[0] + 1),
            "confidence": float(probs[order[0]]),
        }

    def predict_path(self, path: str | Path, topk: int = 5) -> dict:
        return self.predict_image(Image.open(path), topk)

    def predict_bytes(self, data: bytes, topk: int = 5) -> dict:
        return self.predict_image(Image.open(io.BytesIO(data)), topk)


if __name__ == "__main__":
    # self-check: classify one real Dataset 2 image and sanity-check the
    # predicted class sits inside the true sensor's label distribution.
    test = _PROJECT_ROOT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"
    image = sorted(test.glob("Sensor*.bmp"))[0]
    print(f"image: {image.name}")
    ident = ScannerIdentifier("mobilenet", "grayscale")
    r = ident.predict_path(image)
    print(f"predicted: Sensor{r['predicted_scanner']}  confidence={r['confidence']:.3f}")
    print("top-5:", ", ".join(f"S{s['scanner']}={s['probability']:.2f}" for s in r["top"]))
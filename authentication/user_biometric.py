"""
user_biometric.py — user fingerprint enrollment + verification.

AcademIC prototype. User identity is handled entirely by template enrollment
and matching (this module) — the trained VGG16 / EfficientFormer models are
used ONLY for scanner hardware-fingerprint identification (see
scanner_identification_binary.py). No model is retrained here.

Method (deterministic and explainable):
  1. Each sample: grayscale -> BILINEAR resize to `size` x `size` -> [0,1].
  2. Vectorize and centre the pixels; normalise to unit norm.
  3. Enrollment: mean of the per-image unit vectors, re-normalised. That mean
     vector IS the template — only the derived vector is persisted, never the
     raw fingerprint images.
  4. Verification: cosine similarity between a new capture's unit vector and
     the enrolled template. A match is score >= threshold.

This is NOT a production-grade fingerprint minutiae matcher. It is a
deterministic similarity template suitable for the academic prototype.

Security: templates are serialised as JSON (base64-encoded float32 vector),
never with pickle, so untrusted template files cannot execute code on load.
"""

import base64
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 64                    # template resolution (px x px)
DEFAULT_THRESHOLD = 0.70     # cosine threshold used when none is stored
METHOD = "resized_ridge_correlation"

TEMPLATE_TYPE = "thfsa_user_biometric_template"
TEMPLATE_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_image(image):
    """Accept a PIL Image, a file path, or raw image bytes."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, os.PathLike)):
        return Image.open(image)
    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(image)))
    raise TypeError("image must be a PIL Image, path, or bytes")


def _encode_vector(vec: np.ndarray) -> str:
    return base64.b64encode(np.asarray(vec, dtype="float32").tobytes()).decode("ascii")


def _decode_vector(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="float32")


def extract_template_vector(image, size: int = SIZE) -> np.ndarray:
    """Return the centred, unit-norm pixel vector for one fingerprint image."""
    img = _as_image(image).convert("L").resize(
        (size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype="float32").reshape(-1)
    arr = arr - arr.mean()
    norm = np.linalg.norm(arr)
    if norm < 1e-8:            # degenerate all-flat input
        arr = np.ones(arr.shape[0], dtype="float32")
    else:
        arr = arr / norm
    return arr


def enroll_user_fingerprint(images, size: int = SIZE,
                            threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Enroll one or more fingerprint images -> a single derived template.

    Only the averaged, normalised vector (plus metadata) is returned. No raw
    fingerprint images are stored.
    """
    images = list(images)
    if not images:
        raise ValueError("no fingerprint images provided for enrollment")
    vectors = [extract_template_vector(im, size) for im in images]
    mean = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    mean = mean / norm if norm >= 1e-8 else np.ones_like(mean)

    return {
        "type": TEMPLATE_TYPE,
        "version": TEMPLATE_VERSION,
        "method": METHOD,
        "size": size,
        "threshold": float(threshold),
        "num_images": len(images),
        "created_at": _now_iso(),
        "vector_b64": _encode_vector(mean.astype("float32")),
    }


def _validate_template(template: dict) -> dict:
    if not isinstance(template, dict):
        raise ValueError("template must be a dict")
    if template.get("type") != TEMPLATE_TYPE:
        raise ValueError(f"not a {TEMPLATE_TYPE} template")
    if "vector_b64" not in template:
        raise ValueError("template has no vector")
    return template


def verify_user_fingerprint(image, template: dict,
                            threshold: float | None = None) -> dict:
    """Compare a new fingerprint against the enrolled user template.

    Returns: match bool, similarity score, and the threshold used.
    """
    t = _validate_template(template)
    probe = extract_template_vector(image, int(t.get("size", SIZE)))
    enrolled = _decode_vector(t["vector_b64"])
    score = float(np.dot(probe, enrolled))          # both unit norm -> cosine
    thr = float(threshold) if threshold is not None \
        else float(t.get("threshold", DEFAULT_THRESHOLD))
    return {
        "match": bool(score >= thr),
        "score": round(score, 6),
        "threshold": thr,
    }


# ---------------------------------------------------------------------------
# Serialization — JSON only. No pickle anywhere near user-controlled files.
# ---------------------------------------------------------------------------

def save_user_template(path, template: dict) -> None:
    _validate_template(template)
    Path(path).write_text(json.dumps(template), encoding="utf-8")


def load_user_template(path) -> dict:
    """Load a template file, validating it came out of save_user_template."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_template(data)
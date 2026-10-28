"""
test_user_biometric.py — standalone checks for the user fingerprint factor.

Run:  python tests/test_user_biometric.py     (plain asserts, no pytest)
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from authentication import user_biometric as bio

FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def imprint(seed, impression=0, size=90, cell=9, noise=8):
    """Deterministic pseudo-ridge fingerprint. `impression` adds modest
    capture noise to the same latent pattern (same seed = same finger)."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size), dtype="int16")
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if rng.random() < 0.42:
                block = np.array(
                    [[rng.integers(170, 255) if r % 2 == 0 else 40
                      for r in range(cell)] for _ in range(cell)],
                    dtype="int16")
                arr[y:y + cell, x:x + cell] = block
    if impression:
        n = np.random.default_rng(seed * 1000 + impression)
        arr = arr + n.integers(-noise, noise + 1, size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def finger(seed, impressions=(0, 1, 2)):
    return [imprint(seed, i) for i in impressions]


def scenario5_matching_scores():
    """Genuine/genuine similarity must beat genuine/impostor."""
    tpl = bio.enroll_user_fingerprint(finger(1))
    genuine = bio.verify_user_fingerprint(imprint(1, impression=3), tpl)
    impostor = bio.verify_user_fingerprint(imprint(42), tpl)
    check("S5", genuine["score"] > impostor["score"],
          f"genuine {genuine['score']:.3f} vs impostor {impostor['score']:.3f}")
    check("S5", genuine["match"] is True, f"genuine match={genuine['match']}")
    check("S5", impostor["match"] is False, f"impostor match={impostor['match']}")


def scenario6_threshold():
    """One tunable hyperparameter (threshold), default 0.70, overridable."""
    tpl = bio.enroll_user_fingerprint(finger(7))
    probe = imprint(7, impression=4)
    d = bio.verify_user_fingerprint(probe, tpl)   # stored default from template
    check("S6", d["threshold"] == 0.70, f"stored default threshold {d['threshold']}")
    check("S6", bio.DEFAULT_THRESHOLD == 0.70, "module default")
    check("S6", d["match"] is True, f"genuine at default must pass ({d['score']:.3f})")

    strict = bio.verify_user_fingerprint(probe, tpl, threshold=1.0)  # impossible
    check("S6", strict["match"] is False and strict["threshold"] == 1.0,
          "threshold override must reject")
    loose = bio.verify_user_fingerprint(probe, tpl, threshold=0.0)
    check("S6", loose["match"] is True, "threshold override must accept")

    import inspect
    params = set(inspect.signature(
        bio.verify_user_fingerprint).parameters) - {"image", "template", "threshold"}
    check("S6", not params, f"unexpected knobs: {params}")


def scenario7_json_not_pickle():
    """Templates persist as JSON (base64 vector), never pickle."""
    tpl = bio.enroll_user_fingerprint(finger(11))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tpl.json"
        bio.save_user_template(p, tpl)
        raw = p.read_text(encoding="utf-8")
        check("S7", raw.lstrip().startswith("{"), "file must be JSON object")
        bottom = json.loads(raw)
        check("S7", bottom["type"] == bio.TEMPLATE_TYPE, "type tag")
        check("S7", bottom["version"] == 1, "version")
        b64 = bottom.get("vector_b64", "")
        check("S7", isinstance(b64, str) and len(b64) > 0, "vector base64")
        reloaded = bio.load_user_template(p)
        check("S7", json.dumps(reloaded, sort_keys=True)
              == json.dumps(bottom, sort_keys=True), "load round-trips exact")
        check("S7", "__main__" not in raw, "no pickle payload")


def scenario8_privileged_denied():
    """root/admin-style usernames rejected; negative probe must not match."""
    try:
        from authentication import auth_flow
        for name in ("root", "admin", "ADMINISTRATOR"):
            try:
                auth_flow.register(name, "pw1234", finger(3), 5)
                check("S8", False, f"{name} must be rejected")
            except ValueError as e:
                check("S8", "reserved" in str(e), f"{name}: {e}")
    except ImportError as e:
        check("S8", False, f"auth_flow import failed: {e}")

    tpl = bio.enroll_user_fingerprint(finger(21))
    check("S8", bio.verify_user_fingerprint(imprint(999), tpl)["match"] is False,
          "negative probe must not match")


def main():
    scenario5_matching_scores()
    scenario6_threshold()
    scenario7_json_not_pickle()
    scenario8_privileged_denied()
    if FAILURES:
        print("test_user_biometric FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("test_user_biometric passed")


if __name__ == "__main__":
    main()
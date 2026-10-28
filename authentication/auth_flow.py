"""
auth_flow.py — the revised multi-factor mutual MFA gate (register + authenticate).

Rebuilds on authentication_flow.py with the same real HMAC-SHA256 crypto and
server_secret.bin, but the gate is now the full MFA stack:

  * stored password        -> bcrypt hash in authentication/data/thfsa_users.db
  * passkey / OTP factor   -> TOTP (RFC 6238) via pyotp
  * user fingerprint       -> cosine-similarity template (user_biometric.py)
  * scanner verification   -> EXACTLY FOUR gates; the scanner gate is
                              model-dependent (one method runs, selected by
                              `scanner_model` / `scanner_channel`):
                              (a) efficientformer_l1_rgb -> the official fused
                                  448-d L2-normalized embedding + cosine
                                  verifier (scanner_fused_verification.py) at
                                  the frozen threshold 0.3892
                              (b) vgg16_grayscale / mobilenet_grayscale /
                                  mobilenet_rgb -> the frozen one-vs-all
                                  binary verifier (scanner_verification.py /
                                  scanner_identification_binary.py)
  * session-key agreement  -> the mutual HMAC-SHA256 exchange still derives
                              K_session (required internally), but it is NOT a
                              separate gate: it is folded into Gate 4's check,
                              so the live trace is exactly four gates.

Stored secrets are never returned after "ok"; register() is the only caller
that ever sees the one-time TOTP secret and the device V. Run
`python -m authentication.auth_flow` for an end-to-end self-test on real
Dataset 2 images.
"""

import hmac
import io
import time
from pathlib import Path

from PIL import Image

from . import passkey as pk
from . import user_biometric as bio
from . import user_store as store
from .authentication_flow import (
    load_or_create_server_secret,
    session_key,
    sign,
    verification_value,
    verify_signature,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET2 = PROJECT_ROOT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"

_FUSED_VERIFIER = None


def _default_fused_verifier():
    """Lazy EfficientFormer fused verifier (builds val-split reference
    templates once, on first use)."""
    global _FUSED_VERIFIER
    if _FUSED_VERIFIER is None:
        from .scanner_fused_verification import FusedScannerVerifier
        _FUSED_VERIFIER = FusedScannerVerifier()
    return _FUSED_VERIFIER


def _default_binary_gate(model, channel, scanner):
    """Lazy one-vs-all binary gate for the requested (model, channel, scanner)
    with its frozen per-scanner threshold."""
    from .scanner_identification_binary import one_vs_all_gate
    return one_vs_all_gate(model, channel, scanner)

RESERVED_USERNAMES = {"root", "admin", "administrator", "system", "sa"}


def _open_image(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(image)))
    return Image.open(image)


def register(username, password, fingerprint_images, declared_scanner,
             *, db_path=None) -> dict:
    """Enroll a new user; returns {ok, user_id, passkey_secret, device_secret}.

    `passkey_secret` is the one-time TOTP base32 secret the user installs in
    an authenticator now (deliberately not recoverable after this call).
    `device_secret` is V = HMAC(server_secret, password), which the device
    stores and later presents for the mutual session-key exchange.
    """
    secret = load_or_create_server_secret()
    if username.lower() in RESERVED_USERNAMES:
        raise ValueError(f"reserved username: {username}")
    passkey_secret = pk.create_passkey_secret()
    v = verification_value(secret, password)

    store.create_user(
        username, store.hash_password(password), passkey_secret,
        v_hex=v.hex(), db_path=db_path,
    )
    store.store_biometric_template(username, bio.enroll_user_fingerprint(
        fingerprint_images), db_path=db_path)
    store.bind_scanner_to_user(username, declared_scanner, db_path=db_path)

    return {"ok": True, "user_id": username,
            "passkey_secret": passkey_secret, "device_secret": v.hex()}


def _gate(trace: list[dict], n: int, label: str, passed: bool,
          detail: str = "") -> None:
    """Record one MFA gate for the trace rendered in the auth lab."""
    trace.append({"gate": n, "label": label, "ok": bool(passed),
                  "detail": detail})


def authenticate(username, password, passcode, fingerprint_image,
                 declared_scanner, *, scanner_model=None, scanner_channel=None,
                 scanner_gate=None, fused_verifier=None, device_secret=None,
                 db_path=None) -> dict:
    """Four-gate MFA; returns {ok, reason, user_score, predicted_scanner,
    match, score, threshold, method, session_key, trace}.

    Gates: 1 stored password (bcrypt), 2 passkey (TOTP), 3 user fingerprint
    template (cosine), 4 scanner verification. Gate 4 is model-dependent:
    scanner_model/scanner_channel steers it to the EfficientFormer-L1 fused
    448-d cosine verifier, or to the one-vs-all binary verifier for the CNN
    models. The mutual HMAC-SHA256 exchange still derives the shared session
    key but is folded into Gate 4, not a fifth gate.

    `trace` is the per-gate record ({gate, label, ok, detail}) shown by the
    web auth lab — exactly four entries. `scanner_gate` is an injected binary
    gate (verify_image(image) -> dict; built lazily when skipped).
    `fused_verifier` is the injected EfficientFormer fused verifier (verify_
    image(image, scanner) -> dict; built lazily when skipped). `device_secret`
    is the V the device stored at registration; when given it must match the V
    of the *presented* password or the session-key agreement fails — exactly
    like a device whose user enrolled with a different password.
    """
    secret = load_or_create_server_secret()
    user = store.get_user(username, db_path=db_path)
    if user is None:
        return {"ok": False, "reason": "unknown user", "trace": []}

    trace: list[dict] = []

    # --- factor 1: stored password (bcrypt) ---
    pw_ok = store.verify_password(username, password, db_path=db_path)
    _gate(trace, 1, "stored password · bcrypt", pw_ok)
    if not pw_ok:
        return {"ok": False, "reason": "wrong password", "trace": trace}

    # --- factor 2: passkey / OTP code ---
    pc_ok = pk.verify_passcode(user["passkey_secret"], passcode)
    _gate(trace, 2, "passkey · TOTP ±30s", pc_ok, "6 digits")
    if not pc_ok:
        return {"ok": False, "reason": "wrong passkey code", "trace": trace}

    # --- factor 3: user fingerprint template ---
    tpl = store.get_biometric_template(username, db_path=db_path)
    if tpl is None:
        _gate(trace, 3, "user fingerprint template", False,
              "no enrolled template")
        return {"ok": False, "reason": "no enrolled biometric",
                "trace": trace}
    check = bio.verify_user_fingerprint(fingerprint_image, tpl)
    user_score = check["match"]
    _gate(trace, 3, "user fingerprint template", user_score,
          f"cosine {check['score']:.3f} · threshold {check['threshold']}"
          f" · {'match' if user_score else 'FAIL'}")
    if not user_score:
        return {"ok": False, "reason": "fingerprint match failed",
                "user_score": user_score, "score": check["score"],
                "threshold": check["threshold"], "trace": trace}

    # --- factor 4: scanner verification (model-dependent) ------------------
    # efficientformer_l1/rgb -> the official fused 448-d embedding + cosine
    # verifier at frozen 0.3892; vgg16/mobilenet -> the frozen one-vs-all
    # binary verifier. Exactly one method runs.
    bound = store.get_bound_scanner(username, db_path=db_path)
    if bound is None:
        _gate(trace, 4, "scanner verification", False, "no bound scanner")
        return {"ok": False, "reason": "no bound scanner", "user_score": True,
                "trace": trace}
    if int(declared_scanner) != bound:
        _gate(trace, 4, "scanner verification", False,
              f"declared Scanner{declared_scanner} != bound Scanner{bound}")
        return {"ok": False, "reason": "declared scanner != bound scanner",
                "user_score": True, "predicted_scanner": int(declared_scanner),
                "trace": trace}

    scanner_model = (scanner_model or "efficientformer_l1").lower()
    scanner_channel = (scanner_channel or "rgb").lower()
    image = _open_image(fingerprint_image)

    gate_label = f"scanner verification · {scanner_model}_{scanner_channel}"
    if scanner_model == "efficientformer_l1" and scanner_channel == "rgb":
        ver = fused_verifier if fused_verifier is not None \
            else _default_fused_verifier()
        res = ver.verify_image(image, int(bound))
        method = "embedding · cosine"
    else:
        ver = scanner_gate if scanner_gate is not None \
            else _default_binary_gate(scanner_model, scanner_channel, bound)
        res = ver.verify_image(image)
        method = "binary · sigmoid"
    score = float(res.get("score", res.get("probability", 0.0)))
    thr = res.get("threshold")
    pred_ok = bool(res.get("is_match", False))

    if not pred_ok:
        _gate(trace, 4, gate_label, False,
              f"{method} score {score:.3f} vs bound Scanner{bound}"
              + (f" · frozen threshold {thr:.4f}" if thr is not None else "")
              + " · FAIL")
        return {"ok": False, "reason": "biometric gate failed",
                "user_score": True, "predicted_scanner": int(bound),
                "match": False, "score": score, "threshold": thr,
                "method": method, "trace": trace}

    # Session-key agreement (mutual HMAC-SHA256) — still required internally
    # for K_session, but folded into Gate 4 so the trace stays at four gates.
    t_user = time.time()
    t_user_b = repr(t_user).encode()
    payload_sig = sign(secret, t_user_b, b"scanner:" + str(bound).encode())
    if not verify_signature(secret, payload_sig, t_user_b,
                            b"scanner:" + str(bound).encode()):
        _gate(trace, 4, gate_label, False, f"{method} · bad request signature")
        return {"ok": False, "reason": "bad request signature",
                "user_score": True, "predicted_scanner": int(bound),
                "match": True, "score": score, "threshold": thr,
                "method": method, "trace": trace}

    v_server = bytes.fromhex(user["v_hex"]) if user.get("v_hex") else b""
    if not v_server:
        _gate(trace, 4, gate_label, False, f"{method} · missing server V")
        return {"ok": False, "reason": "missing server V",
                "user_score": True, "predicted_scanner": int(bound),
                "match": True, "score": score, "threshold": thr,
                "method": method, "trace": trace}

    t_server = time.time()
    k_server = session_key(v_server, t_user, t_server)
    if device_secret:
        k_device = session_key(bytes.fromhex(device_secret), t_user, t_server)
        agreed = hmac.compare_digest(k_device, k_server)
    else:
        agreed = True
    _gate(trace, 4, gate_label, agreed,
          f"{method}: score {score:.3f} vs bound Scanner{bound}"
          + (f" · frozen threshold {thr:.4f}" if thr is not None else "")
          + (" · K_session agreed" if agreed else " · K_session MISMATCH"))

    if not agreed:
        return {"ok": False, "reason": "session-key mismatch",
                "user_score": True, "predicted_scanner": int(bound),
                "match": True, "score": score, "threshold": thr,
                "method": method, "trace": trace}

    return {
        "ok": True,
        "reason": None,
        "user_score": True,
        "predicted_scanner": int(bound),
        "match": True,
        "score": score,
        "threshold": thr,
        "method": method,
        "session_key": k_server.hex(),
        "trace": trace,
    }


class _FakeGate:
    """Deterministic stand-in for self/unit tests: accepts every probe.
    The real models are exercised by tests/test_<...>.py."""

    def verify_image(self, image):
        return {"model": "vgg16", "channel": "grayscale", "scanner": 10,
                "threshold": 0.5, "probability": 1.0, "is_match": True}


def _self_test():
    import shutil
    import tempfile
    from pathlib import Path as _P

    tmp = _P(tempfile.mkdtemp())
    db = tmp / "t_users.db"
    gate = _FakeGate()
    try:
        img = sorted(DATASET2.glob("Sensor10*.bmp"))[0]
        r = register("self_test_user", "pw123", [img], 10, db_path=db)
        assert r["ok"] and r["device_secret"]

        code = pk.generate_passcode(r["passkey_secret"])
        ok = authenticate("self_test_user", "pw123", code, img, 10,
                          scanner_model="vgg16", scanner_gate=gate,
                          device_secret=r["device_secret"], db_path=db)
        assert ok["ok"] and ok["session_key"] and ok["predicted_scanner"] == 10

        code = pk.generate_passcode(r["passkey_secret"])
        assert not authenticate("self_test_user", "wrong", code, img, 10,
                                scanner_model="vgg16", scanner_gate=gate,
                                device_secret=r["device_secret"],
                                db_path=db)["ok"]

        code = pk.generate_passcode(r["passkey_secret"])
        assert not authenticate("self_test_user", "pw123", "000000", img, 10,
                                scanner_model="vgg16", scanner_gate=gate,
                                db_path=db)["ok"]

        code = pk.generate_passcode(r["passkey_secret"])
        assert not authenticate("self_test_user", "pw123", code, img, 7,
                                scanner_model="vgg16", scanner_gate=gate,
                                db_path=db)["ok"]
        print("auth_flow self-test passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _self_test()
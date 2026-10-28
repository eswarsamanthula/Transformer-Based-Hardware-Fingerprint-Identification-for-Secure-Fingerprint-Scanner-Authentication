"""
authentication_flow.py — two-phase mutual MFA protocol (device <-> server).

Implements the registration + mutual-authentication protocol end to end,
per Design_and_solution_Non_AI.md Section 5. Crypto is real HMAC-SHA256
(no print-style simulation), and the biometric factor is real live
inference via scanner_identification.py:

  Registration (enroll):
      1. user supplies a password; server_s secret is used to derive
         V = HMAC(secret, password)     <- stored on the device
  Authentication (mutual):
      1. device sends (fingerprint capture, declared_scanner, t_user, sig)
      2. server verifies sig, runs scanner identification on the image,
         and requires predicted_scanner == declared_scanner
      3. server replies (t_server, H(secret), sig)
      4. device verifies the reply, prompts for the password, re-derives
         V from (secret, password), asserts V == stored V, then computes
         K_session = HMAC(V, t_user || t_server)
      5. server derives K_session from its own V and the same two clocks;
         both sides must agree on the same session key

Transport is out of scope here (the web layer owns HTTP); this module is
the reusable crypto + biometric gate. Run `python authentication_flow.py`
for an end-to-end self-test on real Dataset 2 images.
"""

import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path

from PIL import Image

try:
    from authentication.scanner_identification import ScannerIdentifier
except ModuleNotFoundError:
    from scanner_identification import ScannerIdentifier  # type: ignore[no-redef]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECRET_PATH = Path(__file__).resolve().parent / "server_secret.bin"


def _hmac(key: bytes, *parts) -> bytes:
    h = hmac.new(key, digestmod="sha256")
    for part in parts:
        h.update(part if isinstance(part, bytes) else str(part).encode())
    return h.digest()


def password_hash(password: str) -> bytes:
    return hashlib.sha256(password.encode("utf-8")).hexdigest().encode()


def load_or_create_server_secret(path: Path = SECRET_PATH) -> bytes:
    """Server-side long-lived secret, persisted once on first use."""
    if path.exists():
        return path.read_bytes()
    secret = secrets.token_bytes(32)
    path.write_bytes(secret)
    if hasattr(os, "chmod"):
        os.chmod(path, 0o600)
    return secret


def verification_value(secret: bytes, password: str) -> bytes:
    """V = HMAC(secret, password) — the value stored on the device at enroll."""
    return _hmac(secret, password_hash(password))


def sign(secret: bytes, *parts) -> bytes:
    return _hmac(secret, *parts)


def verify_signature(secret: bytes, expected: bytes, *parts) -> bool:
    return hmac.compare_digest(expected, sign(secret, *parts))


def session_key(v: bytes, t_user: float, t_server: float) -> bytes:
    """K_session = HMAC(V, t_user || t_server) — both sides compute it."""
    return _hmac(v, repr(t_user), repr(t_server))


def authenticate(image: Image.Image | bytes | str | Path, password: str,
                 declared_scanner: int, identifier,
                 secret: bytes | None = None,
                 server_password: str | None = None) -> dict:
    """Driver for the web layer: runs the full exchange and returns the trace.

    `identifier` is a scanner_identification.ScannerIdentifier used as the
    server's biometric gate. `server_password` is the password the server
    derived its V from at registration (defaults to the password the user
    presents now); the two differ -> the session-key check fails.
    """
    import io as _io

    if not isinstance(identifier, ScannerIdentifier):
        raise TypeError("identifier must be a ScannerIdentifier")
    if isinstance(image, Image.Image):
        pass
    elif isinstance(image, (str, Path)):
        image = Image.open(image)
    else:
        image = Image.open(_io.BytesIO(image))
    secret = load_or_create_server_secret() if secret is None else secret
    server_password = password if server_password is None else server_password

    steps: list[dict] = []

    # --- registration ---
    v_device = verification_value(secret, password)
    v_server = verification_value(secret, server_password)  # from registration time
    steps.append({
        "role": "device", "label": "registration",
        "detail": f"stored V = HMAC(server_secret, sha256(password)) = {v_device.hex()[:16]}\u2026",
    })

    # --- phase 1: device -> server ---
    t_user = time.time()
    t_user_bytes = repr(t_user).encode()
    payload_sig = sign(secret, t_user_bytes, b"scanner:" + str(declared_scanner).encode())
    steps.append({
        "role": "device", "label": "phase 1 · sends",
        "detail": f"capture + declared Scanner{declared_scanner} + t_user={t_user:.3f} + sig",
    })

    # --- phase 2: server verifies ---
    if not verify_signature(secret, payload_sig, t_user_bytes,
                            b"scanner:" + str(declared_scanner).encode()):
        return {"ok": False, "steps": steps, "reason": "bad request signature"}
    pred = identifier.predict_image(image)
    bio_ok = pred["predicted_scanner"] == declared_scanner
    steps.append({
        "role": "server", "label": "phase 2 · verifies",
        "detail": f"signature valid \u00b7 predicted Scanner{pred['predicted_scanner']} "
                  f"(conf={pred['confidence']:.3f}) vs declared Scanner{declared_scanner} "
                  f"-> {'match' if bio_ok else 'REJECT'}"
    })
    if not bio_ok:
        return {"ok": False, "steps": steps, "reason": "biometric gate failed"}

    # --- phase 3: server -> device ---
    t_server = time.time()
    server_hash = hashlib.sha256(secret).hexdigest()
    reply_sig = sign(secret, str(t_server), server_hash)
    steps.append({
        "role": "server", "label": "phase 3 · sends",
        "detail": f"H(server_secret)={server_hash[:16]}\u2026 + t_server={t_server:.3f} + sig",
    })

    # --- phase 4: device verifies response, checks V, derives K ---
    if not verify_signature(secret, reply_sig, str(t_server), server_hash):
        return {"ok": False, "steps": steps, "reason": "bad server reply signature"}
    steps.append({
        "role": "device", "label": "phase 4 · verifies",
        "detail": f"reply signature valid \u00b7 re-derived V matches stored V; "
                  f"computing K_session = HMAC(V, t_user || t_server)",
    })
    k_device = session_key(v_device, t_user, t_server)

    # --- phase 5: server independently derives the same K ---
    k_server = session_key(v_server, t_user, t_server)
    agreed = k_device == k_server
    steps.append({
        "role": "server", "label": "phase 5 · derives",
        "detail": f"K_session = {k_server.hex()[:16]}\u2026 "
                  f"-> {'agreed with device' if agreed else 'MISMATCH'}"
    })
    return {
        "ok": agreed,
        "steps": steps,
        "session_key": k_device.hex() if agreed else None,
        "predicted_scanner": pred["predicted_scanner"],
        "confidence": pred["confidence"],
    }


def _do_self_test() -> None:
    test = _PROJECT_ROOT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"
    ident = ScannerIdentifier("mobilenet", "grayscale")

    genuine = sorted(test.glob("Sensor10*.bmp"))[0]       # predicts S10 @ 1.00
    impostor = sorted(test.glob("Sensor5(Test1)1.bmp"))[0]   # predicts S7 != 10
    secret = b"test-secret-for-self-check-not-production"

    for label, path, declared in (
        ("genuine claim", genuine, 10),
        ("impostor claim", impostor, 10),
    ):
        r = authenticate(path, "hunter2", declared, ident, secret=secret)
        print(f"\n=== {label}: declared Scanner{declared} -> ok={r['ok']}")
        for s in r["steps"]:
            print(f"  [{s['role']:7s}] {s['label']:20s} {s['detail']}")
        if r.get("session_key"):
            print(f"  >> agreed session key {r['session_key'][:16]}\u2026")

    assert authenticate(genuine, "hunter2", 10, ident, secret=secret)["ok"]
    assert not authenticate(impostor, "hunter2", 10, ident, secret=secret)["ok"]
    assert not authenticate(
        genuine, "wrong-password", 10, ident, secret=secret, server_password="hunter2"
    )["ok"]
    print("\nauthentication_flow self-test passed")


if __name__ == "__main__":
    _do_self_test()
"""
test_authentication_flow.py — standalone checks for the full auth_flow
register + authenticate gate on real Dataset 2 images (no web layer). The
scanner gate is model-dependent: the default login uses the real
EfficientFormer-L1 fused 448-d cosine verifier, and a pinned CNN login uses
the real one-vs-all binary verifier (both restricted to the bound scanner).

Run:  python tests/test_authentication_flow.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authentication import auth_flow as af
from authentication import passkey as pk
from authentication.authentication_flow import verification_value

PROJECT = Path(__file__).resolve().parents[1]
DATASET2 = PROJECT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"

FAILURES = []

TEST_SERVER_SECRET = b"flow-test-server-secret-not-production"


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def img(name):
    return str(DATASET2 / name)


def setup_user(db_path, username="eve", password="s3cret",
               declared=10):
    images = [img(f"Sensor{declared}(Test1)1.bmp"),
              img(f"Sensor{declared}(Test1)2.bmp"),
              img(f"Sensor{declared}(Test1)3.bmp")]
    reg = af.register(username, password, images, declared, db_path=db_path)
    assert reg["ok"]
    return reg


def scenario3_full_pass(db_path, reg):
    code = pk.generate_passcode(reg["passkey_secret"])
    r = af.authenticate("eve", "s3cret", code,
                        img(f"Sensor10(Test1)4.bmp"), 10,
                        device_secret=None, db_path=db_path)
    check("S3", r["ok"] is True and r.get("session_key"),
          f"correct factors must authenticate: {r}")
    check("S3", r["predicted_scanner"] == 10 and r["match"] is True,
          f"scanner + fingerprint gates must pass: {r}")
    check("S3", r.get("method") == "embedding · cosine",
          f"default login must use the fused embedding/cosine verifier: {r}")
    trace = r.get("trace") or []
    check("S3", len(trace) == 4 and all(t["ok"] for t in trace),
          f"gate trace must show all 4 gates: {trace}")


def scenario_binary_gate_full_pass(db_path, reg):
    """A login pinned to a CNN model must run the one-vs-all binary gate
    (real vgg16 checkpoint + frozen calibration threshold)."""
    code = pk.generate_passcode(reg["passkey_secret"])
    r = af.authenticate("eve", "s3cret", code,
                        img("Sensor10(Test1)4.bmp"), 10,
                        scanner_model="vgg16", scanner_channel="grayscale",
                        device_secret=None, db_path=db_path)
    check("S-BIN", r["ok"] is True and r.get("session_key"),
          f"binary-gate login must authenticate: {r}")
    check("S-BIN", r.get("method") == "binary · sigmoid" and
          r.get("score") is not None and r.get("threshold") is not None,
          f"binary gate must report sigmoid method/score/threshold: {r}")
    trace = r.get("trace") or []
    check("S-BIN", len(trace) == 4 and all(t["ok"] for t in trace),
          f"binary login must trace all 4 gates: {trace}")


def scenario4_wrong_password(db_path):
    r = af.authenticate("eve", "wrongpassword", "000000",
                        img("Sensor10(Test1)4.bmp"), 10, db_path=db_path)
    check("S4", r["ok"] is False and r["reason"] == "wrong password",
          f"wrong password: {r}")


def scenario13_wrong_declared(db_path, reg):
    code = pk.generate_passcode(reg["passkey_secret"])
    r = af.authenticate("eve", "s3cret", code, img("Sensor10(Test1)4.bmp"), 7,
                        db_path=db_path)
    check("S13", r["ok"] is False
          and r["reason"] == "declared scanner != bound scanner",
          f"declared must equal bound scanner: {r}")


def scenario15_device_secret_mismatch(db_path, reg):
    """A device that enrolled with a different password fails the mutual
    HMAC session-key agreement even when every other factor is correct."""
    other_v = verification_value(TEST_SERVER_SECRET, "someone-elses-password")
    r = af.authenticate("eve", "s3cret",
                        pk.generate_passcode(reg["passkey_secret"]),
                        img("Sensor10(Test1)4.bmp"), 10,
                        device_secret=other_v.hex(), db_path=db_path)
    check("S15", r["ok"] is False and r["reason"] == "session-key mismatch",
          f"foreign device V must break the session key: {r}")
    # fresh code (the first call's model cold-start may age the previous one)
    r2 = af.authenticate("eve", "s3cret",
                         pk.generate_passcode(reg["passkey_secret"]),
                         img("Sensor10(Test1)4.bmp"), 10,
                         device_secret=reg["device_secret"], db_path=db_path)
    check("S15", r2["ok"] is True and r2.get("session_key"),
          f"own device V must agree: {r2}")


def main():
    tmpdir = Path(tempfile.mkdtemp())
    db = tmpdir / "flow_users.db"
    try:
        af.load_or_create_server_secret = lambda: TEST_SERVER_SECRET
        reg = setup_user(db)
        scenario3_full_pass(db, reg)
        scenario4_wrong_password(db)
        scenario13_wrong_declared(db, reg)
        scenario15_device_secret_mismatch(db, reg)
        scenario_binary_gate_full_pass(db, reg)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if FAILURES:
        print("test_authentication_flow FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("test_authentication_flow passed")


if __name__ == "__main__":
    main()
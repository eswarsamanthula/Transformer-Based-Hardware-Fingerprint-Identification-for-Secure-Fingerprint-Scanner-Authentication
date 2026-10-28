"""
test_auth_web.py — standalone end-to-end checks over the Flask register/login
routes (hit the real app with the Flask test client; the CNN one-vs-all
scanner gate and the EfficientFormer fused-cosine gate are replaced by
deterministic doubles that accept the bound scanner).

Run:  python tests/test_auth_web.py
"""

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authentication import auth_flow, passkey as pk
from web import app as webapp

PROJECT = Path(__file__).resolve().parents[1]
DATASET2 = PROJECT / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"

FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"{name}: {detail}")


class FakeBinaryGate:
    """Replaces the one-vs-all binary scanner gate in the web routes:
    accepts every probe for the bound scanner."""

    def verify_image(self, image):
        return {"model": "vgg16", "channel": "grayscale", "scanner": 10,
                "threshold": 0.5, "probability": 1.0, "is_match": True}

    def verify_bytes(self, data):
        return self.verify_image(io.BytesIO(data))


class FakeFusedVerifier:
    """Replaces the EfficientFormer fused-cosine gate in the web routes:
    accepts every probe."""

    def verify_image(self, image, scanner_n):
        return {"model": "efficientformer_l1", "channel": "rgb",
                "scanner": int(scanner_n), "threshold": 0.389249255,
                "score": 0.99, "is_match": True}

    def verify_bytes(self, data, scanner_n):
        return self.verify_image(io.BytesIO(data), scanner_n)


def bmp(name):
    return (DATASET2 / name).read_bytes()


def register(client, username, password, images=("Sensor10(Test1)1.bmp",
                                                  "Sensor10(Test1)2.bmp")):
    data = {"username": username, "password": password,
            "password2": password, "declared_scanner": "10"}
    for i, name in enumerate(images):
        data[f"image"] = (io.BytesIO(bmp(name)), f"{i}_{name}")
    return client.post("/api/auth/register", data=data,
                       content_type="multipart/form-data")


def login(client, username, password, passcode, image="Sensor10(Test1)4.bmp",
          model=None, channel=None):
    data = {"username": username, "password": password, "passcode": passcode,
            "declared_scanner": "10",
            "image": (io.BytesIO(bmp(image)), image)}
    if model:
        data["model"] = model
    if channel:
        data["channel"] = channel
    return client.post("/api/auth/login", data=data,
                       content_type="multipart/form-data")


def test_login_page_smoke(client):
    r = client.get("/auth")
    check("web-smoke", r.status_code == 200, f"GET /auth -> {r.status_code}")
    html = r.get_data(as_text=True)
    check("web-smoke", "register" in html and "overlay" in html,
          "auth page must embed register form + passkey overlay")
    check("web-smoke", 'id="overlay" hidden' in html.replace("\n", " "),
          "passkey overlay must start hidden")
    check("web-smoke", "btn-mfa" in html and "btn-identify" in html,
          "auth page must embed the infer lab (identify/authenticate deck)")
    check("web-smoke", "EfficientFormer-L1 RGB" in html and "448-D" in html,
          "auth page must describe the fused 448-D cosine protocol")
    check("web-smoke", "four gates" in html.replace("\n", " ") and
          "five gates" not in html.replace("\n", " "),
          "auth page must present exactly four gates (no fifth HMAC gate)")
    check("web-smoke", "one-vs-all sigmoid" not in html.replace("\n", " ") and
          "threshold 0.5" not in html.replace("\n", " "),
          "no obsolete one-vs-all-sigmoid-0.5 wording on the auth page")
    for old in ("/login", "/infer"):
        r = client.get(old)
        check("web-redir", r.status_code == 302 and "/auth" in r.headers.get("Location", ""),
              f"GET {old} -> {r.status_code} {r.headers.get('Location')}")


def test_verify_lab(client):
    data = {"scanner": "10",
            "image": (io.BytesIO(bmp("Sensor10(Test1)4.bmp")),
                      "4_Sensor10(Test1)4.bmp")}
    r = client.post("/api/verify", data=data,
                    content_type="multipart/form-data")
    body = r.get_json()
    check("web-verify", r.status_code == 200 and body.get("is_match") is True,
          f"/api/verify (fused) -> {r.status_code} {body}")
    check("web-verify", body.get("method") == "embedding · cosine",
          f"fused verify must report the embedding/cosine method: {body}")
    check("web-verify", body.get("score") is not None and
          body.get("threshold") is not None,
          "verify must return a cosine score and the frozen threshold")
    check("web-verify", body.get("threshold") == 0.389249255,
          f"verify must report the frozen threshold: {body}")
    off = body.get("official") or {}
    check("web-verify", off.get("split") == "test_dataset2" and
          off.get("far") is not None and off.get("frr") is not None,
          "fused verify must carry official Dataset 2 FAR/FRR/EER context")

    data = {"scanner": "10", "model": "vgg16", "channel": "grayscale",
            "image": (io.BytesIO(bmp("Sensor10(Test1)4.bmp")),
                      "4_Sensor10(Test1)4.bmp")}
    r = client.post("/api/verify", data=data,
                    content_type="multipart/form-data")
    body = r.get_json()
    check("web-verify", r.status_code == 200 and body.get("is_match") is True,
          f"/api/verify (binary) -> {r.status_code} {body}")
    check("web-verify", body.get("method") == "binary · sigmoid" and
          body.get("probability") is not None and
          body.get("threshold") == 0.5,
          f"binary verify must report sigmoid method + threshold: {body}")
    check("web-verify", body.get("official") is None,
          "binary verify carries no official fused metrics")


def test_duplicate_register(client):
    r = register(client, "alice", "pw1234")
    check("web-dupe", r.status_code == 200 and r.get_json().get("ok"),
          f"register alice -> {r.status_code} {r.get_json()}")
    d = r.get_json()
    check("web-dupe", d.get("passkey_secret") and d.get("otpauth_uri"),
          "registration must return one-time passkey secret + otpauth uri")
    r2 = register(client, "alice", "pw1234")
    check("web-dupe", r2.status_code == 409, f"duplicate -> {r2.status_code}")
    return d


def test_wrong_password(client, reg):
    r = login(client, "alice", "wrongpass", pk.generate_passcode(reg["passkey_secret"]))
    body = r.get_json()
    check("web-pw", body["ok"] is False and body["reason"] == "wrong password",
          f"wrong password: {body}")


def test_wrong_passcode(client, reg):
    r = login(client, "alice", "pw1234", "000000")
    body = r.get_json()
    check("web-otp", body["ok"] is False and body["reason"] == "wrong passkey code",
          f"wrong OTP: {body}")


def test_full_login_success(client, reg):
    code = pk.generate_passcode(reg["passkey_secret"])
    r = login(client, "alice", "pw1234", code)
    body = r.get_json()
    check("web-ok", body.get("ok") is True and body.get("session_key"),
          f"full login (fused gate): {body}")
    check("web-ok", body.get("predicted_scanner") == 10, "predicted bound scanner")
    check("web-ok", body.get("method") == "embedding · cosine",
          f"default login must use the fused embedding/cosine gate: {body}")
    trace = body.get("trace") or []
    check("web-ok", len(trace) == 4 and all(t["ok"] for t in trace),
          f"successful login must trace all 4 gates: {trace}")


def test_full_login_binary_gate(client, reg):
    code = pk.generate_passcode(reg["passkey_secret"])
    r = login(client, "alice", "pw1234", code, model="vgg16",
              channel="grayscale")
    body = r.get_json()
    check("web-bin", body.get("ok") is True and body.get("session_key"),
          f"full login (binary gate): {body}")
    check("web-bin", body.get("method") == "binary · sigmoid" and
          body.get("score") == 1.0 and body.get("threshold") == 0.5,
          f"vgg16 login must use the one-vs-all binary gate: {body}")
    trace = body.get("trace") or []
    check("web-bin", len(trace) == 4 and all(t["ok"] for t in trace),
          f"binary login must trace all 4 gates: {trace}")


def main():
    tmpdir = Path(tempfile.mkdtemp())
    os.environ["THFSA_DB_PATH"] = str(tmpdir / "web_users.db")
    try:
        webapp.SCANNER_GATE = FakeBinaryGate()
        webapp.FUSED_VERIFIER = FakeFusedVerifier()
        auth_flow.load_or_create_server_secret = (
            lambda: b"web-test-server-secret")
        client = webapp.app.test_client()
        test_login_page_smoke(client)
        test_verify_lab(client)
        reg = test_duplicate_register(client)
        test_wrong_password(client, reg)
        test_wrong_passcode(client, reg)
        test_full_login_success(client, reg)
        test_full_login_binary_gate(client, reg)
    finally:
        os.environ.pop("THFSA_DB_PATH", None)
        shutil.rmtree(tmpdir, ignore_errors=True)
    if FAILURES:
        print("test_auth_web FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("test_auth_web passed")


if __name__ == "__main__":
    main()
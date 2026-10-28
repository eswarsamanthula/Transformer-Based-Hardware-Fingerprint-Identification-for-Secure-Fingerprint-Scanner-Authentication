"""
app.py — THFSA results dashboard.

Single-page dashboard over the results CSVs already produced by the
pipeline. It reads nothing but real files, so the numbers on screen can
never drift from what src/test.py actually wrote:

    results/tables/results_summary.csv         — multiclass scanner ID (val + test)
    results/tables/results_binary_summary.csv  — one-vs-all verification, per scanner
    logs/training[/colab]/*.csv                — per-epoch training curves

Run:  python web/app.py   (http://127.0.0.1:5001)
"""

from pathlib import Path
import base64
import csv
import io
import os
import re
import sys

import qrcode
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from authentication.scanner_identification import (  # noqa: E402
    ScannerIdentifier,
    available_models,
)
from authentication import authentication_flow  # noqa: E402
from authentication import auth_flow  # noqa: E402

RESULTS_PATH = BASE / "results" / "tables"
# The "new_result/" tree no longer exists (consolidated into results/tables);
# this merge-if-present fallback is kept so old archives would still load.
ADDITIONAL_RESULTS_PATH = BASE / "new_result" / "results" / "tables"
LOGS_PATH = BASE / "logs" / "training"
METRICS_PATH = BASE / "results" / "metrics"
GRAPHS_PATH = BASE / "results" / "graphs"

app = Flask(__name__)
app.secret_key = os.environ.get("THFSA_SECRET_KEY", "thfsa-dev-key")  # dev-only
app.config["SESSION_COOKIE_HTTPONLY"] = True


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v == "":
                    row[k] = None
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def _merged_csv(fname: str, key_cols: list[str]) -> list[dict]:
    merged, seen = [], set()
    for directory in (RESULTS_PATH, ADDITIONAL_RESULTS_PATH):
        p = directory / fname
        if not p.exists():
            continue
        for r in _read_csv(p):
            k = tuple(r[c] for c in key_cols)
            if k not in seen:
                seen.add(k)
                merged.append(r)
    return merged


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/archive")
def archive():
    return render_template("archive.html", active="archive")


@app.get("/results")
def results():
    return render_template("results.html")


@app.get("/results/graphs/<path:fname>")
def results_graph(fname):
    """Serve the generated PNG/SVG results from results/graphs (not web/static)."""
    return send_from_directory(GRAPHS_PATH, fname)


@app.get("/api/results")
def api_results():
    """Final results page data, read from results/metrics & results/tables."""
    def read(name):
        p = METRICS_PATH / name
        return _read_csv(p) if p.exists() else []

    graphs = sorted(
        p.name for p in GRAPHS_PATH.glob("*.png")
    ) + sorted(
        p.name for p in GRAPHS_PATH.glob("*.svg")
    )
    return jsonify(
        models=read("final_model_comparison.csv"),
        scanner=read("scanner_wise_results.csv"),
        thresholds=read("threshold_summary.csv"),
        auth=read("authentication_summary.csv"),
        graphs=graphs,
        sources={
            "tables": [p.name for p in sorted(RESULTS_PATH.glob("*.csv"))],
            "metrics": [p.name for p in sorted(METRICS_PATH.glob("*.csv"))],
        },
    )


@app.get("/api/results-full")
def api_results_full():
    """Full results with all 14 models split by family, from results_summary.csv."""
    def read(name):
        p = METRICS_PATH / name
        return _read_csv(p) if p.exists() else []

    try:
        rows = _merged_csv("results_summary.csv", ["model", "channel", "split"])
    except FileNotFoundError:
        rows = []

    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key.setdefault((r["model"], r["channel"]), {}).setdefault(r["split"], r)

    all_models = []
    for (model, channel), splits in sorted(by_key.items()):
        all_models.append({
            "model": model, "channel": channel,
            "val": splits.get("val") or {},
            "test": splits.get("test") or {},
        })

    TRANSFORMER = [
        "efficientformer_l1", "convnext_tiny", "mobilevit_xs", "swin_tiny",
        "vit_b16", "deit_small", "tinyvit_5m", "pit_ti",
    ]
    CNN = ["efficientnet_b3", "mobilenet", "resnet50", "resnet152", "vgg16", "lenet5"]

    def best_for(model_name):
        candidates = [m for m in all_models if m["model"] == model_name]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m["test"].get("accuracy", 0) or 0)

    transformer = [best_for(m) for m in TRANSFORMER if best_for(m)]
    cnn = [best_for(m) for m in CNN if best_for(m)]

    graphs = sorted(
        p.name for p in GRAPHS_PATH.glob("*.png")
    ) + sorted(
        p.name for p in GRAPHS_PATH.glob("*.svg")
    )

    verif_path = RESULTS_PATH / "efficientformer_l1_rgb_verification.csv"
    return jsonify(
        transformer=transformer,
        cnn=cnn,
        verification=_read_csv(verif_path) if verif_path.exists() else [],
        scanner=read("scanner_wise_results.csv"),
        thresholds=read("threshold_summary.csv"),
        auth=read("authentication_summary.csv"),
        graphs=graphs,
    )


@app.get("/api/repo")
def api_repo():
    """Inventory of the project's own (pre-new_result) artifacts on disk."""
    checkpoints = []
    for p in sorted((BASE / "checkpoints").rglob("*")):
        if p.is_file() and p.suffix == ".pt":
            rel = p.relative_to(BASE / "checkpoints")
            checkpoints.append({
                "category": rel.parts[0],
                "name": p.name,
                "size_mb": round(p.stat().st_size / 1e6, 1),
                "binary": "binary" in p.name,
            })

    tables = []
    for p in sorted(RESULTS_PATH.glob("*.csv")):
        rows = _read_csv(p)
        tables.append({
            "name": p.name,
            "columns": list(rows[0]) if rows else [],
            "rows": rows,
        })

    metrics = []
    for p in sorted(METRICS_PATH.glob("*.csv")):
        rows = _read_csv(p)
        metrics.append({
            "name": p.name,
            "columns": list(rows[0]) if rows else [],
            "rows": rows,
        })

    graphs = sorted(
        p.name for p in GRAPHS_PATH.glob("*.png")
    ) + sorted(
        p.name for p in GRAPHS_PATH.glob("*.svg")
    )

    logs = []
    for directory in (LOGS_PATH, LOGS_PATH / "colab"):
        if not directory.exists():
            continue
        src = "training" if directory == LOGS_PATH else "training/colab"
        for p in sorted(directory.glob("*.csv")):
            rows = _read_csv(p)
            final = rows[-1] if rows else {}
            logs.append({
                "run": p.stem,
                "source": src,
                "epochs": len(rows),
                "final_val_acc": final.get("val_acc"),
            })
    logs.sort(key=lambda r: (r["source"], r["run"]))

    return jsonify(checkpoints=checkpoints, tables=tables, metrics=metrics,
                   graphs=graphs, logs=logs)


@app.get("/api/multiclass")
def api_multiclass():
    try:
        rows = _merged_csv("results_summary.csv", ["model", "channel", "split"])
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 500

    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key.setdefault((r["model"], r["channel"]), {}).setdefault(r["split"], r)

    models = []
    for (model, channel), splits in sorted(by_key.items()):
        models.append({
            "model": model,
            "channel": channel,
            "val": splits.get("val") or {},
            "test": splits.get("test") or {},
        })
    return jsonify(models=models)


@app.get("/api/binary")
def api_binary():
    try:
        rows = _merged_csv("results_binary_summary.csv", ["model", "channel", "scanner", "split"])
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 500

    def scanner_key(r):
        s = r.get("scanner")
        return (0, -1) if str(s).lower() == "mean" else (0, int(float(s)))

    rows.sort(key=lambda r: (r.get("channel", ""), *scanner_key(r)))
    return jsonify(rows=rows)


@app.get("/api/curves")
def api_curves():
    # Project dir wins; colab/ only fills runs that exist nowhere else.
    out = {}
    for directory in (LOGS_PATH, LOGS_PATH / "colab"):
        for p in sorted(directory.glob("*.csv")):
            if p.stem not in out:
                try:
                    out[p.stem] = _read_csv(p)
                except FileNotFoundError:
                    continue
    return jsonify(out)


# ---- live inference & authentication ----------------------------------------

TEST_DATA_DIR = BASE / "dataset" / "Dataset2" / "Fingerprint_of_Test1_and_Test2"
VERIFICATION_CSV = RESULTS_PATH / "efficientformer_l1_rgb_verification.csv"

IDENT_CACHE: dict[tuple, ScannerIdentifier] = {}
CACHE_LIMIT = 4


def _verification_context() -> dict:
    """Official Dataset 2 numbers at the frozen 0.3892 threshold, from the
    verification CSV (per the fused EfficientFormer-L1 cosine protocol)."""
    if not VERIFICATION_CSV.exists():
        return {}
    for row in _read_csv(VERIFICATION_CSV):
        if str(row.get("split", "")).lower() == "test_dataset2":
            return {
                "split": "test_dataset2",
                "threshold": row.get("threshold"),
                "far": row.get("far"),
                "frr": row.get("frr"),
                "eer": row.get("eer"),
            }
    return {}


def _cached_ident(model: str, channel: str) -> ScannerIdentifier:
    key = (model, channel)
    if key not in IDENT_CACHE:
        if len(IDENT_CACHE) >= CACHE_LIMIT:
            IDENT_CACHE.pop(next(iter(IDENT_CACHE)))
        IDENT_CACHE[key] = ScannerIdentifier(model, channel)
    return IDENT_CACHE[key]


def _image_bytes(form) -> tuple[str | None, bytes]:
    """Accept either an upload or a validated Dataset 2 sample name."""
    if "image" in form.files:
        b = form.files["image"].read()
        if b:
            return None, b
    name = form.form.get("sample", "")
    if name:
        if not re.fullmatch(r"Sensor\d+\(Test[12]\)\d+\.bmp", name):
            return "invalid sample name", b""
        path = (TEST_DATA_DIR / name).resolve()
        if not str(path).startswith(str(TEST_DATA_DIR.resolve())) or not path.exists():
            return "sample not found", b""
        return None, path.read_bytes()
    return "missing image", b""


@app.get("/auth")
def auth():
    return render_template("auth.html")


@app.get("/login")
def login():
    return redirect(url_for("auth"))


@app.get("/infer")
def infer():
    return redirect(url_for("auth"))


# Deck verification models — only those with multiclass checkpoints on disk.
# When `model` is efficientformer_l1 and `channel` is rgb the scanner
# verification uses the official fused 448-D cosine verifier (embedding
# path); otherwise it falls back to the frozen one-vs-all binary verifier.
# The option label carries the gate tag so the Verify hint can show it.
DECK_VERIFICATION_MODELS = [
    {"model": "efficientformer_l1", "channel": "rgb",
     "gate": "embedding · cosine · 448-D"},
    {"model": "mobilenet", "channel": "grayscale",
     "gate": "binary · sigmoid"},
    {"model": "mobilenet", "channel": "rgb",
     "gate": "binary · sigmoid"},
    {"model": "vgg16", "channel": "grayscale",
     "gate": "binary · sigmoid"},
]


@app.get("/api/options")
def api_options():
    live = {f"{m['model']}|{m['channel']}" for m in available_models()}
    deck_models = [
        {"model": d["model"], "channel": d["channel"], "gate": d["gate"]}
        for d in DECK_VERIFICATION_MODELS
        if f"{d['model']}|{d['channel']}" in live
    ]
    return jsonify(
        deck_models=deck_models,
        scanners=list(range(1, 11)),
    )


@app.get("/api/samples")
def api_samples():
    if not TEST_DATA_DIR.exists():
        return jsonify(samples=[])
    samples = []
    by_scanner: dict[int, list[Path]] = {}
    for p in sorted(TEST_DATA_DIR.glob("Sensor*.bmp")):
        m = re.match(r"^Sensor(\d+)", p.stem)
        n = int(m.group(1)) if m else 0
        by_scanner.setdefault(n, []).append(p)
    for scanner, paths in sorted(by_scanner.items()):
        for p in paths:
            samples.append({"name": p.name, "scanner": scanner})
    return jsonify(samples=samples)


@app.post("/api/identify")
def api_identify():
    err, data = _image_bytes(request)
    if err:
        return jsonify(error=err), 400
    model = request.form.get("model") or "mobilenet"
    channel = request.form.get("channel") or "grayscale"
    try:
        ident = _cached_ident(model, channel)
        return jsonify(ident.predict_bytes(data))
    except (FileNotFoundError, ValueError) as e:
        return jsonify(error=str(e)), 400


# ---- model-dependent verify dispatch ----------------------------------------

BINARY_VERIFY_CACHE: dict = {}  # {(m,c,n): one-vs-all ScannerVerifier}
CACHE_LIMIT = 12
SCANNER_GATE = None  # test hook: set to a fake gate to avoid disk I/O


def _scanner_gate(model, channel, scanner):
    """One-vs-all binary gate backed by the frozen per-scanner threshold.
    Binary checks are individually cached (keyed on model+channel+scanner);
    tests swap in a fake via SCANNER_GATE."""
    if SCANNER_GATE is not None:
        return SCANNER_GATE
    key = (model, channel, scanner)
    if key not in BINARY_VERIFY_CACHE:
        if len(BINARY_VERIFY_CACHE) >= CACHE_LIMIT:
            BINARY_VERIFY_CACHE.pop(next(iter(BINARY_VERIFY_CACHE)))
        from authentication.scanner_identification_binary import one_vs_all_gate
        BINARY_VERIFY_CACHE[key] = one_vs_all_gate(model, channel, scanner)
    return BINARY_VERIFY_CACHE[key]


@app.post("/api/verify")
def api_verify():
    """Model-dependent scanner verification.

    EfficientFormer-L1 RGB uses the official fused 448-D embedding +
    cosine verifier; the CNN models use the frozen one-vs-all binary
    verifier. The response carries the method label and (for the fused
    path only) the official Dataset 2 FAR / FRR / EER for the
    operating point so the deck can render Genuine / Impostor -> metrics."""
    err, data = _image_bytes(request)
    if err:
        return jsonify(error=err), 400
    scanner_str = request.form.get("scanner")
    if not scanner_str:
        return jsonify(error="scanner must be an integer 1..10"), 400
    try:
        scanner = int(scanner_str)
    except ValueError:
        return jsonify(error="scanner must be an integer 1..10"), 400
    model = (request.form.get("model") or "efficientformer_l1").lower()
    channel = (request.form.get("channel") or "rgb").lower()

    if model == "efficientformer_l1" and channel == "rgb":
        try:
            result = _fused_verifier().verify_bytes(data, scanner)
        except (FileNotFoundError, ValueError, AssertionError) as e:
            return jsonify(error=str(e)), 400
        result["method"] = "embedding · cosine"
        result["official"] = _verification_context()
    else:
        gate = _scanner_gate(model, channel, scanner)
        result = gate.verify_bytes(data)
        result["method"] = "binary · sigmoid"
    return jsonify(result)


@app.post("/api/mfa")
def api_mfa():
    err, data = _image_bytes(request)
    if err:
        return jsonify(error=err), 400
    model = request.form.get("model") or "mobilenet"
    channel = request.form.get("channel") or "grayscale"
    password = request.form.get("password", "")
    server_password = request.form.get("server_password") or None
    declared_str = request.form.get("declared_scanner")
    if not declared_str:
        return jsonify(error="declared_scanner must be an integer 1..10"), 400
    try:
        declared = int(declared_str)
    except ValueError:
        return jsonify(error="declared_scanner must be an integer 1..10"), 400
    if not password:
        return jsonify(error="password required"), 400
    try:
        ident = _cached_ident(model, channel)
        result = authentication_flow.authenticate(
            data, password, declared, ident, server_password=server_password,
        )
        return jsonify(result)
    except (FileNotFoundError, ValueError) as e:
        return jsonify(error=str(e)), 400


# ---- register / login (full MFA auth_flow) --------------------------------

FUSED_VERIFIER = None  # lazy singleton; tests may swap in a fake


def _fused_verifier():
    """Lazy EfficientFormer fused-cosine verifier (builds the val-split
    reference templates once, on first use)."""
    global FUSED_VERIFIER
    if FUSED_VERIFIER is None:
        from authentication.scanner_fused_verification import (
            FusedScannerVerifier,
        )
        FUSED_VERIFIER = FusedScannerVerifier()
    return FUSED_VERIFIER


def _auth_images():
    """All uploaded 'image' files plus comma-separated Dataset 2 sample names."""
    images = []
    for f in request.files.getlist("image"):
        b = f.read()
        if b:
            images.append(Image.open(io.BytesIO(b)))
    for name in request.form.get("samples", "").split(","):
        name = name.strip()
        if not name:
            continue
        if not re.fullmatch(r"Sensor\d+\(Test[12]\)\d+\.bmp", name):
            raise ValueError(f"invalid sample name: {name}")
        path = (TEST_DATA_DIR / name).resolve()
        if not str(path).startswith(str(TEST_DATA_DIR.resolve())) or not path.exists():
            raise ValueError(f"sample not found: {name}")
        images.append(Image.open(path))
    return images


def _parsed_scanner() -> int:
    raw = request.form.get("declared_scanner", "")
    if not raw:
        raise ValueError("declared_scanner must be an integer 1..10")
    try:
        n = int(raw)
    except ValueError:
        raise ValueError("declared_scanner must be an integer 1..10")
    if not 1 <= n <= 10:
        raise ValueError("declared_scanner must be an integer 1..10")
    return n


def _qrcode_data_uri(text: str) -> str:
    """Render `text` as a PNG QR code, returned as a base64 data URI for the
    browser overlay (Google Authenticator scans it straight from the page)."""
    qr = qrcode.make(text)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/api/auth/register")
def api_auth_register():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("password2", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        return jsonify(error="username must be 3-32 letters, digits, _ . -"), 400
    if len(password) < 4:
        return jsonify(error="password too short (min 4 characters)"), 400
    if password != confirm:
        return jsonify(error="passwords do not match"), 400
    try:
        declared = _parsed_scanner()
        images = _auth_images()
    except ValueError as e:
        return jsonify(error=str(e)), 400
    if not images:
        return jsonify(error="at least one fingerprint image is required for enrollment"), 400
    try:
        r = auth_flow.register(username, password, images, declared)
    except ValueError as e:  # duplicate user
        return jsonify(error=str(e)), 409
    otpauth = (f"otpauth://totp/THFSA:{username}"
               f"?secret={r['passkey_secret']}&issuer=THFSA")
    session["device_secret"] = r["device_secret"]
    return jsonify(
        ok=True,
        user_id=r["user_id"],
        passkey_secret=r["passkey_secret"],   # one-time: shown in the overlay
        device_secret=r["device_secret"],
        otpauth_uri=otpauth,
        qr_data=_qrcode_data_uri(otpauth),
    )


@app.post("/api/auth/login")
def api_auth_login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    passcode = request.form.get("passcode", "").strip()
    if not username or not password or not passcode:
        return jsonify(error="username, password and passkey code are required"), 400
    try:
        declared = _parsed_scanner()
        images = _auth_images()
    except ValueError as e:
        return jsonify(error=str(e)), 400
    if not images:
        return jsonify(error="a fingerprint image is required"), 400
    model = (request.form.get("model") or "efficientformer_l1").lower()
    channel = (request.form.get("channel") or "rgb").lower()
    try:
        # Gate 4 is model-dependent: efficientformer_l1/rgb uses the fused
        # 448-D cosine verifier; the CNN models use the one-vs-all binary
        # gate. Only the matching verifier is built.
        if model == "efficientformer_l1" and channel == "rgb":
            fused, gate = _fused_verifier(), None
        else:
            fused, gate = None, _scanner_gate(model, channel, declared)
        result = auth_flow.authenticate(
            username, password, passcode, images[0], declared,
            scanner_model=model, scanner_channel=channel,
            scanner_gate=gate, fused_verifier=fused,
            device_secret=session.get("device_secret"),
        )
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
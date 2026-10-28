"""make_verification_graphs.py — official fused cosine-embedding scanner
verification graphs for the Mid Review PPT (EfficientFormer-L1 · RGB).

RUN FROM REPO ROOT:
    .venv\\Scripts\\python.exe experiments\\make_verification_graphs.py

PROTOCOL (as recorded in results/tables/efficientformer_l1_rgb_verification.csv):
  - Frozen EfficientFormer-L1 (RGB) multiclass checkpoint.
  - Penultimate (pre-head) 448-d embeddings per image.
  - All-pairs cosine similarity within each split:
      validation (Dataset 1 val, 606 imgs) => C(606,2)=183,315 pairs
      test      (Dataset 2,    200 imgs)  => C(200,2)=19,900 pairs
  - Genuine pair = same scanner; impostor = different scanner.
  - Threshold = val EER point (0.389249...), frozen, then scored on Dataset 2.

The script asserts its recomputed numbers against the official table BEFORE
writing any PNG, so a mismatch fails loudly instead of emitting a wrong figure.

Outputs (300 DPI, white background, one clean graph per file):
  results/graphs/verification/efficientformer_l1_rgb_roc_curve.png
  results/graphs/verification/efficientformer_l1_rgb_dataset2_score_distribution.png
  results/graphs/verification/efficientformer_l1_rgb_far_frr_threshold.png
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)
os.chdir(SRC)
from config import MODEL_INPUT_SIZE
from dataset import load_or_build_split, scan_test_dataset, R307Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from models import build_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(REPO, "results", "graphs", "verification")

# Tolerances: processor/ALGO float drift only; structural changes would break these.
TOL_AUC, TOL_TH, TOL_RATE = 1e-4, 1e-3, 1e-3

OFFICIAL = {
    "val_auc": 0.9952045869743981,
    "val_th": 0.38924925516520564,
    "val_eer": 0.025255013594642432,
    "test_auc": 0.9864404385964912,
    "test_far": 0.10261111111111111,
    "test_frr": 0.03,
}

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "legend.fontsize": 13,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})
GREEN, RED, BLUE, GRAY = "#2ca02c", "#c44e52", "#4c72b0", "#7f7f7f"


def relocated(p):
    normalized = str(p).replace("\\", "/")
    marker = "/dataset/"
    index = normalized.lower().find(marker)
    if index >= 0:
        relative = normalized[index + 1:]
    elif normalized.lower().startswith("dataset/"):
        relative = normalized
    else:
        return str(p)
    return os.path.join(REPO, *relative.split("/"))


def embed(samples, m, model_name, channel):
    size = MODEL_INPUT_SIZE[model_name]
    dl = DataLoader(R307Dataset(samples, size, channel), batch_size=32,
                    shuffle=False, num_workers=0)
    feats = []
    def hook(module, inp, out):
        t = inp[0].detach()
        if t.ndim > 2:
            t = torch.nn.functional.adaptive_avg_pool2d(t, 1).flatten(1)
        feats.append(t.numpy())
    h = m.head.register_forward_hook(hook)
    with torch.no_grad():
        for x, _ in dl:
            m(x)
    h.remove()
    return np.concatenate(feats)


def pairs(E, labels):
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = En @ En.T
    iu = np.triu_indices(len(E), k=1)
    scores = S[iu]
    y = (labels[iu[0]] == labels[iu[1]]).astype(int)
    return scores, y


def main():
    os.makedirs(OUT, exist_ok=True)

    val, test = load_or_build_split()[1], scan_test_dataset()
    for s in val:
        s.path = relocated(s.path)

    model_name, channel = "efficientformer_l1", "rgb"
    m = build_model(model_name, num_classes=10, in_channels=3)
    ck = torch.load(os.path.join(REPO, "checkpoints", "transformer",
                                 "efficientformer_l1_rgb_best.pt"),
                    map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("state_dict", ck))
    m.eval()

    E = embed(val, m, model_name, channel)
    F = embed(test, m, model_name, channel)

    yv = np.array([s.scanner_idx for s in val], dtype=int)
    yt = np.array([s.scanner_idx for s in test], dtype=int)
    sv, yv_ = pairs(E, yv)
    st, yt_ = pairs(F, yt)

    val_auc = roc_auc_score(yv_, sv)
    val_fpr, val_tpr, val_th = roc_curve(yv_, sv)
    val_far = val_fpr
    val_frr = 1.0 - val_tpr
    idx = int(np.argmin(np.abs(val_far - val_frr)))
    eer = (val_far[idx] + val_frr[idx]) / 2.0
    th_star = float(val_th[idx])

    test_auc = roc_auc_score(yt_, st)
    far_t = float(((st >= th_star) & (yt_ == 0)).sum()) / float((yt_ == 0).sum())
    frr_t = float(((st < th_star) & (yt_ == 1)).sum()) / float((yt_ == 1).sum())

    # ---- verify against the official table before drawing anything ----
    assert abs(val_auc - OFFICIAL["val_auc"]) < TOL_AUC
    assert abs(th_star - OFFICIAL["val_th"]) < TOL_TH
    assert abs(eer - OFFICIAL["val_eer"]) < TOL_RATE
    assert abs(test_auc - OFFICIAL["test_auc"]) < TOL_AUC
    assert abs(far_t - OFFICIAL["test_far"]) < TOL_RATE
    assert abs(frr_t - OFFICIAL["test_frr"]) < TOL_RATE
    print(f"official numbers reproduced: val AUC={val_auc:.6f} th*={th_star:.6f} "
          f"EER={eer:.6f} | test AUC={test_auc:.6f} FAR={far_t:.6f} FRR={frr_t:.6f}")

    # ---- 1. ROC: validation + Dataset 2 on one clean curve pair ----
    fpr_t, tpr_t, _ = roc_curve(yt_, st)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(val_fpr, val_tpr, color=BLUE, lw=2.6, label=f"Validation (D1) — AUC {val_auc:.4f} ({val_auc*100:.2f}%)")
    ax.plot(fpr_t, tpr_t, color=RED, lw=2.6, label=f"Dataset 2 — AUC {test_auc:.4f} ({test_auc*100:.2f}%)")
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Chance")
    ax.scatter([val_far[idx]], [val_tpr[idx]], zorder=5,
               color=BLUE, edgecolor="white", s=60,
               label=f"Val EER operating point ({eer*100:.2f}%)")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (1 − FRR)")
    ax.set_title("EfficientFormer-L1 (RGB)\nScanner Verification ROC — Fused Cosine")
    ax.legend(loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e6e6e6", lw=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "efficientformer_l1_rgb_roc_curve.png"))
    plt.close(fig)

    # ---- 2. Dataset 2 genuine vs impostor score distribution, frozen threshold ----
    bins = np.linspace(min(st.min(), sv.min()), 1.0, 80)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.hist(st[yt_ == 0], bins=bins, density=True, color=RED, alpha=0.55,
            edgecolor="white", label=f"Impostor pairs (n={int((yt_ == 0).sum()):,})")
    ax.hist(st[yt_ == 1], bins=bins, density=True, color=GREEN, alpha=0.55,
            edgecolor="white", label=f"Genuine pairs (n={int((yt_ == 1).sum()):,})")
    ax.axvline(th_star, color="black", lw=2.2, ls="--",
               label=f"Frozen threshold = {th_star:.4f}")
    ax.annotate(f"FAR {far_t*100:.2f}%", xy=(th_star + 0.01, 1.5), color=RED,
                fontsize=12, fontweight="bold")
    ax.annotate(f"FRR {frr_t*100:.2f}%", xy=(min(st.min(), 0.2), 1.5), color=GREEN,
                fontsize=12, fontweight="bold")
    ax.set_xlabel("Cosine similarity (all-pairs, pre-head embedding)")
    ax.set_ylabel("Density")
    ax.set_title("EfficientFormer-L1 (RGB) — Dataset 2\nGenuine vs Impostor Cosine Scores")
    ax.legend(loc="upper right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e6e6e6", lw=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "efficientformer_l1_rgb_dataset2_score_distribution.png"))
    plt.close(fig)

    # ---- 3. FAR & FRR vs threshold (validation), frozen threshold marked ----
    rth = np.unique(np.concatenate((val_th, [th_star, 1.0])))
    rth = np.sort(rth)
    far_roc = np.array([np.mean((sv >= t) & (yv_ == 0)) for t in rth])
    frr_roc = np.array([np.mean((sv < t) & (yv_ == 1)) for t in rth])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(rth, far_roc * 100, color=RED, lw=2.6, label="FAR (validation)")
    ax.plot(rth, frr_roc * 100, color=GREEN, lw=2.6, label="FRR (validation)")
    ax.axvline(th_star, color="black", lw=2.2, ls="--",
               label=f"Frozen threshold = {th_star:.4f}")
    ax.scatter([th_star], [eer * 100], color="black", zorder=6, s=70,
               edgecolor="white", label=f"Val EER = {eer*100:.2f}%")
    ax.text(th_star + 0.004, eer * 100 - 6, f"EER {eer*100:.2f}% @ th = {th_star:.4f}",
            fontsize=12, fontweight="bold")
    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("EfficientFormer-L1 (RGB) — FAR & FRR vs Threshold\n(threshold calibrated on validation, frozen before Dataset 2)")
    ax.legend(loc="center right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e6e6e6", lw=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "efficientformer_l1_rgb_far_frr_threshold.png"))
    plt.close(fig)

    print("wrote:")
    for fn in ("efficientformer_l1_rgb_roc_curve.png",
               "efficientformer_l1_rgb_dataset2_score_distribution.png",
               "efficientformer_l1_rgb_far_frr_threshold.png"):
        print("  ", os.path.relpath(os.path.join(OUT, fn), REPO))


if __name__ == "__main__":
    main()
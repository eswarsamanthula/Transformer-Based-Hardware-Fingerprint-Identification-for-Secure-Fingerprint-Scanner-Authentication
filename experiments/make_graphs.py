"""make_graphs.py — produce ALL evaluation graphs from the frozen official
checkpoints (no retraining, no writes to official outputs).

Graphs-only mode: every deliverable is a figure under results/graphs/<sub>/.
Prediction caches under experiments/predictions/ are an internal intermediate
(regenerated on demand from official checkpoints); they are not deliverables.

Run from repo root:
    python experiments/make_graphs.py --smoke   # import + setup check only
    python experiments/make_graphs.py           # full graph build
"""
import csv
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))

import common                                                  # noqa: E402
from common import (                                           # noqa: E402
    REPO, gen_dir, load_model, predict_probs, val_and_test_samples,
    prediction_csv_path, binary_prediction_csv_path, ckpt_path,
    FINAL_MULTICLASS, BINARY_SETS, FEATURE_MODELS, SCANNER_COLORS,
)
import cache as cache_mod                                      # noqa: E402
from config import MODEL_REGISTRY, MODEL_INPUT_SIZE, NUM_SCANNERS, TEST_DATA_ROOT  # noqa: E402
from models import count_params                                # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from sklearn.metrics import (                                 # noqa: E402
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    roc_auc_score,
)

SEED = 42
np.random.seed(SEED)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
LAB = [f"S{i+1}" for i in range(10)]

T0 = time.time()
def pt(msg):
    print(f"[{time.time()-T0:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# cache readers
# ---------------------------------------------------------------------------
def read_multiclass(model, channel, split):
    """Rows -> (scanner[], pred[], probs Nx10) or None if file missing."""
    p = prediction_csv_path(model, channel, split)
    if not os.path.exists(p):
        return None
    scanners, preds, probs = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            scanners.append(int(r["scanner"]))
            preds.append(int(r["pred"]))
            probs.append([float(r[f"p{i}"]) for i in range(10)])
    return np.array(scanners), np.array(preds), np.array(probs)


def read_binary(model, channel, split):
    """Rows -> (1-based target[], label[], score[])."""
    p = binary_prediction_csv_path(model, channel, split)
    if not os.path.exists(p):
        return None
    tgt, lab, sc = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            tgt.append(int(r["target_scanner"])); lab.append(int(r["label"]))
            sc.append(float(r["score"]))
    return np.array(tgt), np.array(lab), np.array(sc)


def ensure_multiclass(combos):
    for m, ch in combos:
        if not os.path.exists(ckpt_path(m, ch, "multiclass")):
            pt(f"  NO CKPT {m}/{ch} — skipped")
            continue
        for split in ("val", "test"):
            if os.path.exists(prediction_csv_path(m, ch, split)):
                continue
            pt(f"  caching multiclass {m}/{ch}/{split} ...")
            cache_mod.generate_multiclass(m)
            break


def ensure_binary(sets):
    for m, ch in sets:
        for split in ("val", "test"):
            if os.path.exists(binary_prediction_csv_path(m, ch, split)):
                continue
            pt(f"  caching binary {m}/{ch}/{split} ...")
            cache_mod.generate_binary(m)
            break


# ---------------------------------------------------------------------------
# A1 + A2  confusion matrices, per-scanner metrics
# ---------------------------------------------------------------------------
def graph_confusion_per_scanner():
    d = gen_dir("graphs", "confusion_matrices")
    d2 = gen_dir("graphs", "per_scanner")
    for m, ch in FINAL_MULTICLASS:
        if not os.path.exists(ckpt_path(m, ch, "multiclass")):
            pt(f"  SKIP cm {m}/{ch} (no checkpoint; official preserved)")
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for i, split in enumerate(["val", "test"]):
            scanners, preds, _ = read_multiclass(m, ch, split)
            acc = (preds == scanners).mean()
            for j, (fname, norm) in enumerate([("raw", None), ("norm", "row")]):
                cm = confusion_matrix(scanners, preds, labels=list(range(10)))
                if norm == "row":
                    cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
                ax = axes[i][j]
                im = ax.imshow(cm, cmap="viridis", vmin=0,
                               vmax=None if norm is None else 1)
                ax.set_xticks(range(10), LAB, fontsize=8)
                ax.set_yticks(range(10), LAB, fontsize=8)
                ax.set_title(f"{split} — {fname} (acc={acc:.3f})", fontsize=10)
                thr = 0.005 * (cm.max() if norm is None else 1)
                for r in range(10):
                    for c in range(10):
                        if cm[r, c] >= thr:
                            ax.text(c, r, f"{cm[r,c]:.2f}" if norm else f"{int(cm[r,c])}",
                                    ha="center", va="center", fontsize=5.5, color="white")
        fig.suptitle(f"Confusion matrices — {m} ({ch})", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(d, f"cm_{m}_{ch}.png")); plt.close(fig)

        fig2, axs = plt.subplots(1, 2, figsize=(13, 4.6))
        for ax, split in zip(axs, ["val", "test"]):
            scanners, preds, _ = read_multiclass(m, ch, split)
            prec, rec, f1, _ = precision_recall_fscore_support(
                scanners, preds, labels=list(range(10)), zero_division=0)
            accs = np.zeros(10)
            for s in range(10):
                m_ = scanners == s
                accs[s] = accuracy_score(scanners[m_], preds[m_]) if m_.any() else 0.0
            x = np.arange(10); w = 0.2
            ax.bar(x - 1.5*w, prec, w, label="precision", color="#4c72b0")
            ax.bar(x - 0.5*w, rec, w, label="recall", color="#dd8452")
            ax.bar(x + 0.5*w, f1, w, label="F1", color="#55a868")
            ax.bar(x + 1.5*w, accs, w, label="accuracy", color="#c44e52", alpha=0.85)
            ax.set_xticks(x, LAB, fontsize=8)
            ax.set_title(f"per-scanner — {split}", fontsize=10)
            ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
        fig2.suptitle(f"Per-scanner metrics — {m} ({ch})", fontsize=12)
        fig2.tight_layout(); fig2.savefig(os.path.join(d2, f"per_scanner_{m}_{ch}.png")); plt.close(fig2)
        pt(f"  cm + per-scanner {m}/{ch}")


# ---------------------------------------------------------------------------
# A3  multiclass ROC / PR (one-vs-rest)
# ---------------------------------------------------------------------------
def graph_roc_pr_multiclass():
    d = gen_dir("graphs", "roc_pr")
    official = {}
    with open(os.path.join(REPO, "results", "tables", "results_summary.csv")) as f:
        for r in csv.DictReader(f):
            official[(r["model"], r["channel"], r["split"])] = float(r["auc"])
    for m, ch in FINAL_MULTICLASS:
        if not os.path.exists(ckpt_path(m, ch, "multiclass")):
            pt(f"  SKIP roc/pr {m}/{ch} (no checkpoint)")
            continue
        for split in ("val", "test"):
            scanners, preds, probs = read_multiclass(m, ch, split)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            aucs, aps = [], []
            for c in range(10):
                y = (scanners == c).astype(int)
                fpr, tpr, _ = roc_curve(y, probs[:, c])
                au = auc(fpr, tpr); aucs.append(au)
                ax1.plot(fpr, tpr, color=SCANNER_COLORS[c], lw=1.2, label=f"{LAB[c]} (AUC {au:.2f})")
                prec, rec, _ = precision_recall_curve(y, probs[:, c])
                ap = average_precision_score(y, probs[:, c]); aps.append(ap)
                ax2.plot(rec, prec, color=SCANNER_COLORS[c], lw=1.2, label=f"{LAB[c]} (AP {ap:.2f})")
            ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax1.set(xlabel="FPR", ylabel="TPR", title=f"ROC one-vs-rest — {split}\nmacro-AUC={np.mean(aucs):.4f}")
            ax2.set(xlabel="Recall", ylabel="Precision", title=f"PR one-vs-rest — {split}\nmAP={np.mean(aps):.4f}")
            ax1.legend(fontsize=7, loc="lower right"); ax2.legend(fontsize=7, loc="upper right")
            fig.suptitle(f"Multiclass ROC / PR — {m} ({ch})", fontsize=12)
            fig.tight_layout(); fig.savefig(os.path.join(d, f"roc_pr_{m}_{ch}_{split}.png")); plt.close(fig)
            ok = "" if split not in official or abs(official[(m, ch, split)] - np.mean(aucs)) < 0.02 \
                else f"  NOTE: official AUC={official[(m,ch,split)]:.4f} vs computed {np.mean(aucs):.4f}"
            pt(f"  roc/pr {m}/{ch}/{split} macro-AUC={np.mean(aucs):.4f}{ok}")


# ---------------------------------------------------------------------------
# A4  binary ROC / PR + FAR/FRR/ACC + EER + score distributions
# ---------------------------------------------------------------------------
def graph_roc_pr_binary():
    d = gen_dir("graphs", "roc_pr")
    for m, ch in BINARY_SETS:
        for split in ("val", "test"):
            tgt, lab, sc = read_binary(m, ch, split)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            aucs, aps = [], []
            for s in range(10):
                mask = tgt == (s + 1)
                y = lab[mask]; score = sc[mask]
                fpr, tpr, _ = roc_curve(y, score)
                au = auc(fpr, tpr); aucs.append(au)
                ax1.plot(fpr, tpr, color=SCANNER_COLORS[s], lw=1.2, label=f"{LAB[s]} ({au:.3f})")
                prec, rec, _ = precision_recall_curve(y, score)
                ap = average_precision_score(y, score); aps.append(ap)
                ax2.plot(rec, prec, color=SCANNER_COLORS[s], lw=1.2, label=f"{LAB[s]} (AP {ap:.3f})")
            ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax1.set(xlabel="FPR", ylabel="TPR", title=f"binary ROC — {split}\nmean AUC={np.mean(aucs):.4f}")
            ax2.set(xlabel="Recall", ylabel="Precision", title=f"binary PR — {split}\nmean AP={np.mean(aps):.4f}")
            ax1.legend(fontsize=6); ax2.legend(fontsize=6)
            fig.suptitle(f"Binary one-vs-all ROC / PR — {m} ({ch})", fontsize=12)
            fig.tight_layout(); fig.savefig(os.path.join(d, f"roc_pr_binary_{m}_{ch}_{split}.png")); plt.close(fig)
            pt(f"  binary roc/pr {m}/{ch}/{split} meanAUC={np.mean(aucs):.4f}")


def _scanner_curves(tgt, lab, sc, th):
    """Per-scanner {far,frr,acc} arrays over `th` + per-scanner EER/th*."""
    n_th = len(th)
    n_sc = NUM_SCANNERS
    far = np.zeros((n_sc, n_th)); frr = np.zeros((n_sc, n_th)); acc = np.zeros((n_sc, n_th))
    for s in range(n_sc):
        mask = tgt == (s + 1)
        y = lab[mask]; x = sc[mask]
        n0 = max((y == 0).sum(), 1); n1 = max((y == 1).sum(), 1)
        for i, t in enumerate(th):
            pred = (x >= t).astype(int)
            acc[s, i] = accuracy_score(y, pred)
            far[s, i] = ((pred == 1) & (y == 0)).sum() / n0
            frr[s, i] = ((pred == 0) & (y == 1)).sum() / n1
    i_eer = np.argmin(np.abs(far - frr), axis=1)
    eer = (far[np.arange(n_sc), i_eer] + frr[np.arange(n_sc), i_eer]) / 2
    th_star = th[i_eer]
    return far, frr, acc, eer, th_star


def graph_threshold_scores():
    d = gen_dir("graphs", "threshold_analysis")
    d2 = gen_dir("graphs", "score_distributions")
    th = np.linspace(0.0, 1.0, 401)
    frozen = {}
    cp = os.path.join(REPO, "results", "tables", "binary_threshold_calibration.csv")
    if os.path.exists(cp):
        with open(cp) as f:
            for r in csv.DictReader(f):
                frozen[(r["model"], r["channel"], int(r["scanner"]))] = float(r["threshold"])
    for m, ch in BINARY_SETS:
        tgt_v, lab_v, sc_v = read_binary(m, ch, "val")
        tgt_t, lab_t, sc_t = read_binary(m, ch, "test")

        # --- per-scanner FAR / FRR / ACC vs threshold (validation) ---
        far, frr, acc, eer, th_star = _scanner_curves(tgt_v, lab_v, sc_v, th)
        for metric, mat, yl in [("far", far, "FAR"), ("frr", frr, "FRR"),
                                ("acc", acc, "accuracy")]:
            fig, ax = plt.subplots(figsize=(10, 5.2))
            for s in range(10):
                ax.plot(th, mat[s], color=SCANNER_COLORS[s], lw=1.2,
                        label=f"{LAB[s]}" + (f"  EER={eer[s]:.3f}" if metric == "far" else ""))
            if metric == "far":
                ax.scatter(th_star, eer, color="k", marker="x", s=12, zorder=3)
            ax.set(xlabel="threshold (sigmoid score)", ylabel=yl,
                   title=f"{yl.upper()} vs threshold per scanner (VAL) — {m} ({ch})")
            ax.legend(fontsize=6, ncol=2)
            ax.set_ylim(0, 1.05)
            fig.tight_layout(); fig.savefig(os.path.join(d, f"{metric}_th_{m}_{ch}_val.png")); plt.close(fig)

        # --- pooled FAR/FRR/ACC + EER (validation) ---
        far_p = far.mean(0); frr_p = frr.mean(0); acc_p = acc.mean(0)
        i_eer = int(np.argmin(np.abs(far_p - frr_p)))
        eer_p = (far_p[i_eer] + frr_p[i_eer]) / 2
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(th, far_p, color="#c44e52", lw=1.6, label="FAR")
        ax.plot(th, frr_p, color="#4c72b0", lw=1.6, label="FRR")
        ax.plot(th, acc_p, color="#2ca02c", lw=1.4, ls="--", label="accuracy (pooled)")
        ax.axvline(th[i_eer], color="k", ls=":", lw=1)
        ax.text(th[i_eer] + 0.01, 0.55, f"th*={th[i_eer]:.3f}\nEER≈{eer_p:.4f}", fontsize=9)
        ax.set(xlabel="threshold (sigmoid score)", ylabel="rate",
               title=f"FAR / FRR / ACC vs threshold (VAL pooled) — {m} ({ch})\n"
                     f"EER≈{eer_p:.4f} at th≈{th[i_eer]:.3f} (threshold selection on validation only)")
        ax.legend(); ax.set_ylim(0, 1.05)
        fig.tight_layout(); fig.savefig(os.path.join(d, f"far_frr_{m}_{ch}_val.png")); plt.close(fig)

        # --- Dataset2 at the FROZEN validation threshold (per scanner) ---
        if (m, ch) in {(k[0], k[1]) for k in frozen}:
            t_frozen = np.array([frozen[(m, ch, s)] for s in range(1, 11)])
            t_far, t_frr, t_acc = [], [], []
            for s in range(10):
                mask = tgt_t == (s + 1)
                y = lab_t[mask]; x = sc_t[mask]
                pred = (x >= t_frozen[s]).astype(int)
                t_acc.append(accuracy_score(y, pred))
                t_far.append(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
                t_frr.append(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1))
            x = np.arange(10); w = 0.26
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.bar(x - w, t_far, w, color="#c44e52", label="FAR")
            ax.bar(x, t_frr, w, color="#4c72b0", label="FRR")
            ax.bar(x + w, t_acc, w, color="#2ca02c", label="accuracy")
            ax.set(xticks=x, xticklabels=LAB, ylabel="rate",
                   title=f"Dataset2 per scanner at FROZEN validation threshold — {m} ({ch})\n"
                         f"thresholds: " + ", ".join(f"{t:.2f}" for t in t_frozen))
            ax.legend(); ax.set_ylim(0, 1.05)
            fig.autofmt_xdate()
            fig.tight_layout(); fig.savefig(os.path.join(d, f"frozen_test_{m}_{ch}.png")); plt.close(fig)

        # --- genuine vs impostor score distributions ---
        fig2, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        for ax, (tgt, lab, sc_, split) in zip(axs, [(tgt_v, lab_v, sc_v, "val"),
                                                    (tgt_t, lab_t, sc_t, "test")]):
            gen = sc_[lab == 1]; imp = sc_[lab == 0]
            ax.hist(imp, bins=60, density=True, alpha=0.65, color="#4c72b0", label="impostor")
            ax.hist(gen, bins=60, density=True, alpha=0.65, color="#dd8452", label="genuine")
            ax.set(title=f"{split} — {m} ({ch})\nnG={len(gen)}, nI={len(imp)}",
                   xlabel="score", ylabel="density")
            ax.legend(fontsize=8)
        fig2.suptitle("Genuine vs impostor one-vs-all scores (pooled over scanners)")
        fig2.tight_layout(); fig2.savefig(os.path.join(d2, f"scores_{m}_{ch}.png")); plt.close(fig2)
        pt(f"  threshold/det/scores {m}/{ch} pooled-EER≈{eer_p:.4f} "
           f"mean-EER={eer.mean():.4f} med-EER={np.median(eer):.4f}")


def graph_det():
    """DET (probit) curves per scanner + EER summary across binary models."""
    from scipy.stats import norm
    d = gen_dir("graphs", "threshold_analysis")
    th = np.linspace(0.0, 1.0, 401)
    clip = lambda v: np.clip(v, 1e-6, 1 - 1e-6)  # noqa: E731
    all_eer = {}
    for m, ch in BINARY_SETS:
        tgt_v, lab_v, sc_v = read_binary(m, ch, "val")
        far, frr, acc, eer, th_star = _scanner_curves(tgt_v, lab_v, sc_v, th)
        all_eer[m] = (ch, eer)
        fig, ax = plt.subplots(figsize=(8, 6))
        for s in range(10):
            ax.plot(far[s], frr[s], color=SCANNER_COLORS[s], lw=1.2,
                    label=f"{LAB[s]} (EER {eer[s]:.3f})")
        ax.plot(far.mean(0), frr.mean(0), color="k", lw=2.2, ls="--", label="pooled")
        ax.plot([eer.mean()], [eer.mean()], "o", color="k", markersize=6,
                label=f"mean EER={eer.mean():.3f}")
        ax.set_xscale("function", functions=(lambda v: norm.ppf(clip(v)),
                                             lambda v: norm.cdf(v)))
        ax.set_yscale("function", functions=(lambda v: norm.ppf(clip(v)),
                                             lambda v: norm.cdf(v)))
        ax.set_xlabel("FAR (probit scale)")
        ax.set_ylabel("FRR (probit scale)")
        ax.set_title(f"DET curve (per scanner, VAL) — {m} ({ch})")
        ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(d, f"det_{m}_{ch}.png")); plt.close(fig)

    # EER summary across models (per-scanner spread)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{m} ({ch})" for m, (ch, _) in all_eer.items()]
    data = [eer for _, (_, eer) in all_eer.items()]
    parts = ax.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
    for m, (ch, eer) in all_eer.items():
        x = list(labels).index(f"{m} ({ch})") + 1
        ax.annotate(f"mean {eer.mean():.3f}\nmed {np.median(eer):.3f}",
                    (x, min(eer) - 0.02), fontsize=7, ha="center")
    ax.set(ylabel="per-scanner EER (validation)",
           title="EER spread per binary model (mean vs median vs range)")
    fig.tight_layout(); fig.savefig(os.path.join(d, "eer_summary.png")); plt.close(fig)
    pt("  DET + EER summary done")


# ---------------------------------------------------------------------------
# A5  calibration (ECE / NLL / Brier reliability)
# ---------------------------------------------------------------------------
def graph_calibration():
    d = gen_dir("graphs", "calibration")
    for m, ch in [("vgg16", "grayscale"), ("efficientformer_l1", "rgb"), ("swin_tiny", "grayscale")]:
        if not os.path.exists(ckpt_path(m, ch, "multiclass")):
            pt(f"  SKIP calibration {m}/{ch} (no checkpoint)")
            continue
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for ax, split in zip(axs, ["val", "test"]):
            scanners, preds, probs = read_multiclass(m, ch, split)
            conf = probs.max(axis=1); correct = (preds == scanners).astype(float)
            n = len(conf)
            nll = float(np.mean(-np.log(probs[np.arange(n), scanners] + 1e-12)))
            brier = float(np.mean(np.sum((np.eye(10)[scanners] - probs) ** 2, axis=1)))
            nb = 20
            acc_t, c_t = [], []
            edges = np.linspace(0, 1, nb + 1)
            for k in range(nb):
                m_ = (conf >= edges[k]) & (conf <= edges[k + 1])
                if m_.sum() > 0:
                    acc_t.append(correct[m_].mean()); c_t.append(conf[m_].mean())
            ece = float(np.mean([abs(a - c) for a, c in zip(acc_t, c_t)]))
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.scatter(c_t, acc_t, color="#4c72b0", s=30)
            ax.plot(c_t, acc_t, color="#4c72b0", lw=1)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="frequency",
                   title=f"{split}\nECE={ece:.4f}  NLL={nll:.4f}  Brier={brier:.4f}")
        fig.suptitle(f"Reliability calibration — {m} ({ch})", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(d, f"calibration_{m}_{ch}.png")); plt.close(fig)
        pt(f"  calibration {m}/{ch}")


# ---------------------------------------------------------------------------
# A6  UMAP feature space (embeddings fed to the classifier head)
# ---------------------------------------------------------------------------
def extract_features(model_name, channel, samples):
    import torch
    from dataset import R307Dataset
    from torch.utils.data import DataLoader
    model = load_model(model_name, channel, num_classes=10)
    size = MODEL_INPUT_SIZE[model_name]
    dl = DataLoader(R307Dataset(samples, size, channel), batch_size=32, shuffle=False, num_workers=0)
    feats = []
    def hook(module, inp, out):
        t = inp[0].detach()
        if t.ndim > 2:  # spatial features (Swin, CNNs) → global-average-pool
            t = torch.nn.functional.adaptive_avg_pool2d(t, 1).flatten(1)
        feats.append(t.numpy())
    h = model.head.register_forward_hook(hook)
    with torch.no_grad():
        for x, _ in dl:
            model(x)
    h.remove()
    return np.concatenate(feats)


def graph_umap():
    import umap
    d = gen_dir("graphs", "feature_space")
    for m, ch in FEATURE_MODELS:
        pt(f"  UMAP features {m}/{ch} ...")
        val_s, test_s = val_and_test_samples(ch)
        combined = val_s + test_s
        F = extract_features(m, ch, combined)
        y = np.array([s.scanner_idx for s in combined])
        is_test = np.array([False] * len(val_s) + [True] * len(test_s))
        red = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED, n_jobs=1).fit_transform(F)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
        for ax, mask, title in [(ax1, ~is_test, "validation (Dataset1)"),
                                (ax2, is_test, "test (Dataset2)")]:
            for s in range(10):
                sel = mask & (y == s)
                ax.scatter(red[sel, 0], red[sel, 1], s=9, color=SCANNER_COLORS[s],
                           label=LAB[s], alpha=0.85)
            ax.set(title=title, xticks=[], yticks=[])
            if title.startswith("validation"):
                ax.legend(fontsize=7, markerscale=1.5, loc="best")
        fig.suptitle(f"UMAP of penultimate embeddings — {m} ({ch}) (seed {SEED})", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(d, f"umap_{m}_{ch}.png")); plt.close(fig)
        pt(f"  UMAP done {m}/{ch}")


# ---------------------------------------------------------------------------
# A7  Dataset2 user-wise (Test1/C test)
# ---------------------------------------------------------------------------
def graph_user_wise():
    import common as C
    d = gen_dir("graphs", "user_generalization")
    for m, ch in FINAL_MULTICLASS:
        if not os.path.exists(ckpt_path(m, ch, "multiclass")):
            pt(f"  SKIP user-wise {m}/{ch} (no checkpoint)")
            continue
        scanners, preds, _ = read_multiclass(m, ch, "test")
        users = np.array([C.user_id(os.path.basename(row["filepath"])) for row in
                          csv.DictReader(open(prediction_csv_path(m, ch, "test")))])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
        xp = np.arange(2)
        for i, u in enumerate(["UserA", "UserB"]):
            mask = users == u
            p_, r_, f1_, _ = precision_recall_fscore_support(
                scanners[mask], preds[mask], labels=list(range(10)),
                zero_division=0, average="macro")
            acc = accuracy_score(scanners[mask], preds[mask])
            ax1.bar(xp + (i - 0.5) * 0.38, [acc, f1_], 0.38,
                    color=SCANNER_COLORS[i], label=f"{u} (n={mask.sum()})")
        ax1.set(xticks=xp, xticklabels=["accuracy", "macro F1"], ylabel="score",
                title=f"Dataset2 per-user accuracy & F1 — {m} ({ch})", ylim=(0, 1.05))
        ax1.legend()
        x = np.arange(10); w = 0.38
        for i, u in enumerate(["UserA", "UserB"]):
            accs = []
            for s in range(10):
                m_ = (users == u) & (scanners == s)
                accs.append(accuracy_score(scanners[m_], preds[m_]) if m_.sum() else 0.0)
            ax2.bar(x + (i - 0.5) * w, accs, w, color=SCANNER_COLORS[i],
                    label="Test1" if u == "UserA" else "Test2")
        ax2.set(xticks=x, xticklabels=LAB, ylabel="accuracy",
                title="Dataset2 per-scanner per-user accuracy")
        ax2.legend(fontsize=8); ax2.set_ylim(0, 1.05)
        fig.tight_layout(); fig.savefig(os.path.join(d, f"user_wise_{m}_{ch}.png")); plt.close(fig)
        pt(f"  user-wise {m}/{ch}")


# ---------------------------------------------------------------------------
# A8  explainability: Grad-CAM (vgg16) + input×gradient (others)
# ---------------------------------------------------------------------------
def load_image_tensor(sample, model_name, channel):
    import torch
    from PIL import Image
    size = MODEL_INPUT_SIZE[model_name]
    img = Image.open(sample.path)
    img = img.convert("L") if channel == "grayscale" else img.convert("RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, "float32") / 255.0
    if channel == "grayscale":
        x = torch.from_numpy(arr)[None, None]           # (1,1,H,W)
    else:
        x = torch.from_numpy(arr.transpose(2, 0, 1))[None]
    return img, x


def graph_explainability():
    import torch
    d = gen_dir("graphs", "explainability")
    val_s, test_s = val_and_test_samples("grayscale")
    idx_s5 = next(i for i, s in enumerate(val_s) if s.scanner_idx == 4)
    s2t2 = next(s for s in test_s if s.path.endswith("Sensor2(Test2)6.bmp"))
    s10t2 = next(s for s in test_s if s.path.endswith("Sensor10(Test2)5.bmp"))
    s1t1 = next(s for s in test_s if s.path.endswith("Sensor1(Test1)1.bmp"))
    chosen = [("val Sensor1", val_s[0]), ("val Sensor5", val_s[idx_s5]),
              ("D2 Sensor1(Test1)1", s1t1),
              ("D2 Sensor2(Test2)6 (known-fail)", s2t2),
              ("D2 Sensor10(Test2)5", s10t2)]
    for m, ch in [("vgg16", "grayscale"), ("efficientformer_l1", "rgb"),
                  ("swin_tiny", "grayscale")]:
        if not os.path.exists(ckpt_path(m, ch, "multiclass")):
            pt(f"  SKIP explainability {m}/{ch} (no checkpoint)")
            continue
        model = load_model(m, ch, num_classes=10)
        fig, axs = plt.subplots(len(chosen), 2,
                                figsize=(8.5, 2 * len(chosen) + 1))
        for i, (name, sample) in enumerate(chosen):
            img, x = load_image_tensor(sample, m, ch)
            sal = None
            if m == "vgg16":
                # Grad-CAM: last Conv2d in backbone.features
                conv_name = [nm for nm, md in model.backbone.features.named_modules()
                             if isinstance(md, torch.nn.Conv2d)][-1]
                last_conv = dict(model.backbone.features.named_modules())[conv_name]
                act = {}
                def hook(mod, inp, out):
                    act["a"] = out
                    out.retain_grad()
                h = last_conv.register_forward_hook(hook)
                x.requires_grad_(True)
                out_logits = model(x)
                cls = int(out_logits.argmax(1).item())
                model.zero_grad()
                out_logits[0, cls].backward()
                h.remove()
                A = act["a"][0]                       # (C,H,W)
                w = A.grad.mean(dim=(1, 2))           # (C,)
                cam = torch.relu((w[:, None, None] * A).sum(dim=0))
                cam = cam.detach().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
                from PIL import Image as PIm
                cam = np.asarray(cam)
                cam_img = PIm.fromarray((cam * 255).astype("uint8"), "L").resize(
                    (img.size[1], img.size[0]), PIm.Resampling.BILINEAR)
                sal = np.asarray(cam_img, "float32") / 255.0
                method = "Grad-CAM"
            else:
                x.requires_grad_(True)
                out_logits = model(x)
                cls = int(out_logits.argmax(1).item())
                model.zero_grad()
                out_logits[0, cls].backward()
                g = x.grad[0]
                sal = (g * x[0]).abs().amax(0).detach().numpy()
                sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)
                method = "input×gradient"
            axs[i][0].imshow(np.asarray(img), cmap="gray")
            axs[i][0].set(xticks=[], yticks=[], title=f"{name}\npred={LAB[cls]}")
            axs[i][1].imshow(np.asarray(img), cmap="gray")
            axs[i][1].imshow(sal, cmap="jet", alpha=0.55)
            axs[i][1].set(xticks=[], yticks=[], title=method)
        fig.suptitle(f"Explainability — {m} ({ch})", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(d, f"explain_{m}_{ch}.png")); plt.close(fig)
        pt(f"  explainability {m}/{ch}")


# ---------------------------------------------------------------------------
# A9  image quality (NFIQ2 unavailable -> labeled Laplacian/contrast/entropy)
# ---------------------------------------------------------------------------
def graph_quality():
    try:
        import nfiq2  # noqa: F401
        nfiq_ok = True
    except Exception:
        nfiq_ok = False
    from scipy import ndimage
    d = gen_dir("graphs", "quality")
    val_s, test_s = val_and_test_samples("grayscale")
    all_s = val_s + test_s
    splits = np.array([0] * len(val_s) + [1] * len(test_s))
    n = len(all_s)
    st = np.empty(n); ct = np.empty(n); et = np.empty(n); sc = np.empty(n)
    from PIL import Image
    for i, s in enumerate(all_s):
        img = np.asarray(Image.open(s.path).convert("L"), "float32") / 255.0
        lap = ndimage.laplace(img)
        st[i] = float((lap ** 2).mean())
        ct[i] = float(img.std())
        h = np.histogram(img, bins=64, density=True)[0]
        et[i] = float(-np.sum(h * np.log(h + 1e-12)))
        sc[i] = s.scanner_idx
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, vals, t in zip(axs, [st, ct, et],
                           ["Laplacian variance (sharpness)", "std contrast", "entropy"]):
        ax.hist(vals, bins=40, color="#4c72b0", alpha=0.7)
        ax.set(title=f"{t}\n(all n={n} Dataset1+Dataset2 images)",
               xlabel="value", ylabel="count")
    fig.suptitle("Image quality statistics (NFIQ2 unavailable — labeled substitute analysis)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(d, "quality_histograms.png")); plt.close(fig)

    fig3, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, vals, t in zip(axs, [st, ct, et],
                           ["Laplacian variance (sharpness)", "std contrast", "entropy"]):
        ax.boxplot([vals[sc == s] for s in range(10)], tick_labels=LAB, showfliers=False)
        ax.set(title=f"{t} by scanner", ylabel="value")
    fig3.suptitle("Quality distribution by scanner (val + test combined)")
    fig3.tight_layout(); fig3.savefig(os.path.join(d, "quality_by_scanner.png")); plt.close(fig3)

    scanners, preds, _ = read_multiclass("vgg16", "grayscale", "test")
    accs = np.array([accuracy_score(scanners[scanners == s], preds[scanners == s]) if (scanners == s).any() else 0.0
                     for s in range(10)])
    means = {"sharp": [st[sc == s].mean() for s in range(10)],
             "contrast": [ct[sc == s].mean() for s in range(10)],
             "entropy": [et[sc == s].mean() for s in range(10)]}
    fig2, ax = plt.subplots(figsize=(7, 5))
    rr = []
    for f, col in zip(means, ["#c44e52", "#4c72b0", "#2ca02c"]):
        r = float(np.corrcoef(means[f], list(accs))[0, 1]); rr.append(r)
        ax.scatter(means[f], accs, color=col, label=f"{f} r={r:.2f}")
    ax.set(xlabel="scanner-mean quality", ylabel="vgg16-g test accuracy",
           title="Scanner-level quality vs accuracy")
    ax.legend(); fig2.tight_layout()
    fig2.savefig(os.path.join(d, "quality_vs_accuracy.png")); plt.close(fig2)
    pt(f"  quality graphs (NFIQ2 available: {nfiq_ok}; r={[f'{v:.2f}' for v in rr]})")


# ---------------------------------------------------------------------------
# A10  robustness (in-memory perturbations on Dataset2 only)
# ---------------------------------------------------------------------------
def graph_robustness():
    import torch
    from PIL import Image, ImageFilter
    from dataset import R307Dataset
    d = gen_dir("graphs", "robustness")
    val_s, test_s = val_and_test_samples("grayscale")
    rng = np.random.RandomState(SEED)
    for m, ch in [("vgg16", "grayscale"), ("efficientformer_l1", "rgb"),
                  ("swin_tiny", "grayscale")]:
        if not os.path.exists(ckpt_path(m, ch, "multiclass")):
            pt(f"  SKIP robustness {m}/{ch} (no checkpoint)")
            continue
        model = load_model(m, ch, num_classes=10)
        size = MODEL_INPUT_SIZE[m]
        mode = "L" if ch == "grayscale" else "RGB"
        base = []
        for s in test_s:
            img = Image.open(s.path).convert(mode).resize(
                (size, size), Image.Resampling.BILINEAR)
            base.append(np.asarray(img, "float32") / 255.0)
        base = np.stack(base)
        labels = np.array([s.scanner_idx for s in test_s])

        def score(xa):
            x = torch.from_numpy(xa)
            if ch == "grayscale":
                x = x[:, None]                      # (N,1,H,W)
            else:
                x = x.transpose(0, 3, 1, 2)         # (N,3,H,W)
            preds = []
            with torch.no_grad():
                for i in range(0, len(x), 32):
                    preds.append(model(x[i:i + 32]).argmax(1).numpy())
            return np.concatenate(preds)

        def acc_of(arr):
            return (score(np.array(arr)) == labels).mean()

        clean = acc_of(base)
        deg = []
        for sg in [0.05, 0.1, 0.2]:
            deg.append((acc_of(np.clip(base + rng.normal(0, sg, base.shape), 0, 1)),
                        f"noise σ={sg}"))
        for r in [1, 2, 4]:
            def blur(im, r=r):
                with Image.fromarray((np.clip(im, 0, 1) * 255).astype("uint8")) as pi:
                    return np.asarray(pi.filter(ImageFilter.GaussianBlur(r)), "float32") / 255.0
            deg.append((acc_of([blur(b) for b in base]), f"blur r={r}"))
        for scl in [0.6, 1.4]:
            deg.append((acc_of(np.clip(base * scl, 0, 1)), f"bright ×{scl}"))
        for scl in [0.6, 1.5]:
            deg.append((acc_of(np.clip(0.5 * (1 - scl) + base * scl, 0, 1)), f"contrast ×{scl}"))
        for q in [85, 50, 20]:
            def jpeg(im, q=q):
                b = io.BytesIO()
                with Image.fromarray((np.clip(im, 0, 1) * 255).astype("uint8")) as pi:
                    pi.save(b, "JPEG", quality=q)
                b.seek(0)
                return np.asarray(Image.open(b).convert("RGB" if ch == "rgb" else "L"),
                                  "float32") / 255.0
            deg.append((acc_of([jpeg(b) for b in base]), f"JPEG q={q}"))
        names = [n for _, n in deg]; accs = [a for a, _ in deg]
        fig, ax = plt.subplots(figsize=(12, 4.6))
        xs = np.arange(len(accs))
        ax.bar(xs, accs, color="#4c72b0", alpha=0.85)
        ax.axhline(clean, color="#c44e52", ls="--", label=f"clean {clean:.4f}")
        ax.set(xticks=xs, xticklabels=names, ylabel="Dataset2 accuracy",
               title=f"Robustness to in-memory perturbations (supplementary) — {m} ({ch})",
               ylim=(0, 1.05))
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout(); fig.savefig(os.path.join(d, f"robustness_{m}_{ch}.png")); plt.close(fig)
        pt(f"  robustness {m}/{ch} clean={clean:.4f} max-drop={clean - min(accs):.4f}")


# ---------------------------------------------------------------------------
# A11  efficiency (all loadable combos): params / size / latency vs official acc
# ---------------------------------------------------------------------------
def measure_latency(model, size, channel, batch, n=15, warmup=3):
    import torch
    in_ch = 1 if channel == "grayscale" else 3
    x = torch.randn(batch, in_ch, size, size)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(ts)), float(np.std(ts))


def graph_efficiency():
    d = gen_dir("graphs", "efficiency")
    official = {}
    with open(os.path.join(REPO, "results", "tables", "results_summary.csv")) as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                official[(r["model"], r["channel"])] = float(r["accuracy"])
    rows = []
    for m in MODEL_REGISTRY:
        for ch in ("grayscale", "rgb"):
            if not os.path.exists(ckpt_path(m, ch, "multiclass")):
                continue
            model = load_model(m, ch, num_classes=10)
            size = MODEL_INPUT_SIZE[m]
            nparams = count_params(model)
            ckbytes = os.path.getsize(ckpt_path(m, ch, "multiclass"))
            l1, _ = measure_latency(model, size, ch, batch=1, n=15)
            l8, _ = measure_latency(model, size, ch, batch=8, n=15)
            rows.append({"m": m, "ch": ch, "params": nparams, "size": ckbytes,
                         "lat1": l1, "lat8_img": l8 / 8.0,
                         "acc": official.get((m, ch), 0.0)})
            pt(f"  eff {m}/{ch} p={nparams/1e6:.1f}M lat1={l1:.0f}ms lat8/img={l8/8:.0f}ms")
    tag = [("params", [x["params"] / 1e6 for x in rows], "Parameters (M)"),
           ("size", [x["size"] / 1e6 for x in rows], "Checkpoint size (MB)"),
           ("lat1", [x["lat1"] for x in rows], "CPU latency batch 1 (ms)")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, vals, tl) in zip(axs, tag):
        ax.scatter(vals, [x["acc"] for x in rows], color="#4c72b0", s=28)
        for v, row in zip(vals, rows):
            if row["m"] in ("vgg16", "efficientformer_l1", "swin_tiny", "efficientnet_b3"):
                ax.annotate(f"{row['m'][:10]}", (v, row["acc"]), fontsize=7)
        ax.set(xlabel=tl, ylabel="Dataset2 accuracy (official)",
               title=f"Accuracy vs {tl}\n({len(rows)} combinations, CPU-only)")
        ax.set_ylim(0.0, 1.05)
    fig.tight_layout(); fig.savefig(os.path.join(d, "efficiency_all.png")); plt.close(fig)
    pt(f"  efficiency graphs for {len(rows)} combos")


# ---------------------------------------------------------------------------
# A12  INT8 quantization of VGG16-g (static int8; honest NOT FEASIBLE fallback)
# ---------------------------------------------------------------------------
def graph_int8_vgg16():
    import torch
    d = gen_dir("graphs", "quantization")
    m, ch = "vgg16", "grayscale"
    val_s, test_s = val_and_test_samples(ch)
    test_true = np.array([s.scanner_idx for s in test_s])
    fp32 = load_model(m, ch, num_classes=10)

    def infer(model, samples, chunk=32):
        from dataset import R307Dataset
        from torch.utils.data import DataLoader
        dl = DataLoader(R307Dataset(samples, MODEL_INPUT_SIZE[m], ch), batch_size=chunk,
                        shuffle=False, num_workers=0)
        probs, preds = [], []
        with torch.no_grad():
            for xb, _ in dl:
                y = torch.softmax(model(xb), dim=1)
                probs.append(y.numpy()); preds.append(y.argmax(1).numpy())
        return np.concatenate(probs), np.concatenate(preds)

    fp_probs, fp_preds = infer(fp32, test_s)
    p_, r_, f1_, _ = precision_recall_fscore_support(test_true, fp_preds,
                                                     labels=list(range(10)),
                                                     zero_division=0, average=None)
    fp_acc = (fp_preds == test_true).mean()
    fp_f1 = float(f1_.mean())
    fp_auc = roc_auc_score(np.eye(10)[test_true], fp_probs, multi_class="ovr")
    buf = io.BytesIO(); torch.save(fp32.state_dict(), buf)
    fp_size = buf.tell()

    feasible = False
    try:
        import torch.ao.quantization as q
        model_q = load_model(m, ch, num_classes=10)
        feats = model_q.backbone.features
        arch = [type(mm).__name__ for mm in feats]
        fusion, i = [], 0
        while i < len(feats):
            if arch[i].startswith("Conv2d"):
                stack = [f"backbone.features.{i}"]
                j = i + 1
                while j < len(feats) and (arch[j].startswith("BatchNorm2d") or arch[j] == "ReLU"):
                    stack.append(f"backbone.features.{j}"); j += 1
                fusion.append(stack); i = j
            else:
                i += 1
        q.fuse_modules(model_q, fusion, inplace=True)
        model_q.eval()
        model_q.qconfig = q.QConfig(
            activation=q.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=q.MinMaxObserver.with_args(dtype=torch.qint8))
        q.prepare(model_q, inplace=True)
        from dataset import R307Dataset
        from torch.utils.data import DataLoader
        calib = DataLoader(R307Dataset(val_s, MODEL_INPUT_SIZE[m], ch), batch_size=32,
                           shuffle=False, num_workers=0)
        with torch.no_grad():
            for xb, _ in calib:
                model_q(xb)
        q.convert(model_q, inplace=True)
        int8 = model_q
        q_probs, q_preds = infer(int8, test_s)
        p_, r_, f1_, _ = precision_recall_fscore_support(test_true, q_preds,
                                                         labels=list(range(10)),
                                                         zero_division=0, average=None)
        q_acc = (q_preds == test_true).mean()
        q_f1 = float(f1_.mean())
        q_auc = roc_auc_score(np.eye(10)[test_true], q_probs, multi_class="ovr")
        bufq = io.BytesIO(); torch.save(int8.state_dict(), bufq)
        q_size = bufq.tell()
        feasible = True
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"

    if feasible:
        l_fp1, _ = measure_latency(fp32, MODEL_INPUT_SIZE[m], ch, 1, n=15)
        l_fp8, _ = measure_latency(fp32, MODEL_INPUT_SIZE[m], ch, 8, n=15)
    else:
        q_acc = q_f1 = q_auc = q_size = 0.0
        l_fp1 = l_fp8 = l_q1 = l_q8 = 0.0

    if feasible:
        l_q1, _ = measure_latency(int8, MODEL_INPUT_SIZE[m], ch, 1, n=15)
        l_q8, _ = measure_latency(int8, MODEL_INPUT_SIZE[m], ch, 8, n=15)
        pairs_q = [(fp_acc, q_acc), (fp_f1, q_f1), (fp_size / 1e6, q_size / 1e6)]
        def bar(vals, title, ylabel):
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(["FP32", "INT8"], vals, color=("#4c72b0", "#c44e52"))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v,
                        f"{v:.4f}" if max(vals) < 2 else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)
            ax.set(title=title, ylabel=ylabel)
            fig.tight_layout(); return fig
        for fname, vals, t, yl in [
                ("int8_accuracy", [fp_acc, q_acc], "VGG16-g Dataset2 accuracy (INT8 supplementary)", "accuracy"),
                ("int8_f1", [fp_f1, q_f1], "VGG16-g Dataset2 macro F1", "F1"),
                ("int8_size", [fp_size / 1e6, q_size / 1e6], "VGG16-g serialized size", "MB")]:
            f = bar(vals, t, yl)
            f.savefig(os.path.join(d, f"{fname}.png")); plt.close(f)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["FP32 b1", "INT8 b1", "FP32 b8", "INT8 b8"],
               [l_fp1, l_q1, l_fp8 / 8, l_q8 / 8], color=("#4c72b0", "#c44e52", "#4c72b0", "#c44e52"))
        for b, v in zip(ax.patches, [l_fp1, l_q1, l_fp8 / 8, l_q8 / 8]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set(title="VGG16-g CPU latency (ms/image)", ylabel="ms/image")
        fig.tight_layout(); fig.savefig(os.path.join(d, "int8_latency.png")); plt.close(fig)
        pt(f"  INT8 static feasible: acc {fp_acc:.4f}->{q_acc:.4f}  F1 {fp_f1:.4f}->{q_f1:.4f}  "
           f"AUC {fp_auc:.4f}->{q_auc:.4f}  lat1 {l_fp1:.1f}->{l_q1:.1f}ms  "
           f"size {fp_size/1e6:.2f}->{q_size/1e6:.2f}MB")
    else:
        pt(f"  INT8 static NOT FEASIBLE: {err}")


# ---------------------------------------------------------------------------
def main():
    pt("=== graph build start ===")
    ensure_multiclass(FINAL_MULTICLASS)
    ensure_binary(BINARY_SETS)
    ensure_multiclass(FEATURE_MODELS)
    pt("--- caches ensured ---")
    graph_confusion_per_scanner()
    graph_roc_pr_multiclass()
    graph_roc_pr_binary()
    graph_threshold_scores()
    graph_det()
    graph_calibration()
    graph_umap()
    graph_user_wise()
    graph_explainability()
    graph_quality()
    graph_robustness()
    graph_efficiency()
    graph_int8_vgg16()
    pt("=== graph build complete ===")


if __name__ == "__main__":
    if "--smoke" in sys.argv[1:]:
        pt("smoke: imports ok")
        sys.exit(0)
    main()
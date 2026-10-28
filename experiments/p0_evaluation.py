"""p0_evaluation.py — P0 core evaluation outputs, computed ONLY from the verified
prediction caches under experiments/predictions/ (no model inference, no writes to
official results/tables or checkpoints).

Run from repo root:
    python experiments/p0_evaluation.py

Outputs -> all NEW subdirectories (never overwrites official graphs/metrics):
    results/graphs/p0/multiclass/   confusion matrices, per-scanner metrics, OvR ROC/PR
    results/graphs/p0/binary/       ROC/PR, threshold curves, score distributions, EER
    results/metrics/p0/multiclass/  CSV data for the multiclass artifacts
    results/metrics/p0/binary/      CSV data for the binary artifacts
    results/metrics/p0_figure_map.csv  index of every artifact produced here

Threshold policy: threshold selection (EER / th*) on validation ONLY. Dataset 2 is
scored at the frozen validation threshold; its curves are reported but never tuned.
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, precision_recall_fscore_support,
                             accuracy_score)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))

import common  # noqa: E402

REPO = common.REPO
FINAL = common.FINAL_MULTICLASS
BINARY = common.BINARY_SETS
LAB = [f"S{i+1}" for i in range(10)]

GMC = os.path.join(REPO, "results", "graphs", "p0", "multiclass")
GBN = os.path.join(REPO, "results", "graphs", "p0", "binary")
MMC = os.path.join(REPO, "results", "metrics", "p0", "multiclass")
MBN = os.path.join(REPO, "results", "metrics", "p0", "binary")
FIGMAP = os.path.join(REPO, "results", "metrics", "p0_figure_map.csv")
for _d in (GMC, GBN, MMC, MBN):
    os.makedirs(_d, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
COL = common.SCANNER_COLORS

_figmap_rows = []
def reg(category, fname, path, description, model, channel, split):
    _figmap_rows.append({"category": category, "filename": fname, "filepath":
                         os.path.relpath(path, REPO), "description": description,
                         "model": model, "channel": channel, "split": split, "dpi": 300})


def read_multiclass(m, ch, split):
    p = common.prediction_csv_path(m, ch, split)
    if not os.path.exists(p):
        return None
    sc, pr, prb = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            sc.append(int(r["scanner"])); pr.append(int(r["pred"]))
            prb.append([float(r[f"p{i}"]) for i in range(10)])
    return np.array(sc), np.array(pr), np.array(prb)


def read_binary(m, ch, split):
    p = common.binary_prediction_csv_path(m, ch, split)
    if not os.path.exists(p):
        return None
    t, l, s = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            t.append(int(r["target_scanner"])); l.append(int(r["label"]))
            s.append(float(r["score"]))
    return np.array(t), np.array(l), np.array(s)


def write_tidy(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def macro_curves(probs, labels, n_classes=10):
    """OvR ROC/PR macro-average (interpolated, sklearn-style)."""
    g = np.linspace(0.0, 1.0, 200)
    tprs, recs, precs, aucs, aps = [], [], [], [], []
    for c in range(n_classes):
        y = (labels == c).astype(int)
        fp, tp, _ = roc_curve(y, probs[:, c])
        aucs.append(auc(fp, tp))
        tprs.append(np.interp(g, fp, tp))
        pre, re, _ = precision_recall_curve(y, probs[:, c])
        aps.append(average_precision_score(y, probs[:, c]))
        re = re[::-1]; pre = pre[::-1]              # ascending recall
        recs.append(np.interp(g, re, pre))
        precs.append(np.interp(g, re, pre))
    return (g, np.mean(tprs, 0), np.mean(aucs), np.mean(recs), np.mean(precs),
            np.mean(aps), aucs, aps)


# ---------------------------------------------------------------------------
# MULTICLASS: confusion matrices, per-scanner metrics, OvR ROC/PR
# ---------------------------------------------------------------------------
def multiclass_block(m, ch):
    for split in ("val", "test"):
        d = read_multiclass(m, ch, split)
        if d is None:
            print(f"  SKIP multiclass {m}/{ch}/{split}: cache missing"); continue
        sc, pr, prb = d
        cm = np.zeros((10, 10), dtype=int)
        for a, b in zip(sc, pr):
            cm[a, b] += 1
        cmn = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)

        # confusion matrix figure (raw + normalized)
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
        for ax, mat, t, fmt in [(axs[0], cm, "raw counts", "d"),
                                (axs[1], cmn, "normalized (row) / recall", ".2f")]:
            im = ax.imshow(mat, cmap="viridis", vmax=None if fmt == "d" else 1)
            ax.set_xticks(range(10), LAB, fontsize=8); ax.set_yticks(range(10), LAB, fontsize=8)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"{split} — {t}")
            for rr in range(10):
                for cc in range(10):
                    v = mat[rr, cc]
                    if (fmt == "d" and v >= 1) or (fmt == ".2f" and v >= 0.005):
                        ax.text(cc, rr, format(v, fmt), ha="center", va="center",
                                fontsize=7, color="white")
        acc = (pr == sc).mean()
        fig.suptitle(f"Confusion matrices — {m} ({ch})  acc={acc:.4f}", fontsize=12)
        fig.tight_layout()
        fp = os.path.join(GMC, f"cm_{m}_{ch}_{split}.png"); fig.savefig(fp); plt.close(fig)
        reg("multiclass-confusion", os.path.basename(fp), fp, "raw + normalized confusion matrices",
            m, ch, split)
        write_tidy(os.path.join(MMC, f"cm_raw_{m}_{ch}_{split}.csv"),
                   ["class"] + LAB,
                   [[f"S{i+1}"] + [int(v) for v in cm[i]] for i in range(10)])
        reg("multiclass-confusion-data", f"cm_raw_{m}_{ch}_{split}.csv",
            os.path.join(MMC, f"cm_raw_{m}_{ch}_{split}.csv"), "raw confusion matrix CSV", m, ch, split)
        write_tidy(os.path.join(MMC, f"cm_norm_{m}_{ch}_{split}.csv"),
                   ["class"] + LAB,
                   [[f"S{i+1}"] + [round(float(v), 6) for v in cmn[i]] for i in range(10)])

        # per-scanner metrics
        prec, rec, f1, sup = precision_recall_fscore_support(
            sc, pr, labels=range(10), zero_division=0)
        accs = [float(accuracy_score(sc[sc == i], pr[sc == i])) if (sc == i).any() else 0.0
                for i in range(10)]
        rows = [[f"S{i+1}", f"{accs[i]:.6f}", f"{prec[i]:.6f}", f"{rec[i]:.6f}",
                 f"{f1[i]:.6f}", int(sup[i])] for i in range(10)]
        rows.append(["macro", "", f"{prec.mean():.6f}", f"{rec.mean():.6f}",
                     f"{f1.mean():.6f}", int(sup.sum())])
        rows.append(["overall-acc", f"{acc:.6f}", "", "", "", int(len(sc))])
        write_tidy(os.path.join(MMC, f"per_scanner_{m}_{ch}_{split}.csv"),
                   ["scanner", "accuracy", "precision", "recall", "f1", "support"], rows)
        reg("multiclass-per-scanner-data", f"per_scanner_{m}_{ch}_{split}.csv",
            os.path.join(MMC, f"per_scanner_{m}_{ch}_{split}.csv"),
            "per-scanner accuracy/precision/recall/F1/support + macro/overall", m, ch, split)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
        x = np.arange(10); w = 0.2
        for ax in axs:
            for vals, lbl, col, off in [(prec, "precision", "#4c72b0", -1.5 * w),
                                        (rec, "recall", "#dd8452", -0.5 * w),
                                        (f1, "F1", "#55a868", 0.5 * w),
                                        (np.array(accs), "accuracy(=recall)", "#c44e52", 1.5 * w)]:
                ax.bar(x + off, vals, w, label=lbl, color=col, alpha=0.9)
            ax.set_xticks(x, LAB, fontsize=8); ax.set_ylim(0, 1.05)
        axs[0].set_title("validation"); axs[1].set_title("Dataset2 test")
        axs[1].legend(fontsize=7)
        fig.suptitle(f"Per-scanner metrics — {m} ({ch})", fontsize=12)
        fig.tight_layout()
        fp = os.path.join(GMC, f"per_scanner_{m}_{ch}_{split}.png"); fig.savefig(fp); plt.close(fig)
        reg("multiclass-per-scanner", os.path.basename(fp), fp, "per-scanner bar metrics", m, ch, split)

        # OvR ROC / PR
        g, m_tpr, m_auc, m_rec, m_pre, m_ap, aucs, aps = macro_curves(prb, sc)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
        for c in range(10):
            y = (sc == c).astype(int)
            fp_, tp_, _ = roc_curve(y, prb[:, c])
            ax1.plot(fp_, tp_, color=COL[c], lw=1.0, alpha=0.7,
                     label=f"{LAB[c]} (AUC {auc(fp_, tp_):.2f})")
            pre, re, _ = precision_recall_curve(y, prb[:, c])
            ax2.plot(re, pre, color=COL[c], lw=1.0, alpha=0.7,
                     label=f"{LAB[c]} (AP {average_precision_score(y, prb[:, c]):.2f})")
        ax1.plot(g, m_tpr, color="k", lw=2.2, ls="--", label=f"macro (AUC {m_auc:.4f})")
        ax2.plot(m_rec, m_pre, color="k", lw=2.2, ls="--", label=f"macro (AP {m_ap:.4f})")
        ax1.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax1.set(xlabel="FPR", ylabel="TPR",
                title=f"ROC one-vs-rest {split}  macro-AUC={m_auc:.4f}")
        ax2.set(xlabel="Recall", ylabel="Precision",
                title=f"PR one-vs-rest {split}  macro-AP={m_ap:.4f}")
        ax1.legend(fontsize=6, loc="lower right"); ax2.legend(fontsize=6, loc="upper right")
        fig.suptitle(f"Multiclass OvR ROC / PR — {m} ({ch})", fontsize=12)
        fig.tight_layout()
        fp = os.path.join(GMC, f"roc_pr_{m}_{ch}_{split}.png"); fig.savefig(fp); plt.close(fig)
        reg("multiclass-roc-pr", os.path.basename(fp), fp, "per-class + macro OvR ROC/PR with AUC/AP",
            m, ch, split)
        rocrows = []
        for c in range(10):
            y = (sc == c).astype(int)
            f_, t_, th_ = roc_curve(y, prb[:, c])
            for fi, ti, thi in zip(f_, t_, th_):
                rocrows.append([f"S{c+1}", f"{fi:.6f}", f"{ti:.6f}", f"{thi:.6f}"])
            rocrows.append([f"S{c+1}", f"{f_[-1]:.6f}", f"{t_[-1]:.6f}", ""])
        write_tidy(os.path.join(MMC, f"roc_ovr_{m}_{ch}_{split}.csv"),
                   ["class", "fpr", "tpr", "threshold"], rocrows)
        prrows = []
        for c in range(10):
            y = (sc == c).astype(int)
            p_, r_, th_ = precision_recall_curve(y, prb[:, c])
            for pi, ri, thi in zip(p_, r_, th_):
                prrows.append([f"S{c+1}", f"{pi:.6f}", f"{ri:.6f}", f"{thi:.6f}"])
            prrows.append([f"S{c+1}", f"{p_[-1]:.6f}", f"{r_[-1]:.6f}", ""])
        write_tidy(os.path.join(MMC, f"pr_ovr_{m}_{ch}_{split}.csv"),
                   ["class", "precision", "recall", "threshold"], prrows)
        print(f"  multiclass {m}/{ch}/{split}: acc={acc:.4f} macro-AUC={m_auc:.4f} "
              f"macro-AP={m_ap:.4f} (n={len(sc)})")


# ---------------------------------------------------------------------------
# BINARY: ROC/PR, threshold curves, EER (val-selected), score distributions
# ---------------------------------------------------------------------------
TH = np.linspace(0.0, 1.0, 401)


def pooled_rates(lab, sc, th):
    n0 = max((lab == 0).sum(), 1); n1 = max((lab == 1).sum(), 1)
    far, frr, acc = np.zeros_like(th), np.zeros_like(th), np.zeros_like(th)
    for i, t in enumerate(th):
        pred = (sc >= t).astype(int)
        acc[i] = accuracy_score(lab, pred)
        far[i] = ((pred == 1) & (lab == 0)).sum() / n0
        frr[i] = ((pred == 0) & (lab == 1)).sum() / n1
    return far, frr, acc


def binary_block(m, ch):
    # score distributions + val-selected threshold first
    tgt_v, lab_v, sc_v = read_binary(m, ch, "val")
    tgt_t, lab_t, sc_t = read_binary(m, ch, "test")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    dist_rows = []
    for ax, (tgt, lab, s_, split) in zip(
            axs, [(tgt_v, lab_v, sc_v, "val"), (tgt_t, lab_t, sc_t, "test")]):
        gen = s_[lab == 1]; imp = s_[lab == 0]
        ax.hist(imp, bins=60, density=True, alpha=0.65, color="#4c72b0", label="impostor")
        ax.hist(gen, bins=60, density=True, alpha=0.65, color="#dd8452", label="genuine")
        ax.set(title=f"{split} — {m} ({ch})\nnG={len(gen)} nI={len(imp)}",
               xlabel="score (sigmoid)", ylabel="density")
        ax.legend(fontsize=8)
        dist_rows.append([split, int(len(gen)), int(len(imp)),
                          f"{gen.mean():.6f}", f"{gen.std():.6f}",
                          f"{imp.mean():.6f}", f"{imp.std():.6f}"])
    fig.suptitle("Genuine vs impostor one-vs-all scores (pooled over 10 scanners)")
    fig.tight_layout()
    fp = os.path.join(GBN, f"scores_{m}_{ch}.png"); fig.savefig(fp); plt.close(fig)
    reg("binary-score-distribution", os.path.basename(fp), fp, "genuine vs impostor score histograms",
        m, ch, "val+test")
    write_tidy(os.path.join(MBN, f"score_dist_{m}_{ch}.csv"),
               ["split", "n_genuine", "n_impostor", "genuine_mean", "genuine_std",
                "impostor_mean", "impostor_std"], dist_rows)
    reg("binary-score-distribution-data", f"score_dist_{m}_{ch}.csv",
        os.path.join(MBN, f"score_dist_{m}_{ch}.csv"),
        "genuine/impostor counts + moments (pooled)", m, ch, "val+test")

    for split, (tgt, lab, sc_) in [("val", (tgt_v, lab_v, sc_v)), ("test", (tgt_t, lab_t, sc_t))]:
        # pooled + per-scanner ROC / PR
        fpr, tpr, _ = roc_curve(lab, sc_)
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(lab, sc_)
        ap = average_precision_score(lab, sc_)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
        per_roc, per_pr = [], []
        for s in range(10):
            mask = tgt == (s + 1)
            y = lab[mask]; x = sc_[mask]
            fp_, tp_, _ = roc_curve(y, x)
            ax1.plot(fp_, tp_, color=COL[s], lw=1.0, alpha=0.7,
                     label=f"{LAB[s]} (AUC {auc(fp_, tp_):.2f})")
            for fi, ti in zip(fp_, tp_):
                per_roc.append([f"S{s+1}", f"{fi:.6f}", f"{ti:.6f}"])
            p_, r_, _ = precision_recall_curve(y, x)
            ax2.plot(r_, p_, color=COL[s], lw=1.0, alpha=0.7,
                     label=f"{LAB[s]} (AP {average_precision_score(y, x):.2f})")
            for pi, ri in zip(p_, r_):
                per_pr.append([f"S{s+1}", f"{pi:.6f}", f"{ri:.6f}"])
        ax1.plot(fpr, tpr, color="k", lw=2.4, ls="--", label=f"pooled (AUC {roc_auc:.4f})")
        ax2.plot(rec, pre, color="k", lw=2.4, ls="--", label=f"pooled (AP {ap:.4f})")
        ax1.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax1.set(xlabel="FPR", ylabel="TPR", title=f"ROC {split}  pooled-AUC={roc_auc:.4f}")
        ax2.set(xlabel="Recall", ylabel="Precision", title=f"PR {split}  pooled-AP={ap:.4f}")
        ax1.legend(fontsize=6, loc="lower right"); ax2.legend(fontsize=6, loc="upper right")
        fig.suptitle(f"Binary one-vs-all ROC / PR — {m} ({ch})", fontsize=12)
        fig.tight_layout()
        fp = os.path.join(GBN, f"roc_pr_{m}_{ch}_{split}.png"); fig.savefig(fp); plt.close(fig)
        reg("binary-roc-pr", os.path.basename(fp), fp, "pooled + per-scanner ROC/PR with AUC/AP",
            m, ch, split)
        write_tidy(os.path.join(MBN, f"roc_pooled_{m}_{ch}_{split}.csv"),
                   ["class", "fpr", "tpr"],
                   [["pooled"] + ["%.6f" % v for v in fpr] + ["%.6f" % v for v in tpr]])
        write_tidy(os.path.join(MBN, f"roc_per_scanner_{m}_{ch}_{split}.csv"),
                   ["class", "fpr", "tpr"], per_roc)
        write_tidy(os.path.join(MBN, f"pr_per_scanner_{m}_{ch}_{split}.csv"),
                   ["class", "precision", "recall"], per_pr)

        # FAR / FRR / ACC vs threshold
        far, frr, acc = pooled_rates(lab, sc_, TH)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, vals, yl in [(axs[0], far, "FAR"), (axs[1], frr, "FRR"),
                             (axs[2], acc, "accuracy")]:
            ax.plot(TH, vals, color="#4c72b0", lw=1.8)
            ax.set(xlabel="threshold (sigmoid score)", ylabel=yl, ylim=(0, 1.05))
            ax.grid(alpha=0.3)
        if split == "val":
            i_eer = int(np.argmin(np.abs(far - frr)))
            eer = (far[i_eer] + frr[i_eer]) / 2
            axs[0].axvline(TH[i_eer], color="k", ls=":", lw=1.2)
            axs[1].axvline(TH[i_eer], color="k", ls=":", lw=1.2)
            axs[0].set_title(f"FAR vs threshold — VAL (EER≈{eer:.4f} @ {TH[i_eer]:.3f})")
            axs[1].set_title(f"FRR vs threshold — VAL (select th*={TH[i_eer]:.3f})")
        else:
            axs[0].set_title("FAR vs threshold — test (no threshold selection)")
            axs[1].set_title("FRR vs threshold — test")
        axs[2].set_title(f"Accuracy vs threshold — {split}")
        fig.suptitle(f"Threshold analysis — {m} ({ch})", fontsize=12)
        fig.tight_layout()
        fp = os.path.join(GBN, f"far_frr_acc_{m}_{ch}_{split}.png"); fig.savefig(fp); plt.close(fig)
        reg("binary-threshold-curves", os.path.basename(fp), fp,
            "FAR / FRR / accuracy vs threshold (pooled)", m, ch, split)
        write_tidy(os.path.join(MBN, f"threshold_{m}_{ch}_{split}.csv"),
                   ["threshold", "far", "frr", "accuracy"],
                   [["%.4f" % t, "%.6f" % a, "%.6f" % b, "%.6f" % c]
                    for t, a, b, c in zip(TH, far, frr, acc)])
        reg("binary-threshold-data", f"threshold_{m}_{ch}_{split}.csv",
            os.path.join(MBN, f"threshold_{m}_{ch}_{split}.csv"),
            "pooled FAR/FRR/accuracy over threshold sweep", m, ch, split)

    # EER summary: threshold selected on validation only
    far_v, frr_v, acc_v = pooled_rates(lab_v, sc_v, TH)
    i_eer = int(np.argmin(np.abs(far_v - frr_v)))
    eer_v = (far_v[i_eer] + frr_v[i_eer]) / 2
    th_star = TH[i_eer]
    mask_v = np.arange(len(TH)) == i_eer
    print_mask = (np.ones_like(TH, dtype=bool) & mask_v)
    pred_t = (sc_t >= th_star).astype(int)
    acc_t = float(accuracy_score(lab_t, pred_t))
    far_t = float(((pred_t == 1) & (lab_t == 0)).sum() / max((lab_t == 0).sum(), 1))
    frr_t = float(((pred_t == 0) & (lab_t == 1)).sum() / max((lab_t == 1).sum(), 1))
    # per-scanner EER on val (for spread reporting)
    per_eer = []
    for s in range(10):
        x = sc_v[tgt_v == (s + 1)]; y = lab_v[tgt_v == (s + 1)]
        f_, t_, _ = roc_curve(y, x)
        eer_p = 1.0
        interp = np.interp(np.linspace(0, 1, 500), f_, t_)
        diff = np.abs(np.linspace(0, 1, 500)[::-1] - interp)
        per_eer.append(float(np.linspace(0, 1, 500)[np.argmin(diff)]))
    write_tidy(os.path.join(MBN, f"eer_{m}_{ch}.csv"),
               ["metric", "value"],
               [["val_pooled_EER", f"{eer_v:.6f}"],
                ["val_th_star", f"{th_star:.6f}"],
                ["test_acc_at_val_th_star", f"{acc_t:.6f}"],
                ["test_far_at_val_th_star", f"{far_t:.6f}"],
                ["test_frr_at_val_th_star", f"{frr_t:.6f}"],
                ["val_mean_per_scanner_EER", f"{np.mean(per_eer):.6f}"],
                ["val_per_scanner_EER", ",".join(f"{v:.4f}" for v in per_eer)]])
    reg("binary-eer", f"eer_{m}_{ch}.csv", os.path.join(MBN, f"eer_{m}_{ch}.csv"),
        "EER (validation) + Dataset2 metrics at the val-frozen threshold", m, ch, "val+test")
    print(f"  binary {m}/{ch}: val pooled-EER={eer_v:.4f} @ th={th_star:.4f} | "
          f"test@th* acc={acc_t:.4f} far={far_t:.4f} frr={frr_t:.4f} | "
          f"mean per-scanner EER(val)={np.mean(per_eer):.4f}")


def write_figmap():
    with open(FIGMAP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "filename", "filepath",
                                          "description", "model", "channel",
                                          "split", "dpi"])
        w.writeheader(); w.writerows(_figmap_rows)


def main():
    print("=== P0 evaluation (caches only) ===")
    for m, ch in FINAL:
        print(f"--- multiclass {m} ({ch}) ---")
        multiclass_block(m, ch)
    for m, ch in BINARY:
        print(f"--- binary {m} ({ch}) ---")
        binary_block(m, ch)
    write_figmap()
    print(f"figure map: {FIGMAP}")
    print(f"artifacts: {len(_figmap_rows)}")
    print("=== done ===")


if __name__ == "__main__":
    main()
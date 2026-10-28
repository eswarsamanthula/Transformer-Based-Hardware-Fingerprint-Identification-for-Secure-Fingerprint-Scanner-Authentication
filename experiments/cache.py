"""cache.py — generate + verify the prediction caches from official checkpoints.

Usage (run from repo root):
    python experiments/cache.py --multiclass            # all loadable model/channel combos
    python experiments/cache.py --multiclass --final    # only FINAL_MULTICLASS (detailed eval)
    python experiments/cache.py --multiclass --only vgg16
    python experiments/cache.py --binary                # BINARY_SETS, incremental
    python experiments/cache.py --verify                # multiclass + binary vs official tables

Every cache is a plain CSV under experiments/predictions/. A file that already
exists is skipped, so re-runs are incremental.

--verify compares cached val/test multiclass accuracy against the official
results/tables/results_summary.csv and recomputes per-scanner binary accuracy /
FAR / FRR at threshold 0.5 from the cached scores against
results/tables/results_binary_summary.csv — the harness sanity gate.
"""
import csv
import os
import sys
import time
import math

import numpy as np

import common
from common import (REPO, PRED_DIR, prediction_csv_path, binary_prediction_csv_path,
                    load_model, predict_probs, predict_binary_scores,
                    val_and_test_samples, user_id, ensure, gen_dir)
from config import MODEL_REGISTRY, MODEL_INPUT_SIZE, NUM_SCANNERS


def split_samples(channel):
    val, test = val_and_test_samples(channel)
    return ("val", val), ("test", test)


def generate_multiclass(only: str = "", final: bool = False):
    done, skipped = 0, 0
    t0 = time.time()
    combos = []
    want = set(only.split(",")) if only else None
    print(f"generate_multiclass: registry={len(MODEL_REGISTRY)} final={final} "
          f"want={want} argv={sys.argv[1:]}")
    combos = []
    for m in list(MODEL_REGISTRY):
        for ch in common.CHANNELS:
            if final and (m, ch) not in common.FINAL_MULTICLASS:
                continue
            if want is not None and m not in want:
                continue
            if not os.path.exists(common.ckpt_path(m, ch, "multiclass")):
                skipped += 1
                print(f"SKIP {m}/{ch}: no multiclass checkpoint")
                continue
            combos.append((m, ch))
    print(f"  {len(combos)} combos to check")
    for m, ch in combos:
        for split, samples in split_samples(ch):
            out = prediction_csv_path(m, ch, split)
            if os.path.exists(out):
                continue
            model = load_model(m, ch, num_classes=10)
            probs, preds = predict_probs(model, samples, m, ch)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filepath", "split", "model", "channel", "scanner",
                            "user", "pred", "conf"] + [f"p{i}" for i in range(NUM_SCANNERS)])
                for s, p, pr in zip(samples, preds, probs):
                    w.writerow([s.path, split, m, ch, s.scanner_idx,
                                user_id(s.path), p, f"{pr.max():.6f}"]
                               + [f"{v:.6f}" for v in pr])
            done += 1
            acc = float((preds == np.array([s.scanner_idx for s in samples])).mean())
            print(f"  {m:16s} {ch:10s} {split:4s} n={len(samples):4d} acc={acc:.4f} "
                  f"({time.time()-t0:.0f}s)")
    print(f"multiclass cache: {done} generated, {skipped} skipped-checkpoints, elapsed {time.time()-t0:.0f}s")


def generate_binary(only: str = ""):
    done = 0
    t0 = time.time()
    for m, ch in common.BINARY_SETS:
        if only and only not in m:
            continue
        model_dir_defaults = {"mobilenet": "mobilenet", "vgg16": "vgg16",
                              "efficientformer_l1": "efficientformer_l1"}
        for split, samples in split_samples(ch):
            out = binary_prediction_csv_path(m, ch, split)
            if os.path.exists(out):
                continue
            rows = []
            for s in range(1, NUM_SCANNERS + 1):
                model = load_model(m, ch, num_classes=1, scanner=s)
                sc = predict_binary_scores(model, samples, m, ch)
                labels = np.array([1.0 if x.scanner_idx == s - 1 else 0.0 for x in samples])
                rows.append((sc, labels, np.full(len(sc), s)))
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filepath", "split", "target_scanner", "label", "score"])
                for j, smp in enumerate(samples):
                    for sc_arr, lab_arr, tar_arr in rows:
                        w.writerow([smp.path, split, int(tar_arr[j]),
                                    int(lab_arr[j]), f"{sc_arr[j]:.6f}"])
            done += 1
            total = sum(int(r[1].sum()) for r in rows)
            print(f"  {m:16s} {ch:10s} {split:4s} n={len(samples):4d} "
                  f"pos={total} ({time.time()-t0:.0f}s)")
    print(f"binary cache: {done} files, elapsed {time.time()-t0:.0f}s")


def verify_official():
    """Compare cache accuracies vs results/tables/results_summary.csv (official)."""
    official = {}
    path = os.path.join(REPO, "results", "tables", "results_summary.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["channel"], row["split"])
            official[key] = float(row["accuracy"])
    mismatch, ok, missing = 0, 0, 0
    for m in MODEL_REGISTRY:
        for ch in common.CHANNELS:
            if not os.path.exists(common.ckpt_path(m, ch, "multiclass")):
                if (m, ch, "test") in official:
                    missing += 1
                    print(f"  NOTE {m}/{ch}: official rows exist but checkpoint absent (not verifiable)")
                continue
            for split in ("val", "test"):
                p = prediction_csv_path(m, ch, split)
                if not os.path.exists(p):
                    continue
                acc_c = 0.0
                n = 0
                with open(p) as f:
                    for row in csv.DictReader(f):
                        acc_c += 1.0 if int(row["pred"]) == int(row["scanner"]) else 0.0
                        n += 1
                acc_c /= max(n, 1)
                if (m, ch, split) in official:
                    acc_o = official[(m, ch, split)]
                    d = abs(acc_c - acc_o)
                    if d >= 1e-4:
                        mismatch += 1
                        print(f"  MISMATCH {m}/{ch}/{split}: cached={acc_c:.4f} official={acc_o:.4f}")
                    else:
                        ok += 1
                else:
                    print(f"  no official row for {m}/{ch}/{split}")
    print(f"verify: {ok} match, {mismatch} mismatch, {missing} unverifiable (missing checkpoint)")


def verify_binary():
    """Recompute per-scanner acc/FAR/FRR at threshold 0.5 from cached scores and
    compare with results/tables/results_binary_summary.csv (official; its rows are
    the threshold-0.5 numbers, src/test.py behavior)."""
    official = {}
    path = os.path.join(REPO, "results", "tables", "results_binary_summary.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["scanner"] == "mean":
                continue
            key = (row["model"], row["channel"], int(row["scanner"]), row["split"])
            official[key] = (float(row["accuracy"]), float(row["far"]), float(row["frr"]))
    mismatch, ok, missing = 0, 0, 0
    for m, ch in common.BINARY_SETS:
        for split in ("val", "test"):
            p = binary_prediction_csv_path(m, ch, split)
            if not os.path.exists(p):
                missing += 1
                print(f"  NOTE {m}/{ch}/{split}: no cache file")
                continue
            per = {s: ([], []) for s in range(1, NUM_SCANNERS + 1)}
            with open(p) as f:
                for row in csv.DictReader(f):
                    t = int(row["target_scanner"])
                    per[t][0].append(float(row["score"]))
                    per[t][1].append(int(row["label"]))
            for s in range(1, NUM_SCANNERS + 1):
                sc = np.array(per[s][0])
                lab = np.array(per[s][1])
                pred = (sc >= 0.5).astype(int)
                acc = float((pred == lab).mean())
                far = float(((lab == 0) & (pred == 1)).sum() / max((lab == 0).sum(), 1))
                frr = float(((lab == 1) & (pred == 0)).sum() / max((lab == 1).sum(), 1))
                acc_o, far_o, frr_o = official[(m, ch, s, split)]
                if (abs(acc - acc_o) >= 1e-4 or abs(far - far_o) >= 1e-4
                        or abs(frr - frr_o) >= 1e-4):
                    mismatch += 1
                    print(f"  MISMATCH {m}/{ch}/scanner{s}/{split}: cached acc={acc:.4f} "
                          f"far={far:.4f} frr={frr:.4f} vs official acc={acc_o:.4f} "
                          f"far={far_o:.4f} frr={frr_o:.4f}")
                else:
                    ok += 1
    print(f"verify_binary: {ok} scanner/split rows match, {mismatch} mismatch, "
          f"{missing} missing cache files")


def main():
    args = sys.argv[1:]
    if "--only" in args:
        only = args[args.index("--only") + 1]
    else:
        only = ""
    final = "--final" in args
    if "--multiclass" in args:
        generate_multiclass(only, final)
    if "--binary" in args:
        generate_binary(only)
    if "--verify" in args:
        verify_official()
        verify_binary()


if __name__ == "__main__":
    main()
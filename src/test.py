"""
test.py — evaluates trained checkpoints.

  multiclass: accuracy/precision/recall/F1/AUC on val AND the unseen-user
      test set (Dataset 2), reported separately.
  binary:     one-vs-all, one checkpoint per scanner 1..10. Reports
      accuracy/precision/recall/F1/AUC plus FAR and FRR at threshold 0.5
      on the sigmoid output. Per-scanner lines, then a mean-across-scanners
      summary, then results_binary_summary.csv.

Usage:
    python test.py --model vit_b16 --channel grayscale
    python test.py --model all --channel both --mode binary
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from config import (
    MODEL_REGISTRY, MODEL_INPUT_SIZE, CHANNEL_MODES, CHECKPOINT_DIR,
    NUM_SCANNERS, RESULTS_DIR, model_category,
)
from dataset import get_splits, get_binary_splits
from models import build_model


def evaluate(model, loader, device, num_classes, binary: bool):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            if binary:
                probs = torch.sigmoid(outputs.view(-1)).cpu().numpy()
                labels_np = labels.cpu().numpy()
                preds = (probs >= 0.5).astype(int)
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                labels_np = labels.numpy()
                preds = probs.argmax(axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        if binary:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")  # can happen if a class is missing from a small holdout set

    metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

    if binary:
        # threshold 0.5 on sigmoid output
        pos = all_preds == 1
        neg = all_preds == 0
        tp = int((all_labels[all_preds == 1] == 1).sum()) if pos.any() else 0
        fp = int((all_labels[all_preds == 1] == 0).sum()) if pos.any() else 0
        fn = int((all_labels[all_preds == 0] == 1).sum()) if neg.any() else 0
        tn = int((all_labels[all_preds == 0] == 0).sum()) if neg.any() else 0
        metrics["far"] = fp / (fp + tn) if (fp + tn) else float("nan")
        metrics["frr"] = fn / (fn + tp) if (fn + tp) else float("nan")

    return metrics


def test_one_multiclass(model_name: str, channel_mode: str, device: str):
    in_channels = 1 if channel_mode == "grayscale" else 3
    input_size = MODEL_INPUT_SIZE[model_name]
    batch_size = MODEL_REGISTRY[model_name]["batch_size"]

    ckpt_path = os.path.join(CHECKPOINT_DIR, model_category(model_name), f"{model_name}_{channel_mode}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"SKIP {model_name}/{channel_mode}: no checkpoint at {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name, num_classes=NUM_SCANNERS, in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, val_ds, test_ds = get_splits(input_size, channel_mode)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    val_metrics = evaluate(model, val_loader, device, NUM_SCANNERS, binary=False)
    test_metrics = evaluate(model, test_loader, device, NUM_SCANNERS, binary=False)

    print(f"\n{model_name} / {channel_mode}")
    print(f"  VAL  acc={val_metrics['accuracy']:.4f} prec={val_metrics['precision']:.4f} "
          f"rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}")
    print(f"  TEST (unseen user, Dataset 2)  acc={test_metrics['accuracy']:.4f} prec={test_metrics['precision']:.4f} "
          f"rec={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}")

    return {"model": model_name, "channel": channel_mode, "val": val_metrics, "test": test_metrics}


def test_one_binary(model_name: str, channel_mode: str, device: str):
    """Evaluate all 10 one-vs-all checkpoints; returns rows for the CSV."""
    in_channels = 1 if channel_mode == "grayscale" else 3
    input_size = MODEL_INPUT_SIZE[model_name]
    batch_size = MODEL_REGISTRY[model_name]["batch_size"]

    rows = []
    per_scanner = []
    for target_scanner in range(NUM_SCANNERS):
        scanner_n = target_scanner + 1
        ckpt_path = os.path.join(
            CHECKPOINT_DIR, model_category(model_name),
            f"{model_name}_{channel_mode}_binary_scanner{scanner_n}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP {model_name}/{channel_mode} scanner{scanner_n}: no checkpoint at {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(model_name, num_classes=1, in_channels=in_channels).to(device)
        model.load_state_dict(ckpt["state_dict"])

        _, val_ds_b, test_ds_b = get_binary_splits(input_size, channel_mode, target_scanner)
        val_loader = DataLoader(val_ds_b, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds_b, batch_size=batch_size, shuffle=False, num_workers=2)

        for split_name, loader in (("val", val_loader), ("test", test_loader)):
            m = evaluate(model, loader, device, 1, binary=True)
            rows.append({
                "model": model_name, "channel": channel_mode,
                "scanner": scanner_n, "split": split_name, **m})
            if split_name == "test":
                per_scanner.append(m)

        print(f"  Scanner{scanner_n}: acc={per_scanner[-1]['accuracy']:.4f} "
              f"prec={per_scanner[-1]['precision']:.4f} rec={per_scanner[-1]['recall']:.4f} "
              f"f1={per_scanner[-1]['f1']:.4f} auc={per_scanner[-1]['auc']:.4f} "
              f"far={per_scanner[-1]['far']:.4f} frr={per_scanner[-1]['frr']:.4f}")

    if per_scanner:
        keys = ["accuracy", "precision", "recall", "f1", "auc", "far", "frr"]
        means = {k: float(np.nanmean([m[k] for m in per_scanner])) for k in keys}
        print(f"  MEAN (test, 10 scanners): " + " ".join(f"{k}={means[k]:.4f}" for k in keys))
        rows.append({
            "model": model_name, "channel": channel_mode,
            "scanner": "mean", "split": "test", **means})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help=f"one of {list(MODEL_REGISTRY)} or 'all'")
    parser.add_argument("--channel", required=True, choices=["grayscale", "rgb", "both"])
    parser.add_argument("--mode", default="multiclass", choices=["multiclass", "binary"])
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_names = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    channels = CHANNEL_MODES if args.channel == "both" else [args.channel]

    out_csv = args.out_csv or os.path.join(
        RESULTS_DIR, "tables",
        "results_binary_summary.csv" if args.mode == "binary" else "results_summary.csv")

    rows = []
    for model_name in model_names:
        for channel_mode in channels:
            if args.mode == "binary":
                rows.extend(test_one_binary(model_name, channel_mode, device) or [])
            else:
                result = test_one_multiclass(model_name, channel_mode, device)
                if result:
                    rows.append(result)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if args.mode == "binary":
        header = ["model", "channel", "scanner", "split", "accuracy", "precision", "recall", "f1", "auc", "far", "frr"]
    else:
        header = ["model", "channel", "split", "accuracy", "precision", "recall", "f1", "auc"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            if args.mode == "binary":
                writer.writerow([r["model"], r["channel"], r["scanner"], r["split"],
                                 r["accuracy"], r["precision"], r["recall"], r["f1"], r["auc"], r["far"], r["frr"]])
            else:
                for split_name in ("val", "test"):
                    m = r[split_name]
                    writer.writerow([r["model"], r["channel"], split_name,
                                     m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc"]])
    print(f"\nSummary written to {out_csv}")


if __name__ == "__main__":
    main()
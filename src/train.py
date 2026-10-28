"""
train.py — trains ONE model on ONE channel mode, exactly per the paper's
training procedure:

    - Adam optimizer (per-model lr from config.MODEL_REGISTRY)
    - Multiclass: categorical cross-entropy (softmax output, 10 classes)
    - Binary (one-vs-all): BCEWithLogitsLoss per target scanner (1 class)
    - Categorical/binary accuracy tracked every epoch
    - Early stopping on validation loss, patience = 10, best weights restored
    - No data augmentation
    - Batch size and epoch cap per-model, from config.MODEL_REGISTRY
      (max_epochs per-model; MAX_EPOCHS only as a fallback)

Usage:
    python train.py --model vit_b16 --channel grayscale [--mode multiclass]
    python train.py --model efficientnet_b3 --channel rgb --mode binary
    python train.py --model all --channel both --mode binary   # all 10 scanners x all models

Every run writes:
    ../checkpoints/<category>/<model>_<channel>_best.pt
    ../logs/training/<model>_<channel>.csv   (epoch, train_loss, train_acc, val_loss, val_acc)
Binary runs write one checkpoint + log PER scanner:
    ../checkpoints/<category>/<model>_<channel>_binary_scanner<n>_best.pt
    ../logs/training/<model>_<channel>_binary_scanner<n>.csv
"""

import argparse
import copy
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    MODEL_REGISTRY, MODEL_INPUT_SIZE, CHANNEL_MODES, LEARNING_RATE,
    EARLY_STOP_PATIENCE, MAX_EPOCHS, CHECKPOINT_DIR, LOG_DIR, NUM_SCANNERS,
    model_category,
)


def already_done(model_name: str, channel_mode: str, mode: str) -> bool:
    """True if this (model, channel, mode) run already has its checkpoint(s)
    on disk. Checkpoints are only written at the END of a run (see
    _train_job), so this correctly treats a mid-training crash/disconnect
    as NOT done — that combo will be retrained from scratch, same as any
    combo that never started."""
    cat = model_category(model_name)
    if mode == "multiclass":
        p = os.path.join(CHECKPOINT_DIR, cat, f"{model_name}_{channel_mode}_best.pt")
        return os.path.exists(p)
    return all(
        os.path.exists(os.path.join(
            CHECKPOINT_DIR, cat,
            f"{model_name}_{channel_mode}_binary_scanner{n}_best.pt"))
        for n in range(1, NUM_SCANNERS + 1)
    )
from dataset import get_splits, get_binary_splits
from models import build_model


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool, binary: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            if binary:
                # outputs: (B, 1); labels: (B,) float 0.0/1.0 — match shapes
                loss = criterion(outputs.view(-1), labels)
                preds = (torch.sigmoid(outputs).squeeze(1) >= 0.5).float()
            else:
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def _train_job(model_name, channel_mode, device, stem, datasets, num_classes,
               criterion, binary: bool):
    """Shared train/val loop. `stem` = '<model>_<channel>[_binary_scanner<n>]'."""
    in_channels = 1 if channel_mode == "grayscale" else 3
    input_size = MODEL_INPUT_SIZE[model_name]
    batch_size = MODEL_REGISTRY[model_name]["batch_size"]
    lr = MODEL_REGISTRY[model_name].get("lr", LEARNING_RATE)
    max_epochs = MODEL_REGISTRY[model_name].get("max_epochs", MAX_EPOCHS)

    train_ds, val_ds, _ = datasets  # test (Dataset 2) reserved for test.py only
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(model_name, num_classes=num_classes, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    ckpt_subdir = os.path.join(CHECKPOINT_DIR, model_category(model_name))
    os.makedirs(ckpt_subdir, exist_ok=True)
    log_subdir = os.path.join(LOG_DIR, "training")
    os.makedirs(log_subdir, exist_ok=True)
    log_path = os.path.join(log_subdir, f"{stem}.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True, binary=binary)
            val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False, binary=binary)

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            f.flush()
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                    break

    model.load_state_dict(best_state)  # restore best weights, per paper
    ckpt_path = os.path.join(ckpt_subdir, f"{stem}_best.pt")
    torch.save({
        "model_name": model_name,
        "channel_mode": channel_mode,
        "mode": "binary" if binary else "multiclass",
        "state_dict": best_state,
        "best_val_loss": best_val_loss,
    }, ckpt_path)
    print(f"Saved best checkpoint -> {ckpt_path} (val_loss={best_val_loss:.4f})")
    return ckpt_path


def train_one(model_name: str, channel_mode: str, device: str, mode: str):
    if mode == "multiclass":
        print(f"\n=== Training {model_name} / {channel_mode} / multiclass ===")
        _train_job(
            model_name, channel_mode, device,
            stem=f"{model_name}_{channel_mode}",
            datasets=get_splits(MODEL_INPUT_SIZE[model_name], channel_mode),
            num_classes=NUM_SCANNERS,
            criterion=nn.CrossEntropyLoss(),
            binary=False,
        )
    else:
        for target_scanner in range(NUM_SCANNERS):
            scanner_n = target_scanner + 1  # 1-based, matches Sensor<n>
            print(f"\n=== Training {model_name} / {channel_mode} / binary vs Sensor{scanner_n} ===")
            _train_job(
                model_name, channel_mode, device,
                stem=f"{model_name}_{channel_mode}_binary_scanner{scanner_n}",
                datasets=get_binary_splits(MODEL_INPUT_SIZE[model_name], channel_mode, target_scanner),
                num_classes=1,
                criterion=nn.BCEWithLogitsLoss(),
                binary=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         help=f"one of {list(MODEL_REGISTRY)} or 'all'")
    parser.add_argument("--channel", required=True, choices=["grayscale", "rgb", "both"])
    parser.add_argument("--mode", default="multiclass", choices=["multiclass", "binary"])
    parser.add_argument("--resume", action="store_true",
                         help="skip any (model, channel) combo that already has a "
                              "saved best checkpoint; used to continue an --model all run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_names = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    channels = CHANNEL_MODES if args.channel == "both" else [args.channel]

    for model_name in model_names:
        for channel_mode in channels:
            if args.resume and already_done(model_name, channel_mode, args.mode):
                print(f"SKIP {model_name}/{channel_mode}: checkpoint already exists")
                continue
            train_one(model_name, channel_mode, device, args.mode)


if __name__ == "__main__":
    main()
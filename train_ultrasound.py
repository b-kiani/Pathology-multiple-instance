import os
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

from dataset import UltrasoundDataset


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = Path("/Users/behnamkiani/Downloads/Thyroid")
FOLDS_CSV = DATA_ROOT / "outputs" / "folds" / "strict_multimodal_5folds.csv"

OUTPUT_DIR = DATA_ROOT / "outputs" / "ultrasound_baseline"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FOLD = 0
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
MODEL_NAME = "convnext_tiny"   # can later test efficientnet_b3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# METRICS
# =========================================================
def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return float("nan")


def safe_auprc(y_true, y_prob):
    try:
        return average_precision_score(y_true, y_prob)
    except Exception:
        return float("nan")


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    auc = safe_auc(y_true, y_prob)
    auprc = safe_auprc(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "auroc": auc,
        "auprc": auprc,
        "accuracy": acc,
        "f1": f1,
        "mcc": mcc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold_youden(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(0.0, 1.0, 101)
    best_thr = 0.5
    best_score = -1e9

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        score = sensitivity + specificity - 1.0  # Youden index

        if score > best_score:
            best_score = score
            best_thr = thr

    return float(best_thr)


# =========================================================
# MODEL
# =========================================================
class UltrasoundClassifier(nn.Module):
    def __init__(self, model_name="convnext_tiny", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1
        )

    def forward(self, x):
        logits = self.backbone(x).squeeze(1)
        return logits


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()

        running_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_binary_metrics(all_labels, all_probs, threshold=0.5)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []
    all_patient_ids = []

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        labs = labels.cpu().numpy()

        running_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())
        all_patient_ids.extend(batch["patient_id"])

    epoch_loss = running_loss / len(loader.dataset)

    default_metrics = compute_binary_metrics(all_labels, all_probs, threshold=0.5)
    best_thr = find_best_threshold_youden(all_labels, all_probs)
    youden_metrics = compute_binary_metrics(all_labels, all_probs, threshold=best_thr)

    default_metrics["loss"] = epoch_loss
    default_metrics["best_threshold_youden"] = best_thr

    return default_metrics, youden_metrics, {
        "patient_id": all_patient_ids,
        "y_true": [int(x) for x in all_labels],
        "y_prob": [float(x) for x in all_probs],
    }


# =========================================================
# MAIN
# =========================================================
def main():
    seed_everything(RANDOM_SEED)

    print("=" * 70)
    print("ULTRASOUND-ONLY BASELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Fold: {FOLD}")
    print(f"Model: {MODEL_NAME}")
    print(f"Folds CSV: {FOLDS_CSV}")

    train_ds = UltrasoundDataset(
        folds_csv_path=str(FOLDS_CSV),
        fold=FOLD,
        split="train",
        img_size=IMG_SIZE
    )
    val_ds = UltrasoundDataset(
        folds_csv_path=str(FOLDS_CSV),
        fold=FOLD,
        split="val",
        img_size=IMG_SIZE
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # class weight from training split
    train_df = train_ds.df
    n_neg = int((train_df["label"] == 0).sum())
    n_pos = int((train_df["label"] == 1).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    print(f"Train negatives: {n_neg}")
    print(f"Train positives: {n_pos}")
    print(f"pos_weight: {pos_weight.item():.4f}")

    model = UltrasoundClassifier(model_name=MODEL_NAME, pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auroc = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics, val_metrics_youden, val_preds = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],

            "train_loss": train_metrics["loss"],
            "train_auroc": train_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_mcc": train_metrics["mcc"],
            "train_sensitivity": train_metrics["sensitivity"],
            "train_specificity": train_metrics["specificity"],

            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_mcc": val_metrics["mcc"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_best_threshold_youden": val_metrics["best_threshold_youden"],
            "val_youden_accuracy": val_metrics_youden["accuracy"],
            "val_youden_f1": val_metrics_youden["f1"],
            "val_youden_mcc": val_metrics_youden["mcc"],
            "val_youden_sensitivity": val_metrics_youden["sensitivity"],
            "val_youden_specificity": val_metrics_youden["specificity"],
        }
        history.append(row)

        print(
            f"Train loss={train_metrics['loss']:.4f} | "
            f"AUROC={train_metrics['auroc']:.4f} | "
            f"AUPRC={train_metrics['auprc']:.4f}"
        )
        print(
            f"Val   loss={val_metrics['loss']:.4f} | "
            f"AUROC={val_metrics['auroc']:.4f} | "
            f"AUPRC={val_metrics['auprc']:.4f} | "
            f"ACC={val_metrics['accuracy']:.4f} | "
            f"Sens={val_metrics['sensitivity']:.4f} | "
            f"Spec={val_metrics['specificity']:.4f}"
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]

            ckpt_path = CHECKPOINT_DIR / f"best_fold{FOLD}_{MODEL_NAME}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auroc": best_auroc,
                "config": {
                    "fold": FOLD,
                    "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "model_name": MODEL_NAME,
                }
            }, ckpt_path)

            pred_path = LOG_DIR / f"best_val_predictions_fold{FOLD}_{MODEL_NAME}.csv"
            pd.DataFrame(val_preds).to_csv(pred_path, index=False)

            print(f"Saved new best checkpoint to: {ckpt_path}")

    history_df = pd.DataFrame(history)
    history_csv = LOG_DIR / f"history_fold{FOLD}_{MODEL_NAME}.csv"
    history_df.to_csv(history_csv, index=False)

    summary = {
        "best_val_auroc": float(history_df["val_auroc"].max()),
        "best_val_auprc": float(history_df.loc[history_df["val_auroc"].idxmax(), "val_auprc"]),
        "best_epoch_by_auroc": int(history_df["val_auroc"].idxmax() + 1),
        "fold": FOLD,
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }

    summary_path = LOG_DIR / f"summary_fold{FOLD}_{MODEL_NAME}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved history: {history_csv}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
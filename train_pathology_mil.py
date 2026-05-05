import json
import random
import argparse
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

from dataset import PathologyBagsDataset, pathology_bag_collate_fn


DATA_ROOT = Path("/Users/behnamkiani/Downloads/Thyroid")
FOLDS_CSV = DATA_ROOT / "outputs" / "folds" / "strict_multimodal_5folds.csv"

OUTPUT_DIR = DATA_ROOT / "outputs" / "pathology_mil_ablation"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
MODEL_NAME = "resnet18"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    mcc = matthews_corrcoef(y_true, y_pred)

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


class PatchEncoder(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, feat_dim=256):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        in_dim = self.backbone.num_features

        self.proj = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


class AttentionMIL(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, feat_dim=256, attn_dim=128):
        super().__init__()

        self.encoder = PatchEncoder(
            model_name=model_name,
            pretrained=pretrained,
            feat_dim=feat_dim
        )

        self.attn_v = nn.Linear(feat_dim, attn_dim)
        self.attn_u = nn.Linear(feat_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

    def forward(self, bag, bag_mask):
        B, N, C, H, W = bag.shape

        flat = bag.view(B * N, C, H, W)
        feat = self.encoder(flat)

        D = feat.shape[-1]
        feat = feat.view(B, N, D)

        A_v = torch.tanh(self.attn_v(feat))
        A_u = torch.sigmoid(self.attn_u(feat))
        A = self.attn_w(A_v * A_u).squeeze(-1)

        A = A.masked_fill(~bag_mask, float("-inf"))
        attn = torch.softmax(A, dim=1)

        bag_feat = torch.sum(attn.unsqueeze(-1) * feat, dim=1)
        logits = self.classifier(bag_feat).squeeze(-1)

        return logits, attn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="Train", leave=False):
        bag = batch["bag"].to(device)
        bag_mask = batch["bag_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _ = model(bag, bag_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()

        running_loss += loss.item() * bag.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())

    metrics = compute_binary_metrics(all_labels, all_probs)
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []
    all_patient_ids = []

    for batch in tqdm(loader, desc="Val", leave=False):
        bag = batch["bag"].to(device)
        bag_mask = batch["bag_mask"].to(device)
        labels = batch["label"].to(device)

        logits, _ = model(bag, bag_mask)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        labs = labels.cpu().numpy()

        running_loss += loss.item() * bag.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())
        all_patient_ids.extend(batch["patient_id"])

    metrics = compute_binary_metrics(all_labels, all_probs)
    metrics["loss"] = running_loss / len(loader.dataset)

    predictions = {
        "patient_id": all_patient_ids,
        "y_true": [int(x) for x in all_labels],
        "y_prob": [float(x) for x in all_probs],
    }

    return metrics, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--bag_size", type=str, required=True, choices=["3", "4", "5", "all"])
    args = parser.parse_args()

    fold = args.fold
    bag_size = None if args.bag_size == "all" else int(args.bag_size)
    bag_tag = "all" if bag_size is None else f"bag{bag_size}"

    seed_everything(RANDOM_SEED)

    print("=" * 70)
    print("PATHOLOGY MIL PATCH-COUNT ABLATION")
    print("=" * 70)
    print(f"Fold: {fold}")
    print(f"Bag size: {args.bag_size}")
    print(f"Device: {DEVICE}")

    train_ds = PathologyBagsDataset(
        folds_csv_path=str(FOLDS_CSV),
        fold=fold,
        split="train",
        img_size=IMG_SIZE,
        bag_size=bag_size
    )

    val_ds = PathologyBagsDataset(
        folds_csv_path=str(FOLDS_CSV),
        fold=fold,
        split="val",
        img_size=IMG_SIZE,
        bag_size=bag_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pathology_bag_collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pathology_bag_collate_fn
    )

    train_df = train_ds.df
    n_neg = int((train_df["label"] == 0).sum())
    n_pos = int((train_df["label"] == 1).sum())

    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    model = AttentionMIL(
        model_name=MODEL_NAME,
        pretrained=True,
        feat_dim=256,
        attn_dim=128
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_auroc = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics, val_preds = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        row = {
            "epoch": epoch,
            "fold": fold,
            "bag_size": args.bag_size,
            "bag_tag": bag_tag,

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

            ckpt_path = CHECKPOINT_DIR / f"best_fold{fold}_{MODEL_NAME}_attnmil_{bag_tag}.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auroc": best_auroc,
                    "fold": fold,
                    "bag_size": args.bag_size,
                    "bag_tag": bag_tag,
                },
                ckpt_path
            )

            pred_path = LOG_DIR / f"best_val_predictions_fold{fold}_{MODEL_NAME}_attnmil_{bag_tag}.csv"
            pd.DataFrame(val_preds).to_csv(pred_path, index=False)

            print(f"Saved best checkpoint: {ckpt_path}")

    history_df = pd.DataFrame(history)

    history_csv = LOG_DIR / f"history_fold{fold}_{MODEL_NAME}_attnmil_{bag_tag}.csv"
    history_df.to_csv(history_csv, index=False)

    best_idx = history_df["val_auroc"].idxmax()

    summary = {
        "fold": fold,
        "bag_size": args.bag_size,
        "bag_tag": bag_tag,
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "train_size": len(train_ds),
        "val_size": len(val_ds),

        "best_epoch_by_auroc": int(history_df.loc[best_idx, "epoch"]),
        "best_val_auroc": float(history_df.loc[best_idx, "val_auroc"]),
        "best_val_auprc": float(history_df.loc[best_idx, "val_auprc"]),
        "best_val_accuracy": float(history_df.loc[best_idx, "val_accuracy"]),
        "best_val_f1": float(history_df.loc[best_idx, "val_f1"]),
        "best_val_mcc": float(history_df.loc[best_idx, "val_mcc"]),
        "best_val_sensitivity": float(history_df.loc[best_idx, "val_sensitivity"]),
        "best_val_specificity": float(history_df.loc[best_idx, "val_specificity"]),
    }

    summary_path = LOG_DIR / f"summary_fold{fold}_{MODEL_NAME}_attnmil_{bag_tag}.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTRAINING COMPLETE")
    print(json.dumps(summary, indent=2))
    print(f"Saved history: {history_csv}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
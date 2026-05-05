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

from dataset import MultimodalPatientDataset, multimodal_collate_fn


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = Path("/Users/behnamkiani/Downloads/Thyroid")
FOLDS_CSV = DATA_ROOT / "outputs" / "folds" / "strict_multimodal_5folds.csv"

OUTPUT_DIR = DATA_ROOT / "outputs" / "multimodal_gated_baseline"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FOLD = None
US_IMG_SIZE = 224
PATH_IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

US_MODEL_NAME = "convnext_tiny"
PATH_MODEL_NAME = "resnet18"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# SEED
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
    thresholds = np.linspace(0.0, 1.0, 101)
    best_thr = 0.5
    best_score = -1e9

    for thr in thresholds:
        y_pred = (np.asarray(y_prob) >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        score = sensitivity + specificity - 1.0

        if score > best_score:
            best_score = score
            best_thr = thr

    return float(best_thr)


# =========================================================
# MODEL
# =========================================================
class UltrasoundEncoder(nn.Module):
    def __init__(self, model_name="convnext_tiny", pretrained=True, feat_dim=256):
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
        x = self.backbone(x)
        x = self.proj(x)
        return x


class PathologyEncoderMIL(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, feat_dim=256, attn_dim=128):
        super().__init__()

        self.patch_backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        in_dim = self.patch_backbone.num_features

        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.attn_v = nn.Linear(feat_dim, attn_dim)
        self.attn_u = nn.Linear(feat_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(self, bag, bag_mask):
        """
        bag: [B, N, C, H, W]
        bag_mask: [B, N]
        """
        B, N, C, H, W = bag.shape

        flat = bag.view(B * N, C, H, W)
        feat = self.patch_backbone(flat)
        feat = self.patch_proj(feat)

        D = feat.shape[-1]
        feat = feat.view(B, N, D)

        A_v = torch.tanh(self.attn_v(feat))
        A_u = torch.sigmoid(self.attn_u(feat))
        A = self.attn_w(A_v * A_u).squeeze(-1)

        A = A.masked_fill(~bag_mask, float("-inf"))
        attn = torch.softmax(A, dim=1)

        bag_feat = torch.sum(attn.unsqueeze(-1) * feat, dim=1)

        return bag_feat, attn


class GatedFusion(nn.Module):
    """
    Learns a dynamic gate between ultrasound and pathology features.

    gate close to 1 = more pathology contribution
    gate close to 0 = more ultrasound contribution
    """
    def __init__(self, feat_dim=256):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, us_feat, path_feat):
        combined = torch.cat([us_feat, path_feat], dim=1)
        gate = self.gate_net(combined)

        fused = gate * path_feat + (1.0 - gate) * us_feat
        fused = self.norm(fused)

        return fused, gate


class MultimodalGatedModel(nn.Module):
    def __init__(
        self,
        us_model_name="convnext_tiny",
        path_model_name="resnet18",
        feat_dim=256
    ):
        super().__init__()

        self.us_encoder = UltrasoundEncoder(
            model_name=us_model_name,
            pretrained=True,
            feat_dim=feat_dim
        )

        self.path_encoder = PathologyEncoderMIL(
            model_name=path_model_name,
            pretrained=True,
            feat_dim=feat_dim,
            attn_dim=128
        )

        self.fusion = GatedFusion(feat_dim=feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, us_img, path_bag, bag_mask):
        us_feat = self.us_encoder(us_img)
        path_feat, attn = self.path_encoder(path_bag, bag_mask)

        fused, gate = self.fusion(us_feat, path_feat)

        logits = self.classifier(fused).squeeze(1)

        return logits, attn, gate


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
        us_img = batch["ultrasound_image"].to(device)
        path_bag = batch["pathology_bag"].to(device)
        bag_mask = batch["bag_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, _, _ = model(us_img, path_bag, bag_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()

        running_loss += loss.item() * us_img.size(0)
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
    all_gate_means = []

    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        us_img = batch["ultrasound_image"].to(device)
        path_bag = batch["pathology_bag"].to(device)
        bag_mask = batch["bag_mask"].to(device)
        labels = batch["label"].to(device)

        logits, attn, gate = model(us_img, path_bag, bag_mask)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        labs = labels.cpu().numpy()

        gate_mean = gate.mean(dim=1).cpu().numpy()

        running_loss += loss.item() * us_img.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())
        all_patient_ids.extend(batch["patient_id"])
        all_gate_means.extend(gate_mean.tolist())

    epoch_loss = running_loss / len(loader.dataset)

    default_metrics = compute_binary_metrics(all_labels, all_probs, threshold=0.5)
    best_thr = find_best_threshold_youden(all_labels, all_probs)
    youden_metrics = compute_binary_metrics(all_labels, all_probs, threshold=best_thr)

    default_metrics["loss"] = epoch_loss
    default_metrics["best_threshold_youden"] = best_thr
    default_metrics["mean_gate_pathology_weight"] = float(np.mean(all_gate_means))

    predictions = {
        "patient_id": all_patient_ids,
        "y_true": [int(x) for x in all_labels],
        "y_prob": [float(x) for x in all_probs],
        "gate_pathology_weight_mean": [float(x) for x in all_gate_means],
    }

    return default_metrics, youden_metrics, predictions


# =========================================================
# MAIN
# =========================================================
def main():
    global FOLD

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold index 0-4")
    args = parser.parse_args()

    FOLD = args.fold

    seed_everything(RANDOM_SEED)

    print("=" * 70)
    print("MULTIMODAL GATED FUSION BASELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Fold: {FOLD}")
    print(f"US model: {US_MODEL_NAME}")
    print(f"Path model: {PATH_MODEL_NAME}")
    print(f"Folds CSV: {FOLDS_CSV}")

    train_ds = MultimodalPatientDataset(
        folds_csv_path=str(FOLDS_CSV),
        fold=FOLD,
        split="train",
        us_img_size=US_IMG_SIZE,
        path_img_size=PATH_IMG_SIZE,
        bag_size=None
    )

    val_ds = MultimodalPatientDataset(
        folds_csv_path=str(FOLDS_CSV),
        fold=FOLD,
        split="val",
        us_img_size=US_IMG_SIZE,
        path_img_size=PATH_IMG_SIZE,
        bag_size=None
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn
    )

    train_df = train_ds.df
    n_neg = int((train_df["label"] == 0).sum())
    n_pos = int((train_df["label"] == 1).sum())

    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    print(f"Train negatives: {n_neg}")
    print(f"Train positives: {n_pos}")
    print(f"pos_weight: {pos_weight.item():.4f}")

    model = MultimodalGatedModel(
        us_model_name=US_MODEL_NAME,
        path_model_name=PATH_MODEL_NAME,
        feat_dim=256
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

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE
        )

        val_metrics, val_metrics_youden, val_preds = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )

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
            "val_mean_gate_pathology_weight": val_metrics["mean_gate_pathology_weight"],

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
            f"Spec={val_metrics['specificity']:.4f} | "
            f"GatePath={val_metrics['mean_gate_pathology_weight']:.4f}"
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]

            ckpt_path = CHECKPOINT_DIR / f"best_fold{FOLD}_multimodal_gated.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auroc": best_auroc,
                    "config": {
                        "fold": FOLD,
                        "us_img_size": US_IMG_SIZE,
                        "path_img_size": PATH_IMG_SIZE,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "us_model_name": US_MODEL_NAME,
                        "path_model_name": PATH_MODEL_NAME,
                    }
                },
                ckpt_path
            )

            pred_path = LOG_DIR / f"best_val_predictions_fold{FOLD}_multimodal_gated.csv"
            pd.DataFrame(val_preds).to_csv(pred_path, index=False)

            print(f"Saved new best checkpoint to: {ckpt_path}")

    history_df = pd.DataFrame(history)

    history_csv = LOG_DIR / f"history_fold{FOLD}_multimodal_gated.csv"
    history_df.to_csv(history_csv, index=False)

    best_idx = history_df["val_auroc"].idxmax()

    summary = {
        "best_val_auroc": float(history_df.loc[best_idx, "val_auroc"]),
        "best_val_auprc": float(history_df.loc[best_idx, "val_auprc"]),
        "best_epoch_by_auroc": int(history_df.loc[best_idx, "epoch"]),
        "best_val_accuracy": float(history_df.loc[best_idx, "val_accuracy"]),
        "best_val_f1": float(history_df.loc[best_idx, "val_f1"]),
        "best_val_mcc": float(history_df.loc[best_idx, "val_mcc"]),
        "best_val_sensitivity": float(history_df.loc[best_idx, "val_sensitivity"]),
        "best_val_specificity": float(history_df.loc[best_idx, "val_specificity"]),
        "best_val_mean_gate_pathology_weight": float(
            history_df.loc[best_idx, "val_mean_gate_pathology_weight"]
        ),
        "fold": FOLD,
        "us_model_name": US_MODEL_NAME,
        "path_model_name": PATH_MODEL_NAME,
        "device": DEVICE,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }

    summary_path = LOG_DIR / f"summary_fold{FOLD}_multimodal_gated.json"

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
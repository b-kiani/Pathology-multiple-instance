import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    brier_score_loss,
)


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = Path("/Users/behnamkiani/Downloads/Thyroid")
OUT_DIR = DATA_ROOT / "outputs" / "paper_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_MIL_DIR = DATA_ROOT / "outputs" / "pathology_mil_baseline" / "logs"
GATED_DIR = DATA_ROOT / "outputs" / "multimodal_gated_baseline" / "logs"

PATH_MIL_SUMMARY = PATH_MIL_DIR / "five_fold_summary_resnet18_attnmil.csv"
GATED_SUMMARY = GATED_DIR / "five_fold_summary_multimodal_gated.csv"

FOLDS = [0, 1, 2, 3, 4]


# =========================================================
# HELPERS
# =========================================================
def mean_std_text(values, decimals=4):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.{decimals}f} ± {values.std(ddof=1):.{decimals}f}"


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    bin_data = []

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        if mask.sum() == 0:
            bin_data.append({
                "bin": i,
                "lower": lower,
                "upper": upper,
                "count": 0,
                "confidence": np.nan,
                "accuracy": np.nan,
            })
            continue

        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        weight = mask.mean()

        ece += weight * abs(acc - conf)

        bin_data.append({
            "bin": i,
            "lower": lower,
            "upper": upper,
            "count": int(mask.sum()),
            "confidence": float(conf),
            "accuracy": float(acc),
        })

    return float(ece), pd.DataFrame(bin_data)


def load_prediction_files(model_name):
    all_rows = []

    if model_name == "pathology_mil":
        pattern = "best_val_predictions_fold{}_resnet18_attnmil.csv"
        base_dir = PATH_MIL_DIR
    elif model_name == "gated":
        pattern = "best_val_predictions_fold{}_multimodal_gated.csv"
        base_dir = GATED_DIR
    else:
        raise ValueError(model_name)

    for fold in FOLDS:
        path = base_dir / pattern.format(fold)

        if not path.exists():
            print(f"Missing prediction file: {path}")
            continue

        df = pd.read_csv(path)
        df["fold"] = fold
        df["model"] = model_name
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(f"No prediction files found for {model_name}")

    return pd.concat(all_rows, ignore_index=True)


# =========================================================
# TABLE 1
# =========================================================
def create_main_results_table():
    path_df = pd.read_csv(PATH_MIL_SUMMARY)
    gated_df = pd.read_csv(GATED_SUMMARY)

    metric_map = {
        "AUROC": "best_val_auroc",
        "AUPRC": "best_val_auprc",
        "Accuracy": "best_val_accuracy",
        "F1-score": "best_val_f1",
        "MCC": "best_val_mcc",
        "Sensitivity": "best_val_sensitivity",
        "Specificity": "best_val_specificity",
    }

    rows = []

    for model_label, df in [
        ("Pathology-only Attention MIL", path_df),
        ("Multimodal Gated Fusion", gated_df),
    ]:
        row = {"Model": model_label}

        for metric_name, col in metric_map.items():
            row[metric_name] = mean_std_text(df[col].values)

        rows.append(row)

    table = pd.DataFrame(rows)

    out_csv = OUT_DIR / "table_1_main_results.csv"
    out_excel = OUT_DIR / "table_1_main_results.xlsx"

    table.to_csv(out_csv, index=False)
    table.to_excel(out_excel, index=False)

    print("\nTABLE 1")
    print(table.to_string(index=False))

    return table


# =========================================================
# FIGURE 1: ROC + PR CURVES
# =========================================================
def plot_roc_and_pr():
    path_preds = load_prediction_files("pathology_mil")
    gated_preds = load_prediction_files("gated")

    combined = pd.concat([path_preds, gated_preds], ignore_index=True)

    combined.to_csv(OUT_DIR / "all_best_fold_predictions.csv", index=False)

    # ROC
    plt.figure(figsize=(7, 6))

    for model_name, label in [
        ("pathology_mil", "Pathology-only MIL"),
        ("gated", "Multimodal Gated Fusion"),
    ]:
        df = combined[combined["model"] == model_name]
        y_true = df["y_true"].values
        y_prob = df["y_prob"].values

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_png = OUT_DIR / "figure_1_roc_comparison.png"
    roc_pdf = OUT_DIR / "figure_1_roc_comparison.pdf"
    plt.savefig(roc_png, dpi=300)
    plt.savefig(roc_pdf)
    plt.close()

    # PR
    plt.figure(figsize=(7, 6))

    for model_name, label in [
        ("pathology_mil", "Pathology-only MIL"),
        ("gated", "Multimodal Gated Fusion"),
    ]:
        df = combined[combined["model"] == model_name]
        y_true = df["y_true"].values
        y_prob = df["y_prob"].values

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, linewidth=2, label=f"{label} (AUPRC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve Comparison")
    plt.legend(loc="lower left")
    plt.tight_layout()

    pr_png = OUT_DIR / "figure_1b_pr_comparison.png"
    pr_pdf = OUT_DIR / "figure_1b_pr_comparison.pdf"
    plt.savefig(pr_png, dpi=300)
    plt.savefig(pr_pdf)
    plt.close()

    print(f"\nSaved ROC figure: {roc_png}")
    print(f"Saved PR figure: {pr_png}")


# =========================================================
# FIGURE 2: CALIBRATION
# =========================================================
def plot_calibration():
    path_preds = load_prediction_files("pathology_mil")
    gated_preds = load_prediction_files("gated")

    cal_rows = []

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")

    for model_name, label in [
        ("pathology_mil", "Pathology-only MIL"),
        ("gated", "Multimodal Gated Fusion"),
    ]:
        df = path_preds if model_name == "pathology_mil" else gated_preds

        y_true = df["y_true"].values
        y_prob = df["y_prob"].values

        ece, bin_df = expected_calibration_error(y_true, y_prob, n_bins=10)
        brier = brier_score_loss(y_true, y_prob)

        bin_df["model"] = label
        bin_df.to_csv(OUT_DIR / f"calibration_bins_{model_name}.csv", index=False)

        valid = bin_df.dropna()
        plt.plot(
            valid["confidence"],
            valid["accuracy"],
            marker="o",
            linewidth=2,
            label=f"{label} (ECE={ece:.3f}, Brier={brier:.3f})"
        )

        cal_rows.append({
            "Model": label,
            "ECE": ece,
            "Brier score": brier,
        })

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()

    cal_png = OUT_DIR / "figure_2_calibration_curve.png"
    cal_pdf = OUT_DIR / "figure_2_calibration_curve.pdf"
    plt.savefig(cal_png, dpi=300)
    plt.savefig(cal_pdf)
    plt.close()

    cal_table = pd.DataFrame(cal_rows)
    cal_table.to_csv(OUT_DIR / "calibration_metrics.csv", index=False)

    print("\nCALIBRATION")
    print(cal_table.to_string(index=False))
    print(f"Saved calibration figure: {cal_png}")


# =========================================================
# FIGURE 3: GATE DISTRIBUTION
# =========================================================
def plot_gate_distribution():
    gated_preds = load_prediction_files("gated")

    if "gate_pathology_weight_mean" not in gated_preds.columns:
        print("No gate_pathology_weight_mean column found.")
        return

    plt.figure(figsize=(7, 5))
    plt.hist(gated_preds["gate_pathology_weight_mean"].values, bins=20)
    plt.xlabel("Mean pathology gate weight")
    plt.ylabel("Number of patients")
    plt.title("Distribution of Learned Pathology Gate Weights")
    plt.tight_layout()

    gate_png = OUT_DIR / "figure_3_gate_weight_distribution.png"
    gate_pdf = OUT_DIR / "figure_3_gate_weight_distribution.pdf"
    plt.savefig(gate_png, dpi=300)
    plt.savefig(gate_pdf)
    plt.close()

    summary = gated_preds["gate_pathology_weight_mean"].describe()
    summary.to_csv(OUT_DIR / "gate_weight_distribution_summary.csv")

    print("\nGATE WEIGHT SUMMARY")
    print(summary)
    print(f"Saved gate figure: {gate_png}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("GENERATING PAPER TABLES AND FIGURES")
    print("=" * 80)

    create_main_results_table()
    plot_roc_and_pr()
    plot_calibration()
    plot_gate_distribution()

    print("\nDone.")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
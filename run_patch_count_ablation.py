import json
import subprocess
from pathlib import Path

import pandas as pd


DATA_ROOT = Path("/Users/behnamkiani/Downloads/Thyroid")
TRAIN_SCRIPT = DATA_ROOT / "train_pathology_mil.py"
LOG_DIR = DATA_ROOT / "outputs" / "pathology_mil_baseline" / "logs"

PYTHON_EXE = "/usr/local/bin/python3.10"
FOLDS = [0, 1, 2, 3, 4]
BAG_SIZES = ["3", "4", "5", "all"]


def run_experiment(fold, bag_size):
    print("\n" + "=" * 90)
    print(f"RUNNING PATCH-COUNT ABLATION | FOLD {fold} | BAG SIZE {bag_size}")
    print("=" * 90)

    cmd = [
        PYTHON_EXE,
        str(TRAIN_SCRIPT),
        "--fold",
        str(fold),
        "--bag_size",
        str(bag_size),
    ]

    subprocess.run(cmd, check=True)


def collect_results():
    rows = []

    for bag_size in BAG_SIZES:
        bag_tag = f"bag{bag_size}" if bag_size != "all" else "all"

        for fold in FOLDS:
            summary_path = LOG_DIR / f"summary_fold{fold}_resnet18_attnmil_{bag_tag}.json"
            history_path = LOG_DIR / f"history_fold{fold}_resnet18_attnmil_{bag_tag}.csv"

            if not summary_path.exists():
                print(f"Missing: {summary_path}")
                continue

            with open(summary_path, "r") as f:
                summary = json.load(f)

            row = {
                "bag_size": bag_size,
                "bag_tag": bag_tag,
                "fold": fold,
                "best_val_auroc": summary.get("best_val_auroc"),
                "best_val_auprc": summary.get("best_val_auprc"),
                "best_epoch_by_auroc": summary.get("best_epoch_by_auroc"),
                "train_size": summary.get("train_size"),
                "val_size": summary.get("val_size"),
                "device": summary.get("device"),
            }

            if history_path.exists():
                hist = pd.read_csv(history_path)
                best_idx = hist["val_auroc"].idxmax()
                best_row = hist.loc[best_idx]

                row.update({
                    "best_val_loss": best_row.get("val_loss"),
                    "best_val_accuracy": best_row.get("val_accuracy"),
                    "best_val_f1": best_row.get("val_f1"),
                    "best_val_mcc": best_row.get("val_mcc"),
                    "best_val_sensitivity": best_row.get("val_sensitivity"),
                    "best_val_specificity": best_row.get("val_specificity"),
                    "best_val_threshold_youden": best_row.get("val_best_threshold_youden"),
                })

            rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = LOG_DIR / "patch_count_ablation_all_folds.csv"
    df.to_csv(out_csv, index=False)

    metric_cols = [
        "best_val_auroc",
        "best_val_auprc",
        "best_val_accuracy",
        "best_val_f1",
        "best_val_mcc",
        "best_val_sensitivity",
        "best_val_specificity",
    ]

    agg_rows = []

    for bag_size, group in df.groupby("bag_size"):
        row = {"bag_size": bag_size}

        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=1)

        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)

    order = {"3": 0, "4": 1, "5": 2, "all": 3}
    agg["order"] = agg["bag_size"].map(order)
    agg = agg.sort_values("order").drop(columns=["order"])

    out_agg_csv = LOG_DIR / "patch_count_ablation_summary.csv"
    agg.to_csv(out_agg_csv, index=False)

    print("\n" + "=" * 90)
    print("PATCH-COUNT ABLATION SUMMARY")
    print("=" * 90)
    print(agg.to_string(index=False))

    print("\nSaved:")
    print(out_csv)
    print(out_agg_csv)


def main():
    for bag_size in BAG_SIZES:
        for fold in FOLDS:
            run_experiment(fold, bag_size)

    collect_results()


if __name__ == "__main__":
    main()
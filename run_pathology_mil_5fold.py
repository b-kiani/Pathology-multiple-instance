import json
import subprocess
from pathlib import Path

import pandas as pd


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = Path("/Users/behnamkiani/Downloads/Thyroid")
TRAIN_SCRIPT = DATA_ROOT / "train_pathology_mil.py"
OUTPUT_DIR = DATA_ROOT / "outputs" / "pathology_mil_baseline"
LOG_DIR = OUTPUT_DIR / "logs"

PYTHON_EXE = "/usr/local/bin/python3.10"
FOLDS = [0, 1, 2, 3, 4]


def run_fold(fold: int):
    print("\n" + "=" * 80)
    print(f"RUNNING PATHOLOGY MIL FOLD {fold}")
    print("=" * 80)

    cmd = [
        PYTHON_EXE,
        str(TRAIN_SCRIPT),
        "--fold",
        str(fold)
    ]

    subprocess.run(cmd, check=True)


def collect_results():
    rows = []

    for fold in FOLDS:
        summary_path = LOG_DIR / f"summary_fold{fold}_resnet18_attnmil.json"
        history_path = LOG_DIR / f"history_fold{fold}_resnet18_attnmil.csv"

        if not summary_path.exists():
            print(f"Missing summary: {summary_path}")
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        row = {
            "fold": fold,
            "best_val_auroc": summary.get("best_val_auroc"),
            "best_val_auprc": summary.get("best_val_auprc"),
            "best_epoch_by_auroc": summary.get("best_epoch_by_auroc"),
            "train_size": summary.get("train_size"),
            "val_size": summary.get("val_size"),
            "device": summary.get("device"),
            "model_name": summary.get("model_name"),
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

    if not rows:
        print("No summaries found.")
        return

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

    out_csv = LOG_DIR / "five_fold_summary_resnet18_attnmil.csv"
    df.to_csv(out_csv, index=False)

    metric_cols = [
        "best_val_auroc",
        "best_val_auprc",
        "best_val_loss",
        "best_val_accuracy",
        "best_val_f1",
        "best_val_mcc",
        "best_val_sensitivity",
        "best_val_specificity",
    ]

    aggregate = {}

    for col in metric_cols:
        if col in df.columns:
            aggregate[f"{col}_mean"] = float(df[col].mean())
            aggregate[f"{col}_std"] = float(df[col].std(ddof=1))

    out_json = LOG_DIR / "five_fold_aggregate_resnet18_attnmil.json"

    with open(out_json, "w") as f:
        json.dump(aggregate, f, indent=2)

    print("\n" + "=" * 80)
    print("PATHOLOGY MIL 5-FOLD SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))

    print("\nAggregate:")
    print(json.dumps(aggregate, indent=2))

    print("\nSaved:")
    print(f"- {out_csv}")
    print(f"- {out_json}")


def main():
    for fold in FOLDS:
        run_fold(fold)

    collect_results()


if __name__ == "__main__":
    main()
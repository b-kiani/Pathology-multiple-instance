from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = Path("/Users/behnamkiani/Downloads/Thyroid")
META_CSV = DATA_ROOT / "outputs" / "metadata" / "patient_metadata.csv"
OUT_DIR = DATA_ROOT / "outputs" / "folds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_SEED = 42


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def summarize_split(df: pd.DataFrame, name: str):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"n = {len(df)}")
    if len(df) > 0:
        print(df["label_name"].value_counts().sort_index())


def make_folds(df: pd.DataFrame, cohort_name: str):
    """
    Build 5-fold stratified CV at patient level.
    """
    df = df.copy().reset_index(drop=True)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    df["fold"] = -1

    X = df["patient_id"].values
    y = df["label"].values

    for fold_id, (_, val_idx) in enumerate(skf.split(X, y)):
        df.loc[val_idx, "fold"] = fold_id

    assert (df["fold"] >= 0).all(), "Some rows did not receive a fold."

    # Save master CSV
    out_csv = OUT_DIR / f"{cohort_name}_5folds.csv"
    df.to_csv(out_csv, index=False)

    # Save per-fold JSON
    folds_json = {}
    for fold_id in range(N_SPLITS):
        train_df = df[df["fold"] != fold_id].copy()
        val_df = df[df["fold"] == fold_id].copy()

        folds_json[f"fold_{fold_id}"] = {
            "train_patient_ids": train_df["patient_id"].tolist(),
            "val_patient_ids": val_df["patient_id"].tolist(),
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "train_class_counts": train_df["label_name"].value_counts().to_dict(),
            "val_class_counts": val_df["label_name"].value_counts().to_dict(),
        }

    out_json = OUT_DIR / f"{cohort_name}_5folds.json"
    save_json(folds_json, out_json)

    # Print summary
    print("\n" + "=" * 70)
    print(f"COHORT: {cohort_name}")
    print("=" * 70)
    summarize_split(df, "Full cohort")

    print("\nFold distribution:")
    for fold_id in range(N_SPLITS):
        fold_df = df[df["fold"] == fold_id]
        print(f"\nFold {fold_id}:")
        print(f"  n = {len(fold_df)}")
        print(fold_df['label_name'].value_counts().sort_index().to_dict())

    print(f"\nSaved:")
    print(f"- {out_csv}")
    print(f"- {out_json}")

    return df


def main():
    if not META_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {META_CSV}")

    df = pd.read_csv(META_CSV)

    print("=" * 70)
    print("LOADED METADATA")
    print("=" * 70)
    print(f"Total rows: {len(df)}")
    print(df.head())

    # -------------------------
    # Strict cohort for main paper
    # -------------------------
    strict_df = df[
        (df["num_ultrasound_images"] == 1) &
        (df["num_patches"].between(3, 7))
    ].copy()

    # -------------------------
    # Relaxed cohort for optional analysis
    # -------------------------
    relaxed_df = df[
        (df["num_ultrasound_images"] == 1) &
        (df["num_patches"] >= 1)
    ].copy()

    # Safety checks
    assert strict_df["patient_id"].is_unique, "Strict cohort patient_id not unique."
    assert relaxed_df["patient_id"].is_unique, "Relaxed cohort patient_id not unique."

    assert strict_df["num_ultrasound_images"].eq(1).all()
    assert strict_df["num_patches"].between(3, 7).all()

    assert relaxed_df["num_ultrasound_images"].eq(1).all()
    assert relaxed_df["num_patches"].ge(1).all()

    strict_out = make_folds(strict_df, "strict_multimodal")
    relaxed_out = make_folds(relaxed_df, "relaxed_multimodal")

    # Save compact patient lists too
    save_json(
        strict_out["patient_id"].tolist(),
        OUT_DIR / "strict_multimodal_patient_ids.json"
    )
    save_json(
        relaxed_out["patient_id"].tolist(),
        OUT_DIR / "relaxed_multimodal_patient_ids.json"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
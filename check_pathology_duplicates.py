import json
import hashlib
from pathlib import Path
from itertools import combinations

import pandas as pd
from PIL import Image
import numpy as np


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = Path("/Users/behnamkiani/Downloads/Thyroid")
FOLDS_CSV = DATA_ROOT / "outputs" / "folds" / "strict_multimodal_5folds.csv"
OUT_DIR = DATA_ROOT / "outputs" / "duplicate_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD = 0

# perceptual hash settings
HASH_SIZE = 8          # 8x8 average hash
HAMMING_THRESHOLD = 4  # smaller = stricter, start with 4


# =========================================================
# HELPERS
# =========================================================
def parse_json_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return json.loads(x)


def md5_file(path: str, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def average_hash(path: str, hash_size: int = 8) -> str:
    """
    Simple perceptual average hash (aHash).
    """
    img = Image.open(path).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    mean_val = arr.mean()
    bits = (arr > mean_val).astype(np.uint8).flatten()
    return "".join(str(int(b)) for b in bits)


def hamming_distance(hash1: str, hash2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def load_patch_table(folds_csv_path: str, fold: int):
    df = pd.read_csv(folds_csv_path)

    rows = []
    for _, row in df.iterrows():
        split = "val" if int(row["fold"]) == fold else "train"
        patch_paths = parse_json_list(row["pathology_patch_paths_json"])

        for p in patch_paths:
            rows.append({
                "patient_id": row["patient_id"],
                "label": int(row["label"]),
                "label_name": row["label_name"],
                "fold": int(row["fold"]),
                "split": split,
                "patch_path": p,
                "filename": Path(p).name,
            })

    patch_df = pd.DataFrame(rows)
    return patch_df


# =========================================================
# EXACT DUPLICATES
# =========================================================
def find_exact_duplicates(patch_df: pd.DataFrame):
    records = []

    for i, row in patch_df.iterrows():
        file_hash = md5_file(row["patch_path"])
        records.append({
            **row.to_dict(),
            "md5": file_hash
        })

    hash_df = pd.DataFrame(records)

    dup_groups = []
    for md5, group in hash_df.groupby("md5"):
        if len(group) > 1:
            dup_groups.append(group.copy())

    if not dup_groups:
        return hash_df, pd.DataFrame()

    dup_df = pd.concat(dup_groups, ignore_index=True)

    pair_rows = []
    for md5, group in dup_df.groupby("md5"):
        rows = group.to_dict("records")
        for a, b in combinations(rows, 2):
            pair_rows.append({
                "dup_type": "exact_md5",
                "md5": md5,
                "path_a": a["patch_path"],
                "path_b": b["patch_path"],
                "file_a": a["filename"],
                "file_b": b["filename"],
                "patient_a": a["patient_id"],
                "patient_b": b["patient_id"],
                "split_a": a["split"],
                "split_b": b["split"],
                "cross_split": a["split"] != b["split"],
                "same_patient": a["patient_id"] == b["patient_id"],
                "label_a": a["label_name"],
                "label_b": b["label_name"],
            })

    pair_df = pd.DataFrame(pair_rows)
    return hash_df, pair_df


# =========================================================
# NEAR DUPLICATES
# =========================================================
def find_near_duplicates(patch_df: pd.DataFrame, hash_size=8, threshold=4):
    records = []

    for _, row in patch_df.iterrows():
        ahash = average_hash(row["patch_path"], hash_size=hash_size)
        records.append({
            **row.to_dict(),
            "ahash": ahash
        })

    hash_df = pd.DataFrame(records)

    # bucket by first part of hash to reduce comparisons a bit
    hash_df["bucket"] = hash_df["ahash"].str[:16]

    pair_rows = []

    for bucket, group in hash_df.groupby("bucket"):
        rows = group.to_dict("records")
        if len(rows) < 2:
            continue

        for a, b in combinations(rows, 2):
            dist = hamming_distance(a["ahash"], b["ahash"])
            if dist <= threshold:
                pair_rows.append({
                    "dup_type": "near_ahash",
                    "hamming_distance": dist,
                    "path_a": a["patch_path"],
                    "path_b": b["patch_path"],
                    "file_a": a["filename"],
                    "file_b": b["filename"],
                    "patient_a": a["patient_id"],
                    "patient_b": b["patient_id"],
                    "split_a": a["split"],
                    "split_b": b["split"],
                    "cross_split": a["split"] != b["split"],
                    "same_patient": a["patient_id"] == b["patient_id"],
                    "label_a": a["label_name"],
                    "label_b": b["label_name"],
                    "ahash_a": a["ahash"],
                    "ahash_b": b["ahash"],
                })

    pair_df = pd.DataFrame(pair_rows)
    return hash_df, pair_df


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("CHECKING PATHOLOGY DUPLICATES")
    print("=" * 80)
    print(f"Fold used for train/val leakage check: {FOLD}")

    patch_df = load_patch_table(str(FOLDS_CSV), FOLD)

    print(f"\nTotal pathology patches in strict cohort: {len(patch_df)}")
    print("\nSplit counts:")
    print(patch_df["split"].value_counts())

    print("\nPatient counts by split:")
    print(patch_df.groupby("split")["patient_id"].nunique())

    # -----------------------------------------------------
    # Exact duplicates
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("EXACT DUPLICATES (MD5)")
    print("=" * 80)

    exact_hash_df, exact_pairs = find_exact_duplicates(patch_df)

    exact_hash_csv = OUT_DIR / f"all_pathology_md5_fold{FOLD}.csv"
    exact_hash_df.to_csv(exact_hash_csv, index=False)

    exact_pairs_csv = OUT_DIR / f"exact_duplicate_pairs_fold{FOLD}.csv"
    exact_pairs.to_csv(exact_pairs_csv, index=False)

    if len(exact_pairs) == 0:
        print("No exact duplicate pairs found.")
    else:
        print(f"Exact duplicate pairs found: {len(exact_pairs)}")
        print(f"Cross-split exact duplicate pairs: {int(exact_pairs['cross_split'].sum())}")
        print(f"Same-patient exact duplicate pairs: {int(exact_pairs['same_patient'].sum())}")

        suspicious_exact = exact_pairs[
            (exact_pairs["cross_split"] == True) &
            (exact_pairs["same_patient"] == False)
        ].copy()

        suspicious_exact_csv = OUT_DIR / f"suspicious_exact_cross_split_fold{FOLD}.csv"
        suspicious_exact.to_csv(suspicious_exact_csv, index=False)

        print(f"Suspicious exact cross-split pairs: {len(suspicious_exact)}")
        if len(suspicious_exact) > 0:
            print("\nTop suspicious exact duplicate pairs:")
            print(suspicious_exact.head(10).to_string(index=False))

    # -----------------------------------------------------
    # Near duplicates
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("NEAR DUPLICATES (aHash)")
    print("=" * 80)
    print(f"Hamming threshold: {HAMMING_THRESHOLD}")

    near_hash_df, near_pairs = find_near_duplicates(
        patch_df,
        hash_size=HASH_SIZE,
        threshold=HAMMING_THRESHOLD
    )

    near_hash_csv = OUT_DIR / f"all_pathology_ahash_fold{FOLD}.csv"
    near_hash_df.to_csv(near_hash_csv, index=False)

    near_pairs_csv = OUT_DIR / f"near_duplicate_pairs_fold{FOLD}.csv"
    near_pairs.to_csv(near_pairs_csv, index=False)

    if len(near_pairs) == 0:
        print("No near-duplicate pairs found at this threshold.")
    else:
        print(f"Near-duplicate pairs found: {len(near_pairs)}")
        print(f"Cross-split near-duplicate pairs: {int(near_pairs['cross_split'].sum())}")
        print(f"Same-patient near-duplicate pairs: {int(near_pairs['same_patient'].sum())}")

        suspicious_near = near_pairs[
            (near_pairs["cross_split"] == True) &
            (near_pairs["same_patient"] == False)
        ].copy()

        suspicious_near = suspicious_near.sort_values("hamming_distance").reset_index(drop=True)

        suspicious_near_csv = OUT_DIR / f"suspicious_near_cross_split_fold{FOLD}.csv"
        suspicious_near.to_csv(suspicious_near_csv, index=False)

        print(f"Suspicious near cross-split pairs: {len(suspicious_near)}")
        if len(suspicious_near) > 0:
            print("\nTop suspicious near-duplicate pairs:")
            print(suspicious_near.head(10).to_string(index=False))

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    summary = {
        "fold": FOLD,
        "total_patches": int(len(patch_df)),
        "exact_duplicate_pairs": int(len(exact_pairs)),
        "exact_cross_split_pairs": int(exact_pairs["cross_split"].sum()) if len(exact_pairs) > 0 else 0,
        "exact_suspicious_cross_split_pairs": int(
            ((exact_pairs["cross_split"] == True) & (exact_pairs["same_patient"] == False)).sum()
        ) if len(exact_pairs) > 0 else 0,
        "near_duplicate_pairs": int(len(near_pairs)),
        "near_cross_split_pairs": int(near_pairs["cross_split"].sum()) if len(near_pairs) > 0 else 0,
        "near_suspicious_cross_split_pairs": int(
            ((near_pairs["cross_split"] == True) & (near_pairs["same_patient"] == False)).sum()
        ) if len(near_pairs) > 0 else 0,
        "ahash_hamming_threshold": HAMMING_THRESHOLD,
    }

    summary_json = OUT_DIR / f"duplicate_summary_fold{FOLD}.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(f"- {exact_hash_csv}")
    print(f"- {exact_pairs_csv}")
    print(f"- {near_hash_csv}")
    print(f"- {near_pairs_csv}")
    print(f"- {summary_json}")


if __name__ == "__main__":
    main()
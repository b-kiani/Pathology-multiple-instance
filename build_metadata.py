import re
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = Path("/Users/behnamkiani/Downloads/Thyroid")
OUT_DIR = DATA_ROOT / "outputs" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SAVE_PARQUET = False   # set True later if pyarrow is installed


# =========================================================
# HELPERS
# =========================================================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTS


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if is_image_file(p)])


def extract_base_id(filename: str, modality: str) -> str:
    """
    Extract within-class patient ID.

    Cytology examples:
      100_001.tif -> 100
      101_004.tif -> 101

    Ultrasound examples:
      1.jpg   -> 1
      10.JPG  -> 10
      101.JPG -> 101
    """
    stem = Path(filename).stem.strip()

    if modality == "cyto":
        m = re.match(r"(\d+)_\d+$", stem)
        if m:
            return m.group(1)

        # fallback: take first numeric token
        m = re.match(r"(\d+)", stem)
        if m:
            return m.group(1)

    elif modality == "ultrasound":
        m = re.match(r"(\d+)$", stem)
        if m:
            return m.group(1)

        # fallback
        m = re.match(r"(\d+)", stem)
        if m:
            return m.group(1)

    return stem.lower().replace(" ", "_")


def make_patient_id(label_name: str, base_id: str) -> str:
    return f"{label_name}_{base_id}"


def scan_class_folder(folder: Path, modality: str, label_name: str, label_int: int):
    records = []
    images = list_images(folder)

    print(f"\nScanning: {folder}")
    print(f"Found {len(images)} image files")

    if images:
        print("Sample files:")
        for p in images[:5]:
            print("  ", p.name)

    for img_path in images:
        base_id = extract_base_id(img_path.name, modality)
        patient_id = make_patient_id(label_name, base_id)

        records.append({
            "patient_id": patient_id,
            "base_id": base_id,
            "modality": modality,
            "label_name": label_name,
            "label": label_int,
            "filename": img_path.name,
            "path": str(img_path.resolve())
        })

    return records


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 70)
    print("DATA ROOT")
    print("=" * 70)
    print(DATA_ROOT)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT}")

    print("\nContents of DATA_ROOT:")
    for item in sorted(DATA_ROOT.iterdir()):
        print(" -", item.name)

    folder_map = {
        "cyto_benign": DATA_ROOT / "Cytological images of benign thyroid lesions",
        "cyto_ptc": DATA_ROOT / "Cytological images of papillary thyroid carcinoma",
        "us_benign": DATA_ROOT / "Ultrasound images of benign thyroid lesions",
        "us_ptc": DATA_ROOT / "Ultrasound images of papillary thyroid carcinoma",
    }

    print("\n" + "=" * 70)
    print("FOLDER CHECK")
    print("=" * 70)
    for key, folder in folder_map.items():
        print(f"{key}: {folder} | exists={folder.exists()} | is_dir={folder.is_dir()}")

    all_records = []
    all_records += scan_class_folder(folder_map["cyto_benign"], "cyto", "benign", 0)
    all_records += scan_class_folder(folder_map["cyto_ptc"], "cyto", "ptc", 1)
    all_records += scan_class_folder(folder_map["us_benign"], "ultrasound", "benign", 0)
    all_records += scan_class_folder(folder_map["us_ptc"], "ultrasound", "ptc", 1)

    if len(all_records) == 0:
        print("\nNo image files were found.")
        return

    raw_df = pd.DataFrame(all_records)
    raw_csv = OUT_DIR / "raw_file_index.csv"
    raw_df.to_csv(raw_csv, index=False)

    print("\n" + "=" * 70)
    print("RAW SUMMARY")
    print("=" * 70)
    print(raw_df.head())

    print("\nRaw modality counts:")
    print(raw_df["modality"].value_counts(dropna=False))

    print("\nRaw label counts:")
    print(raw_df["label_name"].value_counts(dropna=False))

    print("\nUnique patient count by class+modality:")
    print(
        raw_df.groupby(["label_name", "modality"])["patient_id"]
        .nunique()
        .reset_index(name="n_unique_patients")
    )

    patient_dict = defaultdict(lambda: {
        "patient_id": None,
        "base_id": None,
        "label_name": None,
        "label": None,
        "ultrasound_paths": [],
        "pathology_patch_paths": []
    })

    conflict_log = []

    for _, row in raw_df.iterrows():
        pid = row["patient_id"]
        modality = row["modality"]
        label = int(row["label"])
        label_name = row["label_name"]
        base_id = row["base_id"]
        path = row["path"]

        patient_dict[pid]["patient_id"] = pid
        patient_dict[pid]["base_id"] = base_id

        if patient_dict[pid]["label"] is None:
            patient_dict[pid]["label"] = label
            patient_dict[pid]["label_name"] = label_name
        elif patient_dict[pid]["label"] != label:
            conflict_log.append({
                "patient_id": pid,
                "issue": "label_conflict",
                "existing_label": patient_dict[pid]["label"],
                "new_label": label,
                "path": path
            })

        if modality == "ultrasound":
            patient_dict[pid]["ultrasound_paths"].append(path)
        elif modality == "cyto":
            patient_dict[pid]["pathology_patch_paths"].append(path)

    rows = []
    for pid, entry in patient_dict.items():
        us_paths = sorted(entry["ultrasound_paths"])
        cyto_paths = sorted(entry["pathology_patch_paths"])

        rows.append({
            "patient_id": pid,
            "base_id": entry["base_id"],
            "label_name": entry["label_name"],
            "label": entry["label"],
            "num_ultrasound_images": len(us_paths),
            "ultrasound_image_path": us_paths[0] if len(us_paths) > 0 else None,
            "all_ultrasound_paths_json": json.dumps(us_paths),
            "num_patches": len(cyto_paths),
            "pathology_patch_paths_json": json.dumps(cyto_paths),
        })

        if len(us_paths) > 1:
            conflict_log.append({
                "patient_id": pid,
                "issue": "multiple_ultrasound_images",
                "paths": us_paths
            })

    patient_df = pd.DataFrame(rows).sort_values(["label", "base_id"]).reset_index(drop=True)

    patient_df["has_ultrasound"] = patient_df["num_ultrasound_images"] >= 1
    patient_df["exactly_one_ultrasound"] = patient_df["num_ultrasound_images"] == 1
    patient_df["patch_count_ok"] = patient_df["num_patches"].between(3, 7)
    patient_df["valid_pair"] = patient_df["has_ultrasound"] & (patient_df["num_patches"] >= 1)

    patient_csv = OUT_DIR / "patient_metadata.csv"
    patient_df.to_csv(patient_csv, index=False)

    if SAVE_PARQUET:
        try:
            patient_parquet = OUT_DIR / "patient_metadata.parquet"
            patient_df.to_parquet(patient_parquet, index=False)
        except Exception as e:
            print(f"\nParquet save skipped: {e}")

    with open(OUT_DIR / "conflicts.json", "w") as f:
        json.dump(conflict_log, f, indent=2)

    summary = {
        "total_raw_files": int(len(raw_df)),
        "total_patients_found": int(len(patient_df)),
        "benign_patients": int((patient_df["label"] == 0).sum()),
        "ptc_patients": int((patient_df["label"] == 1).sum()),
        "patients_with_ultrasound": int(patient_df["has_ultrasound"].sum()),
        "patients_with_exactly_one_ultrasound": int(patient_df["exactly_one_ultrasound"].sum()),
        "patients_patch_count_3_to_7": int(patient_df["patch_count_ok"].sum()),
        "patients_valid_pair": int(patient_df["valid_pair"].sum()),
        "patients_missing_ultrasound": int((~patient_df["has_ultrasound"]).sum()),
        "patients_with_patch_count_outside_3_7": int((~patient_df["patch_count_ok"]).sum()),
        "patients_with_multiple_ultrasound_images": int((patient_df["num_ultrasound_images"] > 1).sum()),
        "num_conflicts": int(len(conflict_log)),
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("PATIENT SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nPatient count by label:")
    print(patient_df["label_name"].value_counts())

    print("\nPatch count distribution:")
    print(patient_df["num_patches"].value_counts().sort_index())

    print("\nUltrasound count distribution:")
    print(patient_df["num_ultrasound_images"].value_counts().sort_index())

    print("\nSaved files:")
    print(f"- {raw_csv}")
    print(f"- {patient_csv}")
    print(f"- {OUT_DIR / 'conflicts.json'}")
    print(f"- {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
from torch.utils.data import DataLoader

from dataset import (
    UltrasoundDataset,
    PathologyBagsDataset,
    MultimodalPatientDataset,
    pathology_bag_collate_fn,
    multimodal_collate_fn,
)

FOLDS_CSV = "/Users/behnamkiani/Downloads/Thyroid/outputs/folds/strict_multimodal_5folds.csv"


def test_ultrasound():
    ds = UltrasoundDataset(
        folds_csv_path=FOLDS_CSV,
        fold=0,
        split="train",
        img_size=224
    )
    print("\n[UltrasoundDataset]")
    print("len =", len(ds))
    sample = ds[0]
    print("image shape:", sample["image"].shape)
    print("label:", sample["label"])
    print("patient_id:", sample["patient_id"])


def test_pathology():
    ds = PathologyBagsDataset(
        folds_csv_path=FOLDS_CSV,
        fold=0,
        split="train",
        img_size=224,
        bag_size=None
    )
    print("\n[PathologyBagsDataset]")
    print("len =", len(ds))
    sample = ds[0]
    print("bag shape:", sample["bag"].shape)
    print("label:", sample["label"])
    print("patient_id:", sample["patient_id"])

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        collate_fn=pathology_bag_collate_fn
    )
    batch = next(iter(loader))
    print("loader bag shape:", batch["bag"].shape)
    print("loader bag_mask shape:", batch["bag_mask"].shape)
    print("loader label shape:", batch["label"].shape)


def test_multimodal():
    ds = MultimodalPatientDataset(
        folds_csv_path=FOLDS_CSV,
        fold=0,
        split="train",
        us_img_size=224,
        path_img_size=224,
        bag_size=None
    )
    print("\n[MultimodalPatientDataset]")
    print("len =", len(ds))
    sample = ds[0]
    print("ultrasound shape:", sample["ultrasound_image"].shape)
    print("pathology bag shape:", sample["pathology_bag"].shape)
    print("label:", sample["label"])
    print("patient_id:", sample["patient_id"])

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        collate_fn=multimodal_collate_fn
    )
    batch = next(iter(loader))
    print("loader ultrasound shape:", batch["ultrasound_image"].shape)
    print("loader pathology bag shape:", batch["pathology_bag"].shape)
    print("loader bag_mask shape:", batch["bag_mask"].shape)
    print("loader label shape:", batch["label"].shape)


if __name__ == "__main__":
    test_ultrasound()
    test_pathology()
    test_multimodal()
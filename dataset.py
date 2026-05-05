import json
import random
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# =========================================================
# IMAGE HELPERS
# =========================================================
def pil_loader(path: str):
    img = Image.open(path).convert("RGB")
    return img


def get_ultrasound_train_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_ultrasound_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_pathology_train_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_pathology_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =========================================================
# DATAFRAME HELPERS
# =========================================================
def load_fold_dataframe(
    folds_csv_path: str,
    fold: int,
    split: str
) -> pd.DataFrame:
    """
    split: 'train' or 'val'
    """
    df = pd.read_csv(folds_csv_path)
    if split == "train":
        df = df[df["fold"] != fold].copy()
    elif split == "val":
        df = df[df["fold"] == fold].copy()
    else:
        raise ValueError("split must be 'train' or 'val'")
    return df.reset_index(drop=True)


def parse_json_list(x) -> List[str]:
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return json.loads(x)


# =========================================================
# DATASETS
# =========================================================
class UltrasoundDataset(Dataset):
    def __init__(
        self,
        folds_csv_path: str,
        fold: int,
        split: str,
        img_size: int = 224,
        transform=None
    ):
        self.df = load_fold_dataframe(folds_csv_path, fold, split)

        self.df = self.df[self.df["num_ultrasound_images"] == 1].reset_index(drop=True)

        if transform is None:
            transform = (
                get_ultrasound_train_transform(img_size)
                if split == "train"
                else get_ultrasound_val_transform(img_size)
            )

        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = pil_loader(row["ultrasound_image_path"])
        image = self.transform(image)

        label = torch.tensor(int(row["label"]), dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "patient_id": row["patient_id"]
        }


class PathologyBagsDataset(Dataset):
    """
    Returns one patient bag of pathology patches.
    Suitable for MIL.
    """
    def __init__(
        self,
        folds_csv_path: str,
        fold: int,
        split: str,
        img_size: int = 224,
        transform=None,
        bag_size: Optional[int] = None,
        sample_strategy: str = "random"
    ):
        """
        bag_size:
            None -> use all patches
            int  -> sample/pad to fixed bag size

        sample_strategy:
            'random' or 'first'
        """
        self.df = load_fold_dataframe(folds_csv_path, fold, split)

        self.df = self.df[self.df["num_patches"] >= 1].reset_index(drop=True)

        if transform is None:
            transform = (
                get_pathology_train_transform(img_size)
                if split == "train"
                else get_pathology_val_transform(img_size)
            )

        self.transform = transform
        self.bag_size = bag_size
        self.sample_strategy = sample_strategy
        self.split = split

    def __len__(self):
        return len(self.df)

    def _choose_patch_paths(self, patch_paths: List[str]) -> List[str]:
        if self.bag_size is None:
            return patch_paths

        n = len(patch_paths)

        if n >= self.bag_size:
            if self.sample_strategy == "random" and self.split == "train":
                return random.sample(patch_paths, self.bag_size)
            return patch_paths[:self.bag_size]

        # if n < bag_size, repeat samples to reach fixed size
        chosen = patch_paths.copy()
        while len(chosen) < self.bag_size:
            if self.sample_strategy == "random" and self.split == "train":
                chosen.append(random.choice(patch_paths))
            else:
                chosen.append(patch_paths[len(chosen) % n])
        return chosen

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patch_paths = parse_json_list(row["pathology_patch_paths_json"])
        patch_paths = self._choose_patch_paths(patch_paths)

        bag = []
        for p in patch_paths:
            img = pil_loader(p)
            img = self.transform(img)
            bag.append(img)

        bag = torch.stack(bag, dim=0)  # [N, C, H, W]
        label = torch.tensor(int(row["label"]), dtype=torch.float32)

        return {
            "bag": bag,
            "bag_length": torch.tensor(len(patch_paths), dtype=torch.long),
            "label": label,
            "patient_id": row["patient_id"]
        }


class MultimodalPatientDataset(Dataset):
    """
    Returns:
      - one ultrasound image
      - one pathology bag
      - one patient label
    """
    def __init__(
        self,
        folds_csv_path: str,
        fold: int,
        split: str,
        us_img_size: int = 224,
        path_img_size: int = 224,
        us_transform=None,
        path_transform=None,
        bag_size: Optional[int] = None,
        sample_strategy: str = "random"
    ):
        self.df = load_fold_dataframe(folds_csv_path, fold, split)

        self.df = self.df[
            (self.df["num_ultrasound_images"] == 1) &
            (self.df["num_patches"] >= 1)
        ].reset_index(drop=True)

        if us_transform is None:
            us_transform = (
                get_ultrasound_train_transform(us_img_size)
                if split == "train"
                else get_ultrasound_val_transform(us_img_size)
            )
        if path_transform is None:
            path_transform = (
                get_pathology_train_transform(path_img_size)
                if split == "train"
                else get_pathology_val_transform(path_img_size)
            )

        self.us_transform = us_transform
        self.path_transform = path_transform
        self.bag_size = bag_size
        self.sample_strategy = sample_strategy
        self.split = split

    def __len__(self):
        return len(self.df)

    def _choose_patch_paths(self, patch_paths: List[str]) -> List[str]:
        if self.bag_size is None:
            return patch_paths

        n = len(patch_paths)

        if n >= self.bag_size:
            if self.sample_strategy == "random" and self.split == "train":
                return random.sample(patch_paths, self.bag_size)
            return patch_paths[:self.bag_size]

        chosen = patch_paths.copy()
        while len(chosen) < self.bag_size:
            if self.sample_strategy == "random" and self.split == "train":
                chosen.append(random.choice(patch_paths))
            else:
                chosen.append(patch_paths[len(chosen) % n])
        return chosen

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        us_img = pil_loader(row["ultrasound_image_path"])
        us_img = self.us_transform(us_img)

        patch_paths = parse_json_list(row["pathology_patch_paths_json"])
        patch_paths = self._choose_patch_paths(patch_paths)

        path_bag = []
        for p in patch_paths:
            img = pil_loader(p)
            img = self.path_transform(img)
            path_bag.append(img)

        path_bag = torch.stack(path_bag, dim=0)  # [N, C, H, W]
        label = torch.tensor(int(row["label"]), dtype=torch.float32)

        return {
            "ultrasound_image": us_img,
            "pathology_bag": path_bag,
            "bag_length": torch.tensor(len(patch_paths), dtype=torch.long),
            "label": label,
            "patient_id": row["patient_id"]
        }


# =========================================================
# COLLATE FUNCTIONS
# =========================================================
def multimodal_collate_fn(batch):
    """
    Pads pathology bags in a batch to max bag length.
    """
    us_images = torch.stack([x["ultrasound_image"] for x in batch], dim=0)
    labels = torch.stack([x["label"] for x in batch], dim=0)
    patient_ids = [x["patient_id"] for x in batch]

    bag_lengths = torch.tensor([x["pathology_bag"].shape[0] for x in batch], dtype=torch.long)
    max_len = int(bag_lengths.max().item())

    c, h, w = batch[0]["pathology_bag"].shape[1:]
    padded_bags = []
    bag_masks = []

    for x in batch:
        bag = x["pathology_bag"]
        n = bag.shape[0]

        if n < max_len:
            pad = torch.zeros((max_len - n, c, h, w), dtype=bag.dtype)
            bag = torch.cat([bag, pad], dim=0)

        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:n] = True

        padded_bags.append(bag)
        bag_masks.append(mask)

    padded_bags = torch.stack(padded_bags, dim=0)   # [B, N, C, H, W]
    bag_masks = torch.stack(bag_masks, dim=0)       # [B, N]

    return {
        "ultrasound_image": us_images,
        "pathology_bag": padded_bags,
        "bag_mask": bag_masks,
        "bag_length": bag_lengths,
        "label": labels,
        "patient_id": patient_ids
    }


def pathology_bag_collate_fn(batch):
    """
    Pads pathology bags in a batch to max bag length.
    """
    labels = torch.stack([x["label"] for x in batch], dim=0)
    patient_ids = [x["patient_id"] for x in batch]

    bag_lengths = torch.tensor([x["bag"].shape[0] for x in batch], dtype=torch.long)
    max_len = int(bag_lengths.max().item())

    c, h, w = batch[0]["bag"].shape[1:]
    padded_bags = []
    bag_masks = []

    for x in batch:
        bag = x["bag"]
        n = bag.shape[0]

        if n < max_len:
            pad = torch.zeros((max_len - n, c, h, w), dtype=bag.dtype)
            bag = torch.cat([bag, pad], dim=0)

        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:n] = True

        padded_bags.append(bag)
        bag_masks.append(mask)

    padded_bags = torch.stack(padded_bags, dim=0)   # [B, N, C, H, W]
    bag_masks = torch.stack(bag_masks, dim=0)       # [B, N]

    return {
        "bag": padded_bags,
        "bag_mask": bag_masks,
        "bag_length": bag_lengths,
        "label": labels,
        "patient_id": patient_ids
    }
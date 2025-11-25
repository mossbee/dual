"""
NDTWIN-2009-2010 dataset wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class NDTWINSample:
    path: Path
    label: int
    bbox: Optional[Tuple[float, float, float, float]]


@dataclass
class NDTWINVerificationPair:
    path1: Path
    path2: Path
    label: int  # 0 = different, 1 = same


def _read_key_value_txt(path: Path) -> List[Tuple[int, str]]:
    entries = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(" ", 1)
            entries.append((int(key), value))
    return entries


def _read_bboxes(path: Path):
    boxes = {}
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_id = int(parts[0])
            boxes[img_id] = tuple(float(v) for v in parts[1:5])
    return boxes


def _load_test_pairs(root: Path) -> List[NDTWINVerificationPair]:
    """Load verification pairs from test_pairs.txt."""
    pairs = []
    pairs_file = root / "test_pairs.txt"
    if not pairs_file.exists():
        raise FileNotFoundError(f"test_pairs.txt not found at {pairs_file}")
    with pairs_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            path1, path2, label_str = parts
            label = int(label_str)
            pairs.append(
                NDTWINVerificationPair(
                    path1=root / "ND_TWIN_448" / path1,
                    path2=root / "ND_TWIN_448" / path2,
                    label=label,
                )
            )
    return pairs


class NDTWINDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        crop_bbox: bool = False,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(root)
        self.transform = transform
        self.crop_bbox = crop_bbox
        self.samples = self._build_samples(split)

    def _build_samples(self, split: str) -> List[NDTWINSample]:
        images = dict(_read_key_value_txt(self.root / "images.txt"))
        labels = {img_id: int(lbl) for img_id, lbl in _read_key_value_txt(self.root / "image_class_labels.txt")}
        splits = {img_id: int(flag) for img_id, flag in _read_key_value_txt(self.root / "train_test_split.txt")}
        bbox_map = _read_bboxes(self.root / "bounding_boxes.txt") if self.crop_bbox else {}
        offset_labels = {img_id: lbl - 1 for img_id, lbl in labels.items()}
        desired_flag = 1 if split == "train" else 0
        samples: List[NDTWINSample] = []
        for img_id, path in images.items():
            if splits[img_id] != desired_flag:
                continue
            bbox = None
            if self.crop_bbox:
                bbox = bbox_map.get(img_id)
            samples.append(
                NDTWINSample(
                    path=self.root / "ND_TWIN_448" / path,
                    label=offset_labels[img_id],
                    bbox=bbox,
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.crop_bbox and sample.bbox:
            x, y, w, h = sample.bbox
            image = image.crop((x, y, x + w, y + h))
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": sample.label, "path": str(sample.path)}


class NDTWINVerificationDataset(Dataset):
    """Dataset for loading verification pairs from test_pairs.txt."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.pairs = _load_test_pairs(self.root)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        image1 = Image.open(pair.path1).convert("RGB")
        image2 = Image.open(pair.path2).convert("RGB")
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return {
            "image1": image1,
            "image2": image2,
            "label": pair.label,
            "path1": str(pair.path1),
            "path2": str(pair.path2),
        }

    def get_unique_images(self) -> List[Path]:
        """Return list of all unique image paths in the pairs."""
        unique_paths = set()
        for pair in self.pairs:
            unique_paths.add(pair.path1)
            unique_paths.add(pair.path2)
        return sorted(list(unique_paths))


class NDTWINUniqueImageDataset(Dataset):
    """Dataset for loading unique images for feature extraction."""

    def __init__(self, image_paths: List[Path], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "path": str(path)}



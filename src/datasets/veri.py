"""
VeRi-776 dataset wrapper for ReID.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class VeRiSample:
    path: Path
    pid: int
    camid: int


def _parse_label_file(path: Path) -> Dict[str, Dict[str, int]]:
    tree = ET.parse(path)
    root = tree.getroot()
    mapping: Dict[str, Dict[str, int]] = {}
    for item in root.iter():
        if item.tag.lower() != "item":
            continue
        name = (
            item.findtext("imageName")
            or item.findtext("name")
            or item.findtext("filename")
            or item.findtext("ImageName")
        )
        if not name:
            continue
        vid = item.findtext("vehicleID") or item.findtext("vid")
        cam = item.findtext("cameraID") or item.findtext("camid")
        if vid is None or cam is None:
            continue
        mapping[name] = {"vid": int(vid), "cam": int(cam)}
    if not mapping:
        raise ValueError(f"No entries found in {path}")
    return mapping


def _read_names(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names


class VeRiDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
    ):
        if split not in {"train", "query", "gallery"}:
            raise ValueError("split must be train/query/gallery")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = self._build_samples()
        self.labels = [sample.pid for sample in self.samples]

    def _build_samples(self) -> List[VeRiSample]:
        if self.split == "train":
            img_dir = "image_train"
            names_file = "name_train.txt"
            label_xml = "train_label.xml"
            relabel = True
        elif self.split == "query":
            img_dir = "image_query"
            names_file = "name_query.txt"
            label_xml = "test_label.xml"
            relabel = False
        else:
            img_dir = "image_test"
            names_file = "name_test.txt"
            label_xml = "test_label.xml"
            relabel = False
        mapping = _parse_label_file(self.root / label_xml)
        names = _read_names(self.root / names_file)
        samples: List[VeRiSample] = []
        pid_map: Dict[int, int] = {}
        next_pid = 0
        for name in names:
            if name not in mapping:
                continue
            meta = mapping[name]
            pid = meta["vid"]
            camid = meta["cam"]
            if relabel:
                if pid not in pid_map:
                    pid_map[pid] = next_pid
                    next_pid += 1
                pid = pid_map[pid]
            samples.append(
                VeRiSample(
                    path=self.root / img_dir / name,
                    pid=pid,
                    camid=camid,
                )
            )
        if not samples:
            raise ValueError(f"No samples found for split {self.split}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "pid": sample.pid, "camid": sample.camid, "path": str(sample.path)}



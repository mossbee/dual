"""
Data utilities: transforms, loaders, identity samplers.
"""
from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence

from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_cub_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((550, 550)),
                transforms.RandomCrop(448),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

def build_ndtwin_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_veri_transforms(train: bool, size=(256, 256)):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.Pad(10),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class RandomIdentitySampler(Sampler[int]):
    """
    Samples P identities per batch, each with K instances.
    """

    def __init__(self, labels: Sequence[int], batch_size: int, instances_per_id: int):
        if batch_size % instances_per_id != 0:
            raise ValueError("batch_size must be divisible by instances_per_id")
        self.labels = list(labels)
        self.instances_per_id = instances_per_id
        self.p_per_batch = batch_size // instances_per_id
        self.index_dic = {}
        for idx, label in enumerate(self.labels):
            self.index_dic.setdefault(label, []).append(idx)
        self.length = len(self.labels)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        pids = list(self.index_dic.keys())
        random.shuffle(pids)
        batch = []
        for pid in pids:
            idxs = self.index_dic[pid]
            if len(idxs) < self.instances_per_id:
                idxs = idxs + random.choices(idxs, k=self.instances_per_id - len(idxs))
            else:
                idxs = random.sample(idxs, self.instances_per_id)
            batch.extend(idxs)
            if len(batch) == self.p_per_batch * self.instances_per_id:
                for index in batch:
                    yield index
                batch = []


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    sampler: Optional[Sampler] = None,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = sampler is None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=sampler is not None,
    )



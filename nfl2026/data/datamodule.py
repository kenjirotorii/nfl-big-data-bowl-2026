import random
from pathlib import Path

import lightning as L
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from .dataclass import Prompt


class Transform:
    def __call__(self, prompt: Prompt) -> Prompt:
        raise NotImplementedError


class Transforms(Transform):
    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, prompt: Prompt) -> Prompt:
        for transform in self.transforms:
            prompt = transform(prompt)
        return prompt


class FlipPlay(Transform):
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, prompt: Prompt) -> Prompt:
        if random.random() < self.prob:
            return prompt.flip_along_x
        return prompt


class TemporalShift(Transform):
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, prompt: Prompt) -> Prompt:
        if random.random() < self.prob:
            return prompt.temporal_shift
        return prompt


class CustomDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        custom_transform: Transform | None = None,
    ):
        super().__init__()
        self.files = files
        self.custom_transform = custom_transform

    def len(self) -> int:
        return len(self.files)

    def get(self, idx: int) -> Prompt:
        prompt: Prompt = torch.load(self.files[idx], weights_only=False)
        if self.custom_transform is not None:
            prompt = self.custom_transform(prompt)
        return prompt


def split_train_val(
    files: list[Path], train_ratio: float = 0.8, is_shuffled: bool = False
) -> dict[str, list[Path]]:
    if is_shuffled:
        random.shuffle(files)
    num_train_files = int(len(files) * train_ratio)
    return {"train": files[:num_train_files], "val": files[num_train_files:]}


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        test_batch_size: int = 4,
        trainval_dir: Path = Path("data/prompts/internal"),
        external_trainval_dir: Path | None = None,
        fold: int = 0,
        transforms: list[Transform] | None = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        _transforms: list[Transform] = [FlipPlay()]
        if transforms is not None:
            _transforms += transforms
        self.transforms = Transforms(_transforms)

        trainval_files = sorted(trainval_dir.glob("*.pt"))
        self.train_files = [f for f in trainval_files if f"fold{fold}" not in f.stem]
        self.val_files = [f for f in trainval_files if f"fold{fold}" in f.stem]

        if external_trainval_dir is not None:
            external_trainval_files = sorted(external_trainval_dir.glob("*.pt"))
            self.train_files += [
                f for f in external_trainval_files if f"fold{fold}" not in f.stem
            ]

    def setup(self, stage: str) -> None:
        self.train_dataset = CustomDataset(
            files=self.train_files, custom_transform=self.transforms
        )
        self.val_dataset = CustomDataset(files=self.val_files, custom_transform=None)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch_geometric.data import Batch, HeteroData


@dataclass
class Record:
    pred: Float[Tensor, "A T XY"]
    target: Float[Tensor, "A T XY"]
    ball_land_xy: Float[Tensor, "2"]
    rmse: Float[Tensor, ""]
    num_frames_output: Int[Tensor, ""]

    def to_heterodata(self) -> HeteroData:
        instance = HeteroData()

        player = instance["player"]
        player.num_nodes = self.pred.shape[0]
        player["pred"] = self.pred
        player["target"] = self.target

        meta = instance["metadata"]
        meta.num_nodes = 1
        meta["ball_land_xy"] = self.ball_land_xy.unsqueeze(0)
        meta["num_frames_output"] = self.num_frames_output.unsqueeze(0)
        meta["rmse"] = self.rmse.unsqueeze(0)

        return instance


@dataclass
class RecordQueue:
    max_queue: int
    records: list[Record] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.records = sorted(self.records, key=lambda record: -record.rmse.item())
        self.records = self.records[: self.max_queue]

    @staticmethod
    def new(max_queue: int) -> "RecordQueue":
        return RecordQueue(max_queue=max_queue, records=[])

    def add(self, record: Record) -> "RecordQueue":
        return RecordQueue(max_queue=self.max_queue, records=self.records + [record])

    def __iter__(self) -> Generator[Record, None, None]:
        yield from self.records

    def __len__(self) -> int:
        return len(self.records)

    def save(self, save_path: Path) -> None:
        batch = Batch.from_data_list(
            [record.to_heterodata() for record in self.records]
        )
        torch.save(batch, save_path)

    def clear(self) -> None:
        self.records.clear()


class PredictionRecorder:
    def __init__(self, queue_size: int = 1000) -> None:
        self.queue_size = queue_size
        self.record = RecordQueue.new(queue_size)

    def __call__(self, record: Record, pass_result: int) -> None:
        self.record = self.record.add(record)

    def save(self, save_dir: Path) -> None:
        if len(self.record) > 0:
            self.record.save(save_dir.joinpath(f"worst{self.queue_size}.pt"))

    def reset(self) -> None:
        self.record.clear()

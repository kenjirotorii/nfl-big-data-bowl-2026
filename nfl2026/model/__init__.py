import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import lightning as L
import pandas as pd
import torch
import torch.optim as optim
from jaxtyping import Float, Int
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.loggers.logger import DummyLogger
from torch import Tensor, nn
from torch.optim import lr_scheduler

from ..data.dataclass import XMAX, YMAX, PlayDirection, Prompt
from .decoder import TrajectoryDecoder
from .edge_connector import (
    AgentToBallSpatialEdgeConnector,
    AgentToPolylineSpatialEdgeConnector,
    InterAgentSpatialEdgeConnector,
    InterAgentTemporalEdgeConnector,
    ModeToFeatureEdgeConnector,
)
from .edge_selector import (
    KNNEdgeSelector,
    RadiusEdgeSelector,
    RadiusGraphEdgeSelector,
)
from .encoder import TrajectoryEncoder
from .graph_builder import GraphBuilder, ModeGraphBuilder
from .loss import GaussianMixtureLoss
from .metric import RMSEMetricForEachResult
from .recorder import PredictionRecorder, Record


class Model(Protocol):
    def __call__(
        self, prompt: Prompt, *args: Any, **kwargs: Any
    ) -> Float[Tensor, "A1+A2+~+AN num_output_frames D=5"]:
        pass

    def parameters(self, recurse: bool = True) -> Iterator[nn.parameter.Parameter]:
        pass


def build_trajectory(
    model: Model, prompt: Prompt
) -> Float[Tensor, "A1+A2+~+AN num_output_frames 2"]:
    pred_traj = model(prompt)
    return torch.cumsum(pred_traj[..., :2], dim=-2)


class TrajModel(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_freq_bands: int = 32,
        num_heads: int = 4,
        head_dim: int = 8,
        dropout: float = 0.1,
        num_layers: int = 4,
        learning_rate: float = 0.001,
        min_learning_rate: float = 0.01,
        learning_warmup_steps: int = 0,
        num_input_frames: int = 10,
        num_output_frames: int = 50,
        is_casual: bool = True,
        loss: nn.Module | None = None,
        step_size: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.build_graph = GraphBuilder(
            temporal_edge_connector=InterAgentTemporalEdgeConnector(
                max_historical_aggregate_steps=5, is_casual=is_casual
            ),
            interaction_edge_connector=InterAgentSpatialEdgeConnector(
                edge_selector=RadiusGraphEdgeSelector(
                    search_radius=50.0, max_num_neighbors=22
                )
            ),
            map_edge_connector=AgentToPolylineSpatialEdgeConnector(
                edge_selector=RadiusEdgeSelector(
                    search_radius=60.0, max_num_neighbors=100
                )
            ),
            ball_edge_connector=AgentToBallSpatialEdgeConnector(
                edge_selector=KNNEdgeSelector(k=1)
            ),
        )
        self.build_mode_graph = ModeGraphBuilder(
            feature_edge_connector=ModeToFeatureEdgeConnector()
        )

        self.encoder = TrajectoryEncoder(
            hidden_dim, num_freq_bands, num_layers, num_heads, head_dim, dropout
        )
        self.decoder = TrajectoryDecoder(
            hidden_dim,
            num_heads,
            head_dim,
            dropout,
            num_output_frames,
            step_size,
        )

        if loss is None:
            loss = GaussianMixtureLoss()
        self.loss = loss
        self.metric = RMSEMetricForEachResult()
        self.recorder = PredictionRecorder(queue_size=20)

        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_warmup_steps = learning_warmup_steps
        self.num_output_frames = num_output_frames
        self.num_input_frames = num_input_frames

    def forward(
        self, prompt: Prompt
    ) -> Float[Tensor, "A1+A2+~+AN num_output_frames D=5"]:
        new_prompt = prompt.get(self.num_input_frames)
        graph = self.build_graph(new_prompt)
        mode_graph = self.build_mode_graph(new_prompt)
        emb = self.encoder(graph)
        pred_traj = self.decoder(emb, mode_graph)
        return pred_traj

    def tta(self, prompt: Prompt) -> Float[Tensor, "A1+A2+~+AN num_output_frames XY"]:
        pred_xy = build_trajectory(self, prompt)
        pred_xy_flip = build_trajectory(self, prompt.flip_along_x)
        pred_xy_flip[..., 1] *= -1
        return (pred_xy + pred_xy_flip) / 2.0

    def training_step(self, batch: Prompt, batch_idx: int) -> Float[Tensor, ""]:
        loss = self.compute_loss(batch)
        self.log(
            name="train_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.batch_size,
        )
        return loss

    def validation_step(self, batch: Prompt, batch_idx: int) -> None:
        loss = self.compute_loss(batch)
        self.log(
            name="valid_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.batch_size,
        )
        self.compute_metric(batch)

    def on_validation_epoch_end(self) -> None:
        rmse = self.metric.compute()
        self.log_dict(rmse, prog_bar=True, sync_dist=True)
        self.metric.reset()
        self.save_record()

    def save_record(self) -> None:
        if self.logger is None:
            raise ValueError("The logger is not set.")
        if isinstance(self.logger, MLFlowLogger):
            log_dir = f"{self.logger.experiment.tracking_uri}/{self.logger.experiment_id}/{self.logger.run_id}/artifacts"
        elif isinstance(self.logger.log_dir, str):
            log_dir = self.logger.log_dir
        elif isinstance(self.logger, DummyLogger):
            log_dir = "/tmp/"
        else:
            raise ValueError("Unsupported logger type or log directory.")
        self.recorder.save(Path(log_dir))
        self.recorder.reset()

    def compute_loss(self, batch: Prompt) -> Float[Tensor, ""]:
        pred_traj = self(batch)
        truth = batch["target"]["xy"]
        batched_num_frames_output = batch["metadata"]["num_frames_output"]
        losses = []
        for batch_index, num_frames_output in enumerate(batched_num_frames_output):
            is_current_player = batch["player"].batch == batch_index
            is_current_target = batch["target"].batch == batch_index
            num_frames_to_evaluate = min(self.num_output_frames, num_frames_output)
            is_predicted = batch["player"]["is_predicted"][is_current_player]

            _pred_traj = pred_traj[is_current_player, :num_frames_to_evaluate][
                is_predicted
            ]
            _truth = truth[is_current_target, :num_frames_to_evaluate]
            loss = self.loss(_pred_traj, _truth)

            losses.append(loss)

        return torch.cat(losses, dim=0).mean()

    def compute_metric(self, batch: Prompt) -> None:
        pred_xy = self.tta(batch)
        batch = batch.get(self.num_input_frames)
        truth = batch["target"]["xy"]
        batched_num_frames_output = batch["metadata"]["num_frames_output"]
        pass_results = batch["pass"]["result"]
        ball_land_xy = batch["ball"]["xyh"][:, :2]
        last_observed_xy = batch["player"]["xyh"][:, -1, :2]
        for batch_index, num_frames_output in enumerate(batched_num_frames_output):
            is_current_player = batch["player"].batch == batch_index
            is_current_target = batch["target"].batch == batch_index
            num_frames_to_evaluate = min(self.num_output_frames, num_frames_output)
            is_predicted = batch["player"]["is_predicted"][is_current_player]

            pred = pred_xy[is_current_player, :num_frames_to_evaluate][is_predicted]
            target = truth[is_current_target, :num_frames_to_evaluate]

            self.metric.update(pred, target, pass_results[batch_index].item())

            _last_observed_xy = last_observed_xy[is_current_player][is_predicted][
                :, None, :
            ].expand(-1, self.num_output_frames, -1)
            _pred = (
                pred_xy[is_current_player, : self.num_output_frames][is_predicted]
                + _last_observed_xy
            )
            _target = (
                truth[is_current_target, : self.num_output_frames] + _last_observed_xy
            )
            self.recorder(
                record=Record(
                    pred=_pred.cpu(),
                    target=_target.cpu(),
                    ball_land_xy=ball_land_xy[batch_index].cpu(),
                    rmse=torch.sqrt(torch.mean((pred - target) ** 2)).cpu(),
                    num_frames_output=num_frames_output,
                ),
                pass_result=pass_results[batch_index].item(),
            )

    def inference(self, batch: Prompt) -> pd.DataFrame:
        self.eval()
        pred_xy = self.tta(batch)
        batch = batch.get(self.num_input_frames)
        batched_num_frames_output = batch["metadata"]["num_frames_output"]
        nfl_ids = batch["player"]["id"]
        last_xys = batch["player"]["xyh"][:, -1:, :2]

        output = pd.DataFrame()
        for batch_index, num_frames_output in enumerate(batched_num_frames_output):
            is_current_player = batch["player"].batch == batch_index
            pred = pred_xy[is_current_player, :num_frames_output]
            nfl_id = nfl_ids[is_current_player]
            last_xy = last_xys[is_current_player]
            if pred.shape[1] < num_frames_output:
                pad = pred[:, -1:].repeat(1, num_frames_output - pred.shape[1], 1)
                pred = torch.cat([pred, pad], dim=1)

            is_predicted = batch["player"]["is_predicted"][is_current_player]
            pred_global = pred[is_predicted] + last_xy[is_predicted]

            play_direction = batch["metadata"]["play_direction"][batch_index].item()
            pred_global = self._change_to_original_coordinate(
                pred_global, int(play_direction)
            )
            nfl_id = nfl_id[is_predicted]

            output = pd.concat([output, self._create_submission(pred_global, nfl_id)])

        return output

    def _change_to_original_coordinate(
        self, pred: Float[torch.Tensor, "A T 2"], play_direction: int
    ) -> Float[torch.Tensor, "A T 2"]:
        if play_direction == PlayDirection.right.value:
            return pred
        elif play_direction == PlayDirection.left.value:
            pred[..., 0] = XMAX - pred[..., 0]
            pred[..., 1] = YMAX - pred[..., 1]
            return pred
        else:
            raise ValueError("Invalid play_direction")

    def _create_submission(
        self, pred: Float[torch.Tensor, "A T 2"], nfl_id: Int[torch.Tensor, "A 1"]
    ) -> pd.DataFrame:
        A, T = pred.shape[:2]
        assert nfl_id.shape[0] == A
        nfl_id = nfl_id.repeat(1, T)
        frames = torch.arange(T).unsqueeze(0).repeat(A, 1) + 1
        df = pd.DataFrame()
        df["nfl_id"] = nfl_id.flatten().cpu().numpy()
        df["frame_id"] = frames.flatten().cpu().numpy()
        df["x"] = pred[..., 0].flatten().detach().cpu().numpy()
        df["y"] = pred[..., 1].flatten().detach().cpu().numpy()

        return df

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        lr_total_steps = self._trainer.max_epochs if self._trainer.max_epochs else 10

        def lr_lambda(current_step: int) -> float:
            # Override current_step with the epoch-based step count
            step = self.current_epoch + 1

            if step < self.learning_warmup_steps:
                warmup_factor = step / self.learning_warmup_steps
                return (
                    self.min_learning_rate
                    + (1 - self.min_learning_rate) * warmup_factor
                )

            progress = (step - self.learning_warmup_steps) / (
                lr_total_steps - self.learning_warmup_steps
            )
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            return self.min_learning_rate + (1 - self.min_learning_rate) * cosine_decay

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]

import tempfile
from datetime import datetime
from pathlib import Path

from jsonargparse import Namespace
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)
from lightning.pytorch.cli import LightningArgumentParser, SaveConfigCallback


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath: str | Path,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | None = None,
        save_top_k: int = 1,
        mode: str = "min",
    ) -> None:
        run_name = get_current_time()
        super().__init__(
            dirpath=Path(dirpath) / run_name,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            mode=mode,
        )


class MLFlowSaveConfigCallback(SaveConfigCallback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
    ):
        super().__init__(
            parser, config, config_filename, overwrite, multifile, save_to_log_dir=False
        )

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if trainer.is_global_zero:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / "config.yaml"
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
                trainer.logger.experiment.log_artifact(
                    local_path=config_path, run_id=trainer.logger.run_id
                )

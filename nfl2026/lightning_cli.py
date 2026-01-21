from lightning.pytorch.cli import LightningCLI

from .cli_utils import MLFlowSaveConfigCallback
from .data.datamodule import DataModule


def main() -> None:
    # simple usage: poetry run cli fit --config data/models/model.yaml
    LightningCLI(
        datamodule_class=DataModule,
        save_config_callback=MLFlowSaveConfigCallback,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    main()

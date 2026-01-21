# nfl-big-data-bowl-2026

11th Place Solution for Kaggle Competition "NFL Big Data Bowl 2026 - Prediction"

## Preliminary

- Ubuntu: 22.04
- CUDA: 12.8
- VSCode
- uv

## Data Preparation

You need to download datasets from kaggle competition pages. Once you download datasets, move it to create the following structure.
```shell
nfl-big-data-bowl-2026
└── data
    ├── 2026    # NFL Big Data Bowl 2026 - Prediction
    │   ├── input_2023_w01.csv
    │   ├── ...
    │   ├── output_2023_w18.csv
    │   └── supplementary.csv # from "NFL Big Data Bowl 2026 - Analytics"
    └── 2021    # NFL Big Data Bowl 2021
        ├── raw
        │   ├── games.csv
        │   ├── players.csv
        │   ├── plays.csv
        │   ├── week1.csv
        │   ├── ...
        │   └── week17.csv
        └── processed
```

After you completed to prepare datasets, run the following commands to create cache data for training.

```shell
uv sync
uv run preprocess_external_data
uv run build_dataset
```

## Training

```shell
uv run cli fit --config data/models/model.yaml
```
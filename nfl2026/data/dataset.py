from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .dataclass import (
    XMAX,
    YMAX,
    PassAttribute,
    PassCoverage,
    PassResult,
    PlayerPosition,
    PlayerRole,
    Prompts,
)
from .feature_builder import build_feature


def convert_player_height(height: str) -> float:
    ft, inches = height.split("-")
    return float(ft) * 12.0 + float(inches)


def convert_player_role(player_role: str) -> int:
    return PlayerRole[player_role.replace(" ", "")].value


def convert_player_position(player_position: str) -> int:
    if player_position in PlayerPosition._member_names_:
        return PlayerPosition[player_position].value
    return PlayerPosition.Other.value


def wrap_angle_deg(s: np.ndarray) -> np.ndarray:
    return ((s + 180.0) % 360.0) - 180.0


def convert_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["player_height"] = [convert_player_height(d) for d in df["player_height"]]
    df["player_weight"] = [float(d) for d in df["player_weight"]]
    df["s"] = df["s"].fillna(0.0)
    df["a"] = df["a"].fillna(0.0)
    df["o"] = df["o"].fillna(0.0)
    df["dir"] = df["dir"].fillna(0.0)

    for col in ["o", "dir"]:
        deg = wrap_angle_deg(df[col].values.astype(float))
        deg = -wrap_angle_deg(deg - 90.0)
        df[col] = np.deg2rad(deg)
    df["player_weight"] = df["player_weight"].fillna(df["player_weight"].median())
    df["player_role"] = [convert_player_role(d) for d in df["player_role"]]
    df["player_position"] = [convert_player_position(d) for d in df["player_position"]]
    return df


def unify_right_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror rightward plays so all samples are 'right' oriented (x,y, dir, o, ball_land)."""
    if "play_direction" not in df.columns:
        return df
    df = df.copy()
    is_left = df["play_direction"].eq("left")
    # positions
    if "x" in df.columns:
        df.loc[is_left, "x"] = XMAX - df.loc[is_left, "x"]
    if "y" in df.columns:
        df.loc[is_left, "y"] = YMAX - df.loc[is_left, "y"]
    # angles in degrees
    for col in ("dir", "o"):
        if col in df.columns:
            df.loc[is_left, col] = (df.loc[is_left, col] + 180.0) % 360.0
    # ball landing
    if "ball_land_x" in df.columns:
        df.loc[is_left, "ball_land_x"] = XMAX - df.loc[is_left, "ball_land_x"]
    if "ball_land_y" in df.columns:
        df.loc[is_left, "ball_land_y"] = YMAX - df.loc[is_left, "ball_land_y"]
    return df


def add_direction_map(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    map = (
        input_df.groupby("uid")
        .apply(lambda x: x["play_direction"].unique()[0])
        .to_frame()
        .reset_index()
    )
    map.columns = ["uid", "play_direction"]
    return pd.merge(output_df, map, "left", "uid")


def add_unique_play_id(df: pd.DataFrame) -> pd.DataFrame:
    df["uid"] = [f"{g}_{p}" for g, p in zip(df["game_id"], df["play_id"], strict=False)]
    return df


def sort_by_nfl_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["nfl_id", "frame_id"])


def get_pass_attribute(supplementary_file: Path) -> dict[str, PassAttribute]:
    df = pd.read_csv(supplementary_file)
    df = add_unique_play_id(df)
    pass_results = {
        str(df.loc[i, "uid"]): str(df.loc[i, "pass_result"]) for i in range(df.shape[0])
    }

    df = df.fillna({"team_coverage_man_zone": "UNKNOWN_COVERAGE"})
    pass_coverage = {
        str(df.loc[i, "uid"]): str(df.loc[i, "team_coverage_man_zone"]).replace(
            "_COVERAGE", ""
        )
        for i in range(df.shape[0])
    }

    return {
        uid: PassAttribute(
            torch.tensor([PassResult[pass_results[uid]].value], dtype=torch.long),
            torch.tensor([PassCoverage[pass_coverage[uid]].value], dtype=torch.long),
        )
        for uid in pass_results.keys()
    }


def has_one_qb(df: pd.DataFrame) -> bool:
    df_passer = df[df["player_role"] == PlayerRole.Passer.value]
    passer_nfl_id = df_passer["nfl_id"].unique()
    return len(passer_nfl_id) == 1


def has_players_to_predict(df: pd.DataFrame) -> bool:
    return True in df["player_to_predict"].unique()


def has_more_than_50step_to_predict(df: pd.DataFrame) -> bool:
    return int(df["num_frames_output"].unique()[0]) > 50


def remove_prompt_with_more_than_50step_to_predict(prompts: Prompts) -> Prompts:
    return [prompt for prompt in prompts if prompt.num_frames_output[0] <= 50]


def create_dataset_per_file(
    input_file: Path,
    output_file: Path | None = None,
    pass_attribute: dict[str, PassAttribute] | None = None,
    max_timesteps: int = 15,
) -> Prompts:
    input_df = pd.read_csv(input_file)
    input_df = unify_right_direction(input_df)
    input_df = convert_dataframe(input_df)
    input_df = add_unique_play_id(input_df)

    if output_file is not None:
        output_df = pd.read_csv(output_file)
        output_df = add_unique_play_id(output_df)
        output_df = add_direction_map(input_df, output_df)
        output_df = unify_right_direction(output_df)

    uids = input_df["uid"].unique()

    prompts: Prompts = []
    for i in uids:
        _input_df = input_df[input_df["uid"] == i]
        _input_df = sort_by_nfl_id(_input_df)

        if not has_one_qb(_input_df):
            continue

        if not has_players_to_predict(_input_df):
            continue

        if has_more_than_50step_to_predict(_input_df):
            continue

        if output_file is None:
            _prompts = build_feature(_input_df, max_timesteps=max_timesteps)
        else:
            if pass_attribute is None:
                raise ValueError("pass_attribute must have some values")
            _output_df = output_df[output_df["uid"] == i]
            _output_df = sort_by_nfl_id(_output_df)
            _prompts = build_feature(
                _input_df, _output_df, pass_attribute[i], max_timesteps=max_timesteps
            )
        prompts += _prompts

    return prompts


def create_dataset(
    input_files: Sequence[Path],
    output_files: Sequence[Path],
    supplementary_file: Path,
    save_dir: Path,
    max_timesteps: int = 10,
    num_folds: int = 5,
) -> None:
    pass_attribute = get_pass_attribute(supplementary_file)
    count: int = 0
    for input_file, output_file in tqdm(zip(input_files, output_files, strict=False)):
        prompts = create_dataset_per_file(
            input_file,
            output_file=output_file,
            pass_attribute=pass_attribute,
            max_timesteps=max_timesteps,
        )
        for prompt in prompts:
            fold = count % num_folds
            torch.save(
                prompt, save_dir.joinpath(f"prompt_{str(count).zfill(6)}_fold{fold}.pt")
            )
            count += 1


def create_test_dataset(test_input: pd.DataFrame, max_timesteps: int = 20) -> Prompts:
    test_input = unify_right_direction(test_input)
    test_input = convert_dataframe(test_input)
    test_input = add_unique_play_id(test_input)

    uids = test_input["uid"].unique()

    prompts: Prompts = []
    for i in uids:
        _input_df = test_input[test_input["uid"] == i]
        _input_df = sort_by_nfl_id(_input_df)
        prompts += build_feature(_input_df, max_timesteps=max_timesteps)
    return prompts


def main() -> None:
    max_timesteps = 20

    # internal data
    trainval_dir = Path("data/2026")
    save_dir = Path("data/prompts/2026")
    save_dir.mkdir(parents=True, exist_ok=True)
    trainval_input_files = sorted(trainval_dir.glob("input*"))
    trainval_output_files = sorted(trainval_dir.glob("output*"))
    create_dataset(
        trainval_input_files,
        trainval_output_files,
        trainval_dir.joinpath("supplementary_data.csv"),
        save_dir,
        max_timesteps=max_timesteps,
    )

    # external data
    trainval_dir = Path("data/2021/processed")
    save_dir = Path("data/prompts/2021")
    save_dir.mkdir(parents=True, exist_ok=True)
    trainval_input_files = sorted(trainval_dir.glob("input*"))
    trainval_output_files = sorted(trainval_dir.glob("output*"))
    create_dataset(
        trainval_input_files,
        trainval_output_files,
        trainval_dir.joinpath("supplementary_data.csv"),
        save_dir,
        max_timesteps=max_timesteps,
    )


if __name__ == "__main__":
    main()

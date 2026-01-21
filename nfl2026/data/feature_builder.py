import math

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float

from ..geometry import wrap_angle
from .dataclass import (
    XMAX,
    YMAX,
    MetaData,
    PassAttribute,
    PlayDirection,
    Player,
    Polyline,
    Prompt,
    Prompts,
    Target,
)


def aggregate_feature(
    df: pd.DataFrame, feature_name: str
) -> Float[torch.Tensor, "num_players num_timesteps"]:
    assert feature_name in df.columns
    unique_nfl_id = df["nfl_id"].unique()
    feature = []
    for i in unique_nfl_id:
        p = df[df["nfl_id"] == i]
        feature.append(p[feature_name].values)

    return torch.from_numpy(np.stack(feature, axis=0))


def build_player(df: pd.DataFrame, max_timesteps: int = 10) -> Player:
    time_dependent_features = ["x", "y", "dir", "s", "a", "o"]
    time_independent_features = [
        "nfl_id",
        "player_role",
        "player_position",
        "player_to_predict",
        "player_height",
        "player_weight",
    ]
    features: dict[str, torch.Tensor] = {}
    for name in time_dependent_features:
        feature = aggregate_feature(df, name)
        feature = feature.float()
        if name in ["dir", "o"]:
            feature = wrap_angle(feature)

        features[name] = feature

    features["valid"] = torch.ones_like(features["x"]).bool()

    A, T = features["x"].shape[:2]
    if T < max_timesteps:
        features["valid"] = torch.cat(
            [torch.zeros((A, max_timesteps - T)).bool(), features["valid"]], dim=-1
        )
        for name in time_dependent_features:
            padding = features[name][:, 0:1].repeat(1, max_timesteps - T)
            features[name] = torch.cat([padding, features[name]], dim=-1)
    else:
        features["valid"] = features["valid"][:, -max_timesteps:]
        for name in time_dependent_features:
            features[name] = features[name][:, -max_timesteps:]

    for name in time_independent_features:
        features[name] = aggregate_feature(df, name)[:, 0]

    return Player(**features)


def build_metadata(df: pd.DataFrame) -> MetaData:
    game_id = df["game_id"].unique()
    play_id = df["play_id"].unique()
    num_frames_output = df["num_frames_output"].unique()
    ball_land_x = df["ball_land_x"].unique()
    ball_land_y = df["ball_land_y"].unique()
    play_direction = df["play_direction"].unique()
    absolute_yardline_number = df["absolute_yardline_number"].unique()

    if (
        game_id.shape[0] > 1
        or play_id.shape[0] > 1
        or num_frames_output.shape[0] > 1
        or ball_land_x.shape[0] > 1
        or ball_land_y.shape[0] > 1
        or play_direction.shape[0] > 1
        or absolute_yardline_number.shape[0] > 1
    ):
        raise ValueError("There are not unique feature.")

    return MetaData(
        game_id=torch.from_numpy(game_id),
        play_id=torch.from_numpy(play_id),
        num_frames_output=torch.from_numpy(num_frames_output).int(),
        play_direction=torch.tensor([PlayDirection[play_direction[0]].value]).long(),
        ball_land_x=torch.from_numpy(ball_land_x).float(),
        ball_land_y=torch.from_numpy(ball_land_y).float(),
        scrimage_line_x=torch.from_numpy(absolute_yardline_number).float(),
    )


def build_target(input_df: pd.DataFrame, output_df: pd.DataFrame) -> Target:
    if output_df.shape[0] == 0:
        raise ValueError("Empty dataframe")
    x = aggregate_feature(output_df, "x")
    y = aggregate_feature(output_df, "y")
    nfl_id = aggregate_feature(output_df, "nfl_id")[:, 0]
    return Target(x, y, nfl_id)


def build_polylines(df: pd.DataFrame) -> Polyline:
    absolute_yardline_number = float(df["absolute_yardline_number"].values[0])

    visitor_sideline_x = torch.arange(10, XMAX - 10, 5).float()
    visitor_sideline_y = torch.full(visitor_sideline_x.shape, YMAX)
    visitor_sideline_h = torch.full(visitor_sideline_x.shape, 0.0)
    visitor_sideline_xyh = torch.stack(
        [visitor_sideline_x, visitor_sideline_y, visitor_sideline_h], dim=-1
    )

    home_sideline_y = torch.full(visitor_sideline_x.shape, 0.0)
    home_sideline_xyh = torch.stack(
        [visitor_sideline_x, home_sideline_y, visitor_sideline_h], dim=-1
    )

    home_endzone_x = torch.cat(
        [
            torch.zeros(10),
            torch.tensor([0, 5]),
            torch.full((10,), 5),
            torch.tensor([10, 5]),
        ]
    )
    home_endzone_y = torch.cat(
        [
            torch.arange(0, YMAX, 5.33),
            torch.full((2,), YMAX),
            torch.arange(0, YMAX, 5.33).flip(dims=[0]),
            torch.full((2,), 0.0),
        ]
    )
    home_endzone_h = torch.cat(
        [
            torch.full((10,), math.pi / 2.0),
            torch.full((2,), 0.0),
            torch.full((10,), -math.pi / 2.0),
            torch.full((2,), -math.pi),
        ]
    )
    home_endzone_xyh = torch.stack(
        [home_endzone_x, home_endzone_y, home_endzone_h], dim=-1
    )

    visitor_endzone_x = torch.cat(
        [
            torch.full((10,), XMAX),
            torch.tensor([XMAX, XMAX - 5]),
            torch.full((10,), XMAX - 10),
            torch.tensor([XMAX - 10, XMAX - 5]),
        ]
    )
    visitor_endzone_y = torch.cat(
        [
            torch.arange(0, YMAX, 5.33).flip(dims=[0]),
            torch.full((2,), 0.0),
            torch.arange(0, YMAX, 5.33),
            torch.full((2,), YMAX),
        ]
    )
    visitor_endzone_h = torch.cat(
        [
            torch.full((10,), -math.pi / 2.0),
            torch.full((2,), -math.pi),
            torch.full((10,), math.pi / 2.0),
            torch.full((2,), 0.0),
        ]
    )
    visitor_endzone_xyh = torch.stack(
        [visitor_endzone_x, visitor_endzone_y, visitor_endzone_h], dim=-1
    )

    scrimage_line_x = torch.full((10,), absolute_yardline_number)
    scrimage_line_y = torch.arange(0, YMAX, 5.33)
    scrimage_line_h = torch.full((10,), math.pi / 2.0)

    scrimage_line_xyh = torch.stack(
        [scrimage_line_x, scrimage_line_y, scrimage_line_h], dim=-1
    )
    hash_left_y = torch.full(visitor_sideline_x.shape, YMAX / 2.0 + 3.0)
    hash_left_xyh = torch.stack(
        [visitor_sideline_x, hash_left_y, visitor_sideline_h], dim=-1
    )
    hash_right_y = torch.full(visitor_sideline_x.shape, YMAX / 2.0 - 3.0)
    hash_right_xyh = torch.stack(
        [visitor_sideline_x, hash_right_y, visitor_sideline_h], dim=-1
    )

    xyh = torch.cat(
        [
            home_sideline_xyh,
            visitor_sideline_xyh,
            home_endzone_xyh,
            visitor_endzone_xyh,
            scrimage_line_xyh,
            hash_left_xyh,
            hash_right_xyh,
        ],
        dim=0,
    )
    type_int_list = (
        [0] * home_sideline_xyh.shape[0]
        + [1] * visitor_sideline_xyh.shape[0]
        + [2] * home_endzone_xyh.shape[0]
        + [3] * visitor_endzone_xyh.shape[0]
        + [4] * scrimage_line_xyh.shape[0]
        + [5] * hash_left_xyh.shape[0]
        + [6] * hash_right_xyh.shape[0]
    )
    type_int = torch.tensor(type_int_list, dtype=torch.long)
    return Polyline(xyh, type_int)


def build_feature(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame | None = None,
    pass_attribute: PassAttribute | None = None,
    max_timesteps: int = 10,
) -> Prompts:
    metadata = build_metadata(input_df)
    polyline = build_polylines(input_df)
    player = build_player(input_df, max_timesteps)

    if output_df is None:
        return [Prompt.create(player, polyline, metadata)]

    if pass_attribute is None:
        raise ValueError("pass_attribute must have some value")
    target = build_target(input_df, output_df)
    return [Prompt.create(player, polyline, metadata, target, pass_attribute)]

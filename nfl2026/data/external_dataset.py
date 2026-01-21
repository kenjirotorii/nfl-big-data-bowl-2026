from logging import warning
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from tqdm import tqdm


def parse_receiver_name_for_complete_pass(description: str) -> str:
    if "pass" not in description:
        raise ValueError("The description must have 'pass'")
    return description.split("pass ")[1].split("to ")[1].split(" ")[0].split(".")[-1]


def parse_receiver_name_for_incomplete_pass(description: str) -> str:
    if "pass" not in description:
        return ""
    if "INTERCEPTED" in description:
        return ""
    direction_and_name = description.split("pass ")[1].split("to ")
    if len(direction_and_name) < 2:
        return ""
    name_and_noise = direction_and_name[1]
    name = name_and_noise.split(" ")[0]
    if len(name.split(".")) < 2:
        return ""
    return name.split(".")[1]


def parse_receiver_name_for_intercepted_pass(description: str) -> str:
    if "intended for" not in description:
        return ""
    return description.split("intended for ")[1].split(" ")[0].split(".")[1]


def parse_defense_name_for_intercepted_pass(description: str) -> str:
    if "INTERCEPTED by" not in description:
        raise ValueError("The description does not have 'INTERCEPTED by'")
    return description.split("INTERCEPTED by ")[1].split(" ")[0].split(".")[1]


def preprocess_play_data(df: pd.DataFrame) -> pd.DataFrame:
    df["team_coverage_man_zone"] = "UNKNOWN_COVERAGE"
    df = df.rename(
        columns={
            "gameId": "game_id",
            "playId": "play_id",
            "absoluteYardlineNumber": "absolute_yardline_number",
            "passResult": "pass_result",
        }
    )
    df = df.drop(
        [
            "quarter",
            "down",
            "yardsToGo",
            "possessionTeam",
            "playType",
            "yardlineSide",
            "yardlineNumber",
            "offenseFormation",
            "personnelO",
            "defendersInTheBox",
            "numberOfPassRushers",
            "personnelD",
            "typeDropback",
            "preSnapVisitorScore",
            "preSnapHomeScore",
            "gameClock",
            "penaltyCodes",
            "penaltyJerseyNumbers",
            "offensePlayResult",
            "playResult",
            "epa",
            "isDefensivePI",
        ],
        axis=1,
    )
    df["id"] = [f"{g}_{p}" for g, p in zip(df["game_id"], df["play_id"], strict=False)]
    df = df[~df["absolute_yardline_number"].isna()]

    # complete pass
    completed_pass_plays = df.query("pass_result == 'C'")
    completed_pass_plays["target_receiver_name"] = [
        parse_receiver_name_for_complete_pass(desc)
        for desc in completed_pass_plays["playDescription"]
    ]
    completed_pass_plays["intercepted_name"] = [
        "" for _ in range(completed_pass_plays.shape[0])
    ]

    # incomplete pass
    incompleted_pass_plays = df.query("pass_result == 'I'")
    incompleted_pass_plays["target_receiver_name"] = [
        parse_receiver_name_for_incomplete_pass(desc)
        for desc in incompleted_pass_plays["playDescription"]
    ]
    incompleted_pass_plays = incompleted_pass_plays[
        incompleted_pass_plays["target_receiver_name"] != ""
    ]
    incompleted_pass_plays["intercepted_name"] = [
        "" for _ in range(incompleted_pass_plays.shape[0])
    ]

    # intercepted pass
    intercepted_pass_plays = df.query("pass_result == 'IN'")
    intercepted_pass_plays["target_receiver_name"] = [
        parse_receiver_name_for_intercepted_pass(desc)
        for desc in intercepted_pass_plays["playDescription"]
    ]
    intercepted_pass_plays = intercepted_pass_plays[
        intercepted_pass_plays["target_receiver_name"] != ""
    ]
    intercepted_pass_plays["intercepted_name"] = [
        parse_defense_name_for_intercepted_pass(desc)
        for desc in intercepted_pass_plays["playDescription"]
    ]

    pass_plays = pd.concat(
        [completed_pass_plays, incompleted_pass_plays, intercepted_pass_plays]
    )

    return pass_plays.reset_index(drop=True)


def preprocess_player_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "nflId": "nfl_id",
            "height": "player_height",
            "weight": "player_weight",
        }
    )
    df = df.drop(
        ["birthDate", "collegeName", "position", "displayName"],
        axis=1,
    )

    def modify_height(height: int | str) -> str:
        h_str = str(height)
        if "-" in h_str:
            return h_str
        assert len(h_str) == 2
        return f"{h_str[0]}-{h_str[1]}"

    df["player_height"] = [modify_height(h) for h in df["player_height"]]
    return df


def preprocess_tracking_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "nflId": "nfl_id",
            "frameId": "frame_id",
            "gameId": "game_id",
            "playId": "play_id",
            "playDirection": "play_direction",
            "displayName": "player_name",
            "position": "player_position",
        }
    )
    df = df.query("player_name != 'Football'")
    df["id"] = [f"{g}_{p}" for g, p in zip(df["game_id"], df["play_id"], strict=False)]
    df["player_name"] = [d.split(" ")[-1] for d in df["player_name"]]
    df["nfl_id"] = df["nfl_id"].values.astype(int)
    df["player_position"] = [
        "RB" if pos == "HB" else pos for pos in df["player_position"]
    ]
    df["is_offense"] = [
        pos in ["WR", "TE", "RB", "FB", "QB"] for pos in df["player_position"]
    ]
    df = df.drop(["time", "dis", "jerseyNumber", "team", "route"], axis=1)
    return df.reset_index(drop=True)


BALL_LAND_MEAN = np.array([-0.04627916, 0.07634005])
BALL_LAND_COV = np.array([[4.59965055, 0.18354845], [0.18354845, 3.17532727]])


def get_ball_land_xy(
    tracking: pd.DataFrame,
    frame_id_on_pass_received: int,
    target_receiver_name: str,
    intercepted_name: str,
    pass_result: str,
    play_direction: str,
) -> Float[np.ndarray, "1 2"]:
    df = tracking[tracking["frame_id"] == frame_id_on_pass_received]
    if pass_result == "C":
        df = df[df["player_name"] == target_receiver_name]
        df = df.query("is_offense == True")
        xy = df[["x", "y"]].values
        return xy
    if pass_result == "IN":
        if intercepted_name == "":
            raise ValueError("We can not identify the player to intercept.")
        df = df[df["player_name"] == intercepted_name]
        df = df.query("is_offense == False")
        xy = df[["x", "y"]].values
        return xy
    if pass_result == "I":
        df = df[df["player_name"] == target_receiver_name]
        df = df.query("is_offense == True")
        xy = df[["x", "y"]].values
        if play_direction == "right":
            gaussian_noise = np.random.multivariate_normal(
                BALL_LAND_MEAN, BALL_LAND_COV, (1,)
            )
        elif play_direction == "left":
            gaussian_noise = np.random.multivariate_normal(
                -BALL_LAND_MEAN, BALL_LAND_COV, (1,)
            )
        else:
            raise ValueError("play_direction must be left or right")
        return xy + gaussian_noise
    raise ValueError("Invalid pass result")


def identify_target_receiver_id(
    tracking: pd.DataFrame, target_receiver_name: str
) -> int:
    df = tracking[tracking["player_name"] == target_receiver_name]
    df = df.query("is_offense == True")
    nfl_id = df["nfl_id"].unique()
    if nfl_id.shape[0] != 1:
        warning("We cannot identify the received player nfl_id")
        return -1
    return int(nfl_id[0])


def identify_defensive_players_to_predict(
    tracking: pd.DataFrame,
    frame_id_on_pass_released: int,
    target_receiver_name: str,
    airtime: int,
    ball_land_xy: Float[np.ndarray, "1 2"],
) -> list[int]:
    track_on_released = tracking[tracking["frame_id"] == frame_id_on_pass_released]
    target_receiver = track_on_released[
        track_on_released["player_name"] == target_receiver_name
    ].query("is_offense == True")
    if target_receiver.shape[0] != 1:
        raise ValueError("We cannot identify the received player")

    receiver_xy = target_receiver[["x", "y"]].values  # (1, 2)
    defense_players = track_on_released.query("is_offense == False")
    defense_players_xy = defense_players[["x", "y"]].values  # (N, 2)
    distance = np.sqrt(np.sum((defense_players_xy - receiver_xy) ** 2, axis=-1))
    is_within_5yd = distance < 5.0

    max_speed = 12.0  # yards/sec
    dt = 0.1
    distance_to_move_on_air = max_speed * airtime * dt
    distance = np.sqrt(np.sum((defense_players_xy - ball_land_xy) ** 2, axis=-1))
    is_reachable = distance < distance_to_move_on_air

    is_player_to_predict = is_within_5yd | is_reachable
    defensive_nfl_ids = defense_players["nfl_id"].values
    return defensive_nfl_ids[is_player_to_predict].tolist()


def give_player_role(
    player_position: str, is_offense: bool, nfl_id: int, target_id: int
) -> str:
    if nfl_id == target_id:
        return "TargetedReceiver"
    if player_position == "QB":
        return "Passer"
    if is_offense:
        return "OtherRouteRunner"
    return "DefensiveCoverage"


def get_end_frame_id(tracking: pd.DataFrame, pass_result: str) -> int:
    events = tracking["event"].unique()
    invalid = 0
    if pass_result == "C":
        if "pass_outcome_caught" in events:
            frame_id = tracking.query("event == 'pass_outcome_caught'")[
                "frame_id"
            ].values[0]
            return int(frame_id)
        return invalid
    elif pass_result == "I":
        if "pass_outcome_incomplete" in events:
            frame_id = tracking.query("event == 'pass_outcome_incomplete'")[
                "frame_id"
            ].values[0]
            return int(frame_id)
        return invalid
    elif pass_result == "IN":
        if "pass_outcome_interception" in events:
            frame_id = tracking.query("event == 'pass_outcome_interception'")[
                "frame_id"
            ].values[0]
            return int(frame_id)
        return invalid
    return invalid


def get_frame_id_on_pass_received(tracking: pd.DataFrame, pass_result: str) -> int:
    events = tracking["event"].unique()
    invalid = 0
    if pass_result == "C":
        if "pass_arrived" in events:
            frame_id = tracking.query("event == 'pass_arrived'")["frame_id"].values[0]
            return int(frame_id)
        return invalid
    elif pass_result == "I":
        if "pass_arrived" in events:
            frame_id = tracking.query("event == 'pass_arrived'")["frame_id"].values[0]
            return int(frame_id)
        if "pass_outcome_incomplete" not in events:
            frame_id = tracking.query("event == 'pass_outcome_incomplete'")[
                "frame_id"
            ].values[0]
            return int(frame_id)
        return invalid
    elif pass_result == "IN":
        if "pass_arrived" in events:
            frame_id = tracking.query("event == 'pass_arrived'")["frame_id"].values[0]
            return int(frame_id)
        return invalid
    return invalid


def create_input_and_output_data(
    tracking: pd.DataFrame, play: pd.DataFrame, player: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # play dataframe contains only plays pf completed pass.
    ids = tracking["id"].unique()
    input_data = pd.DataFrame()
    output_data = pd.DataFrame()
    for i in tqdm(ids):
        if i not in play["id"].values:
            continue

        pass_result = play[play["id"] == i]["pass_result"].values[0]
        if pass_result not in ["C", "I", "IN"]:
            continue
        target_receiver_name = play[play["id"] == i]["target_receiver_name"].values[0]
        intercepted_name = play[play["id"] == i]["intercepted_name"].values[0]
        absolute_yardline_number = play[play["id"] == i][
            "absolute_yardline_number"
        ].values[0]
        _tracking = tracking[tracking["id"] == i]

        events = _tracking["event"].unique()
        if any(event not in events for event in ["ball_snap", "pass_forward"]):
            continue
        if any(event in events for event in ["pass_tipped"]):
            continue

        frame_id_on_ball_snap = _tracking.query("event == 'ball_snap'")[
            "frame_id"
        ].values[0]
        frame_id_on_pass_released = _tracking.query("event == 'pass_forward'")[
            "frame_id"
        ].values[0]
        if frame_id_on_pass_released - frame_id_on_ball_snap < 10:
            continue

        # get end frame_id
        frame_id_end = get_end_frame_id(_tracking, pass_result)
        if frame_id_end == 0:
            continue

        # get frame_id on pass receivered
        frame_id_on_pass_received = get_frame_id_on_pass_received(
            _tracking, pass_result
        )
        if frame_id_on_pass_received == 0:
            continue

        # clip
        max_frames_from_received_to_end = 2
        frame_id_end = min(
            frame_id_end, frame_id_on_pass_received + max_frames_from_received_to_end
        )

        num_frames_output = frame_id_end - frame_id_on_pass_released
        if num_frames_output < 5:
            continue
        _tracking = _tracking.drop(["event"], axis=1)

        play_direction = _tracking["play_direction"].values[0]
        ball_land_xy = get_ball_land_xy(
            _tracking,
            frame_id_on_pass_received,
            target_receiver_name=target_receiver_name,
            intercepted_name=intercepted_name,
            pass_result=pass_result,
            play_direction=play_direction,
        )
        if ball_land_xy.shape[0] != 1:
            continue

        target_receiver_id = identify_target_receiver_id(
            _tracking, target_receiver_name
        )
        if target_receiver_id == -1:
            continue
        target_defense_id = identify_defensive_players_to_predict(
            _tracking,
            frame_id_on_pass_released,
            target_receiver_name,
            frame_id_end - frame_id_on_pass_released,
            ball_land_xy,
        )
        player_to_predict = [target_receiver_id] + target_defense_id

        # create input dataframe
        input_df = _tracking.query("frame_id >= @frame_id_on_ball_snap").query(
            "frame_id <= @frame_id_on_pass_released"
        )
        input_df["frame_id"] = input_df["frame_id"] - frame_id_on_ball_snap
        input_df["player_to_predict"] = [
            nfl_id in player_to_predict for nfl_id in input_df["nfl_id"]
        ]
        input_df["player_role"] = [
            give_player_role(pos, is_o, i, target_receiver_id)
            for pos, is_o, i in zip(
                input_df["player_position"],
                input_df["is_offense"],
                input_df["nfl_id"],
                strict=False,
            )
        ]
        input_df["ball_land_x"] = ball_land_xy[:, 0].repeat(input_df.shape[0])
        input_df["ball_land_y"] = ball_land_xy[:, 1].repeat(input_df.shape[0])
        input_df["num_frames_output"] = [num_frames_output] * input_df.shape[0]
        input_df["absolute_yardline_number"] = [
            absolute_yardline_number
        ] * input_df.shape[0]
        input_df = pd.merge(input_df, player, "left", "nfl_id")

        # create output dataframe
        output_df = _tracking.query("frame_id > @frame_id_on_pass_released").query(
            "frame_id <= @frame_id_end"
        )
        output_df["frame_id"] = output_df["frame_id"] - frame_id_on_pass_released
        output_df = output_df.query("nfl_id in @player_to_predict")
        output_df = output_df[
            ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "id"]
        ]

        input_data = pd.concat([input_data, input_df])
        output_data = pd.concat([output_data, output_df])

    return input_data.reset_index(drop=True), output_data.reset_index(drop=True)


def preprocess_external_data(
    tracking_files: list[Path], play_file: Path, player_file: Path, save_dir: Path
) -> None:
    play = pd.read_csv(play_file)
    play = preprocess_play_data(play)
    play.to_csv(save_dir.joinpath("supplementary_data.csv"))

    player = pd.read_csv(player_file)
    player = preprocess_player_data(player)

    for tracking_file in tracking_files:
        tracking = pd.read_csv(tracking_file)
        tracking = preprocess_tracking_data(tracking)

        input_data, output_data = create_input_and_output_data(tracking, play, player)

        week = tracking_file.stem.replace("week", "")
        input_data.to_csv(save_dir.joinpath(f"input_2021_w{week.zfill(2)}.csv"))
        output_data.to_csv(save_dir.joinpath(f"output_2021_w{week.zfill(2)}.csv"))


def main() -> None:
    tracking_files = sorted(Path("data/2021/raw").glob("week*"))
    play_file = Path("data/2021/raw/plays.csv")
    player_file = Path("data/2021/raw/players.csv")
    save_dir = Path("data/2021/processed")
    save_dir.mkdir(parents=True, exist_ok=True)

    preprocess_external_data(tracking_files, play_file, player_file, save_dir)


if __name__ == "__main__":
    main()

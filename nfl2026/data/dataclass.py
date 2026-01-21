import random
from dataclasses import dataclass
from enum import IntEnum
from logging import warning
from typing import TypeAlias

import torch
from jaxtyping import Bool, Float, Int
from torch_geometric.data import HeteroData
from typing_extensions import Self

from ..geometry import wrap_angle

XMAX = 120.0
YMAX = 53.3


EdgeArray: TypeAlias = Int[torch.Tensor, "2 num_edges"]


class PlayDirection(IntEnum):
    left = 0
    right = 1


class PlayerRole(IntEnum):
    DefensiveCoverage = 0
    TargetedReceiver = 1
    OtherRouteRunner = 2
    Passer = 3


class PlayerPosition(IntEnum):
    FS = 0
    SS = 1
    S = 2
    CB = 3
    MLB = 4
    OLB = 5
    ILB = 6
    DE = 7
    NT = 8
    DT = 9
    T = 10
    WR = 11
    TE = 12
    RB = 13
    FB = 14
    QB = 15
    Other = 16


class PassResult(IntEnum):
    C = 0
    I = 1  # noqa: E741
    IN = 2


class PassCoverage(IntEnum):
    ZONE = 0
    MAN = 1
    UNKNOWN = 2


@dataclass
class Player:
    x: Float[torch.Tensor, "num_players num_timesteps"]
    y: Float[torch.Tensor, "num_players num_timesteps"]
    dir: Float[torch.Tensor, "num_players num_timesteps"]
    s: Float[torch.Tensor, "num_players num_timesteps"]
    a: Float[torch.Tensor, "num_players num_timesteps"]
    o: Float[torch.Tensor, "num_players num_timesteps"]
    valid: Bool[torch.Tensor, "num_players num_timesteps"]
    nfl_id: Int[torch.Tensor, "num_players"]
    player_role: Int[torch.Tensor, "num_players"]
    player_position: Int[torch.Tensor, "num_players"]
    player_to_predict: Bool[torch.Tensor, "num_players"]
    player_height: Float[torch.Tensor, "num_players"]
    player_weight: Float[torch.Tensor, "num_players"]

    @property
    def num_players(self) -> int:
        return self.x.shape[0]

    @property
    def num_timesteps(self) -> int:
        return self.x.shape[1]

    @property
    def xy(self) -> Float[torch.Tensor, "num_players num_timesteps 2"]:
        return torch.stack([self.x, self.y], dim=-1)

    @property
    def xyh(self) -> Float[torch.Tensor, "num_players num_timesteps 3"]:
        return torch.stack([self.x, self.y, self.dir], dim=-1)

    @property
    def xyhso(self) -> Float[torch.Tensor, "num_players num_timesteps 5"]:
        return torch.stack([self.x, self.y, self.dir, self.s, self.o], dim=-1)

    @property
    def orient_velocity(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        d = wrap_angle(self.o[:, 1:] - self.o[:, :-1])
        return torch.cat([d[:, 0:1], d], dim=1)

    @property
    def orient_acceleration(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        d = wrap_angle(self.orient_velocity[:, 1:] - self.orient_velocity[:, :-1])
        return torch.cat([d[:, 0:1], d], dim=1)

    @property
    def angular_velocity(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        d = wrap_angle(self.dir[:, 1:] - self.dir[:, :-1])
        return torch.cat([d[:, 0:1], d], dim=1)

    @property
    def angular_acceleration(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        d = wrap_angle(self.angular_velocity[:, 1:] - self.angular_velocity[:, :-1])
        return torch.cat([d[:, 0:1], d], dim=1)

    @property
    def dxdydh(self) -> Float[torch.Tensor, "num_players num_timesteps 3"]:
        dxdydh = self.xyh[:, 1:] - self.xyh[:, :-1]
        return torch.cat([dxdydh[:, 0:1], dxdydh], dim=1)

    @property
    def velocity_x(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        return self.s * torch.cos(self.dir)

    @property
    def velocity_y(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        return self.s * torch.sin(self.dir)

    @property
    def velocity_xy(self) -> Float[torch.Tensor, "num_players num_timesteps 2"]:
        return torch.stack([self.velocity_x, self.velocity_y], dim=-1)

    @property
    def acceleration_x(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        return self.a * torch.cos(self.dir)

    @property
    def acceleration_y(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        return self.a * torch.sin(self.dir)

    @property
    def acceleration_xy(self) -> Float[torch.Tensor, "num_players num_timesteps 2"]:
        return torch.stack([self.acceleration_x, self.acceleration_y], dim=-1)

    @property
    def momentum(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        mass_kg = self.player_weight / 2.20462
        return self.s * mass_kg[:, None]

    @property
    def kinetic_energy(self) -> Float[torch.Tensor, "num_players num_timesteps"]:
        mass_kg = self.player_weight / 2.20462
        return 0.5 * mass_kg[:, None] * self.s**2

    @property
    def ball_thrown_xy(self) -> Float[torch.Tensor, "1 2"]:
        is_passer = self.player_role.eq(PlayerRole.Passer.value)
        return self.xy[is_passer, -1]

    @property
    def to_predict(self) -> "Player":
        return Player(
            self.x[self.player_to_predict],
            self.y[self.player_to_predict],
            self.dir[self.player_to_predict],
            self.s[self.player_to_predict],
            self.a[self.player_to_predict],
            self.o[self.player_to_predict],
            self.valid[self.player_to_predict],
            self.nfl_id[self.player_to_predict],
            self.player_role[self.player_to_predict],
            self.player_position[self.player_to_predict],
            self.player_to_predict[self.player_to_predict],
            self.player_height[self.player_to_predict],
            self.player_weight[self.player_to_predict],
        )

    @property
    def except_QB(self) -> "Player":
        is_QB = self.player_role.eq(PlayerRole.Passer.value)
        return Player(
            self.x[~is_QB],
            self.y[~is_QB],
            self.dir[~is_QB],
            self.s[~is_QB],
            self.a[~is_QB],
            self.o[~is_QB],
            self.valid[~is_QB],
            self.nfl_id[~is_QB],
            self.player_role[~is_QB],
            self.player_position[~is_QB],
            self.player_to_predict[~is_QB],
            self.player_height[~is_QB],
            self.player_weight[~is_QB],
        )

    @property
    def QB(self) -> "Player":
        is_QB = self.player_role.eq(PlayerRole.Passer.value)
        return Player(
            self.x[is_QB],
            self.y[is_QB],
            self.dir[is_QB],
            self.s[is_QB],
            self.a[is_QB],
            self.o[is_QB],
            self.valid[is_QB],
            self.nfl_id[is_QB],
            self.player_role[is_QB],
            self.player_position[is_QB],
            self.player_to_predict[is_QB],
            self.player_height[is_QB],
            self.player_weight[is_QB],
        )

    def __getitem__(self, idx: int) -> "Player":
        if idx > self.num_players:
            raise ValueError("Invalid index")
        return Player(
            self.x[idx : idx + 1],
            self.y[idx : idx + 1],
            self.dir[idx : idx + 1],
            self.s[idx : idx + 1],
            self.a[idx : idx + 1],
            self.o[idx : idx + 1],
            self.valid[idx : idx + 1],
            self.nfl_id[idx : idx + 1],
            self.player_role[idx : idx + 1],
            self.player_position[idx : idx + 1],
            self.player_to_predict[idx : idx + 1],
            self.player_height[idx : idx + 1],
            self.player_weight[idx : idx + 1],
        )


@dataclass
class Target:
    x: Float[torch.Tensor, "num_players num_timesteps"]
    y: Float[torch.Tensor, "num_players num_timesteps"]
    nfl_id: Int[torch.Tensor, "num_players"]

    @property
    def num_players(self) -> int:
        return self.x.shape[0]

    @property
    def num_timesteps(self) -> int:
        return self.x.shape[1]

    @property
    def xy(self) -> Float[torch.Tensor, "num_players num_timesteps 2"]:
        return torch.stack([self.x, self.y], dim=-1)


@dataclass
class Polyline:
    xyh: Float[torch.Tensor, "num_polylines XYH"]
    type: Int[torch.Tensor, "num_polylines"]

    @property
    def num_polylines(self) -> int:
        return self.xyh.shape[0]


@dataclass
class MetaData:
    game_id: Int[torch.Tensor, "1"]
    play_id: Int[torch.Tensor, "1"]
    num_frames_output: Int[torch.Tensor, "1"]
    play_direction: Int[torch.Tensor, "1"]
    ball_land_x: Float[torch.Tensor, "1"]
    ball_land_y: Float[torch.Tensor, "1"]
    scrimage_line_x: Float[torch.Tensor, "1"]

    @property
    def ball_land_xy(self) -> Float[torch.Tensor, "1 2"]:
        return torch.stack([self.ball_land_x, self.ball_land_y], dim=-1)


@dataclass
class PassAttribute:
    result: Int[torch.Tensor, "1"]
    coverage: Int[torch.Tensor, "1"]


def _build_ball_feature(player: Player, metadata: MetaData) -> dict[str, torch.Tensor]:
    ball_land_xy = metadata.ball_land_xy
    ball_land_h = torch.atan2(ball_land_xy[..., 1], ball_land_xy[..., 0])
    ball_land_h = -(ball_land_h - torch.pi / 2.0)
    ball_land_xyh = torch.cat([ball_land_xy, ball_land_h.unsqueeze(-1)], dim=-1)

    ball_thrown_xy = player.ball_thrown_xy
    if ball_thrown_xy.shape[0] != 1:
        ball_thrown_xy = torch.tensor([[metadata.scrimage_line_x[0], YMAX / 2]])

    ball_speed = torch.norm(ball_land_xy - ball_thrown_xy, p=2, dim=-1) / (
        metadata.num_frames_output * 0.1
    )

    is_out_of_bounds = (
        (metadata.ball_land_x < 0.0)
        | (metadata.ball_land_x > XMAX)
        | (metadata.ball_land_y < 0.0)
        | (metadata.ball_land_x < YMAX)
    )

    is_target_receiver = player.player_role.eq(PlayerRole.TargetedReceiver.value)
    if is_target_receiver.sum() != 1:
        pass

    wr_xy = player.xy[is_target_receiver, -1]
    wr_vx = player.velocity_x[is_target_receiver, -1]
    wr_vy = player.velocity_y[is_target_receiver, -1]

    ball_diff = ball_land_xy - wr_xy
    distance = torch.norm(ball_diff, dim=-1)
    angle_to_ball = torch.atan2(ball_diff[..., 1], ball_diff[..., 0])

    closing_speed = wr_vx * angle_to_ball.cos() + wr_vy * angle_to_ball.sin()

    closing_speed = torch.clamp(closing_speed, min=8.5)

    distance_run_on_ball_air = closing_speed * metadata.num_frames_output * 0.1

    is_reachable = distance < distance_run_on_ball_air

    ball_features = torch.stack(
        [ball_speed, is_out_of_bounds.float(), is_reachable.float()], dim=-1
    )
    return {"xyh": ball_land_xyh.float(), "feature": ball_features}


def _flip_xyh(xyh: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3"]:
    return torch.stack([xyh[..., 0], YMAX - xyh[..., 1], -xyh[..., 2]], dim=-1)


def _flip_relative_xy(
    xy: Float[torch.Tensor, "... 2"],
) -> Float[torch.Tensor, "... 2"]:
    return torch.stack([xy[..., 0], -xy[..., 1]], dim=-1)


def diff_angle(angle: Float[torch.Tensor, "... T"]) -> Float[torch.Tensor, "... T"]:
    d = wrap_angle(angle[..., 1:] - angle[..., :-1])
    return torch.cat([d[..., 0:1], d], dim=1)


def _velocity_features(
    s: Float[torch.Tensor, "A T"],
    a: Float[torch.Tensor, "A T"],
    h: Float[torch.Tensor, "A T"],
) -> Float[torch.Tensor, "A T D=4"]:
    features = torch.stack([s * h.cos(), a * h.cos(), s * h.sin(), a * h.sin()], dim=-1)
    return features.float()


def _angle_features(
    h: Float[torch.Tensor, "A T"],
    o: Float[torch.Tensor, "A T"],
) -> Float[torch.Tensor, "A T D=6"]:
    diff_o_h = wrap_angle(h - o)
    diff_o = diff_angle(o)
    diff_h = diff_angle(h)
    features = torch.stack(
        [
            diff_o_h.cos(),
            diff_o.cos(),
            diff_h.cos(),
            diff_o_h.sin(),
            diff_o.sin(),
            diff_h.sin(),
        ],
        dim=-1,
    )
    return features.float()


def _kinematic_features(
    s: Float[torch.Tensor, "A T"],
    height: Float[torch.Tensor, "A"],
    weight: Float[torch.Tensor, "A"],
) -> Float[torch.Tensor, "A T D=4"]:
    mass_kg = weight / 2.20462
    features = torch.stack(
        [
            height[:, None].repeat(1, s.shape[1]),
            weight[:, None].repeat(1, s.shape[1]),
            s * mass_kg[:, None],
            0.5 * mass_kg[:, None] * s**2,
        ],
        dim=-1,
    )
    return features.float()


class Prompt(HeteroData):
    @classmethod
    def create(
        cls,
        player: Player,
        polyline: Polyline,
        metadata: MetaData,
        target: Target | None = None,
        pass_attribute: PassAttribute | None = None,
    ) -> Self:
        player_except_qb = player.except_QB

        instance = cls()
        instance["num_players_tensor"] = torch.tensor(
            [player_except_qb.num_players], dtype=torch.int32
        )
        instance["num_polylines_tensor"] = torch.tensor(
            [polyline.num_polylines], dtype=torch.int32
        )
        instance["num_plays_tensor"] = torch.tensor([1], dtype=torch.int32)
        instance["num_frames_output_tensor"] = metadata.num_frames_output

        # players except QB
        p = instance["player"]
        p.num_nodes = player_except_qb.num_players
        p["x"] = player_except_qb.x.float()
        p["y"] = player_except_qb.y.float()
        p["o"] = player_except_qb.o.float()
        p["dir"] = player_except_qb.dir.float()
        p["s"] = player_except_qb.s.float()
        p["a"] = player_except_qb.a.float()
        p["height"] = player_except_qb.player_height.float()
        p["weight"] = player_except_qb.player_weight.float()

        p["id"] = player_except_qb.nfl_id[:, None]
        p["role"] = (
            player_except_qb.player_role[:, None]
            .repeat(1, player_except_qb.num_timesteps)
            .long()
        )
        p["position"] = (
            player_except_qb.player_position[:, None]
            .repeat(1, player_except_qb.num_timesteps)
            .long()
        )
        p["is_predicted"] = player_except_qb.player_to_predict.bool()
        p["predict"] = (
            player_except_qb.player_to_predict[:, None]
            .repeat(1, player_except_qb.num_timesteps)
            .long()
        )
        p["valid"] = player_except_qb.valid
        p["num_frames_output"] = metadata.num_frames_output.expand(
            player_except_qb.num_players
        )

        # QB
        qb = player.QB
        if qb.num_players != 1:
            warning("One play must have one quaterback")
            qb = player[0]

        q = instance["qb"]
        q.num_nodes = qb.num_players
        q["x"] = qb.x.float()
        q["y"] = qb.y.float()
        q["o"] = qb.o.float()
        q["dir"] = qb.dir.float()
        q["s"] = qb.s.float()
        q["a"] = qb.a.float()
        q["height"] = qb.player_height.float()
        q["weight"] = qb.player_weight.float()
        q["valid"] = qb.valid

        # pass attribute
        if pass_attribute is not None:
            res = instance["pass"]
            res.num_nodes = 1
            res["result"] = pass_attribute.result
            res["coverage"] = pass_attribute.coverage

        if target is not None:
            if not torch.all(
                player_except_qb.nfl_id[player_except_qb.player_to_predict]
                == target.nfl_id
            ):
                raise ValueError(
                    "The order of nfl_id is not same between player and target."
                )
            t = instance["target"]
            t.num_nodes = target.num_players

            last_xy = player_except_qb.xy[
                player_except_qb.player_to_predict, -1:
            ].repeat(1, target.xy.shape[1], 1)

            target_xy_local = target.xy - last_xy
            max_timesteps_to_predict = 50
            num_timeseteps_to_predict = target_xy_local.shape[1]
            if num_timeseteps_to_predict < max_timesteps_to_predict:
                target_xy_local = torch.cat(
                    [
                        target_xy_local,
                        target_xy_local[:, -1:].repeat(
                            1, max_timesteps_to_predict - num_timeseteps_to_predict, 1
                        ),
                    ],
                    dim=1,
                )
            else:
                target_xy_local = target_xy_local[:, :max_timesteps_to_predict]

            t["xy"] = target_xy_local.float()  # (A, 50, 2)
            mask = torch.zeros((target.num_players, max_timesteps_to_predict)).bool()
            mask[:, : metadata.num_frames_output[0]] = True
            t["mask"] = mask

        poly = instance["polyline"]
        poly.num_nodes = polyline.num_polylines
        poly["xyh"] = polyline.xyh.float()
        poly["token_id"] = polyline.type.long()

        m = instance["metadata"]
        m.num_nodes = 1
        m["game_id"] = metadata.game_id
        m["play_id"] = metadata.play_id
        m["num_frames_output"] = metadata.num_frames_output
        m["play_direction"] = metadata.play_direction

        b = instance["ball"]
        b.num_nodes = 1
        ball_features = _build_ball_feature(player, metadata)
        b["xyh"] = ball_features["xyh"].float()
        b["feature"] = ball_features["feature"]

        return instance

    @property
    def num_plays(self) -> list[int]:
        return self["num_plays_tensor"].tolist()

    @property
    def num_players(self) -> list[int]:
        return self["num_players_tensor"].tolist()

    @property
    def num_polylines(self) -> list[int]:
        return self["num_polylines_tensor"].tolist()

    @property
    def num_timesteps(self) -> int:
        return self["player"]["x"].shape[1]

    @property
    def num_frames_output(self) -> list[int]:
        return self["num_frames_output_tensor"].tolist()

    @property
    def basic_time_dependent_attributes(self) -> list[str]:
        return ["x", "y", "o", "dir", "s", "a", "valid"]

    @property
    def player_time_dependent_attributes(self) -> list[str]:
        return ["role", "position", "predict"]

    @property
    def time_dependent_attributes(self) -> list[str]:
        return self.basic_time_dependent_attributes + [
            "xyh",
            "dxdydh",
            "speeds",
            "angles",
            "kinematics",
        ]

    @staticmethod
    def with_extra_features(prompt: "Prompt") -> "Prompt":
        p = prompt["player"]
        xyh = torch.stack([p["x"], p["y"], p["dir"]], dim=-1)
        dxdydh = xyh[:, 1:] - xyh[:, :-1]
        dxdydh = torch.cat([dxdydh[:, 0:1], dxdydh], dim=1)
        p["dxdydh"] = dxdydh
        p["xyh"] = torch.stack([p["x"], p["y"], p["dir"], p["s"], p["o"]], dim=-1)
        p["speeds"] = _velocity_features(p["s"], p["a"], p["dir"])
        p["angles"] = _angle_features(p["dir"], p["o"])
        p["kinematics"] = _kinematic_features(p["s"], p["height"], p["weight"])

        q = prompt["qb"]
        xyh = torch.stack([q["x"], q["y"], q["dir"]], dim=-1)
        dxdydh = xyh[:, 1:] - xyh[:, :-1]
        dxdydh = torch.cat([dxdydh[:, 0:1], dxdydh], dim=1)
        q["dxdydh"] = dxdydh
        q["xyh"] = torch.stack([q["x"], q["y"], q["dir"], q["s"], q["o"]], dim=-1)
        q["speeds"] = _velocity_features(q["s"], q["a"], q["dir"])
        q["angles"] = _angle_features(q["dir"], q["o"])
        q["kinematics"] = _kinematic_features(q["s"], q["height"], q["weight"])

        return prompt

    def get(self, num_input_timesteps: int = 10) -> "Prompt":
        prompt: Prompt = self.clone()
        prompt = self.with_extra_features(prompt)
        for attr in (
            self.time_dependent_attributes + self.player_time_dependent_attributes
        ):
            prompt["player"][attr] = prompt["player"][attr][
                :, self.num_timesteps - num_input_timesteps :
            ]
        for attr in self.time_dependent_attributes:
            prompt["qb"][attr] = prompt["qb"][attr][
                :, self.num_timesteps - num_input_timesteps :
            ]
        return prompt

    @property
    def flip_along_x(self) -> "Prompt":
        prompt: Prompt = self.clone()
        for node in ["player", "qb"]:
            prompt[node]["y"] = YMAX - prompt[node]["y"]
            prompt[node]["dir"] = -prompt[node]["dir"]
            prompt[node]["o"] = -prompt[node]["o"]

        if "target" in prompt.node_types:
            prompt["target"]["xy"] = _flip_relative_xy(prompt["target"]["xy"])
        prompt["ball"]["xyh"] = _flip_xyh(prompt["ball"]["xyh"])
        return prompt

    @property
    def temporal_shift(self) -> "Prompt":
        prompt: Prompt = self.clone()

        is_target = prompt["player"]["is_predicted"]
        xy = torch.stack(
            [prompt["player"]["x"][is_target], prompt["player"]["y"][is_target]], dim=-1
        )
        target_xy_global = prompt["target"]["xy"] + xy[:, -1:]
        target_xy_global = torch.cat([xy[:, -1:], target_xy_global[:, :-1]], dim=1)
        prompt["target"]["xy"] = target_xy_global - xy[:, -2:-1]
        prompt["player"]["num_frames_output"] += 1
        prompt["metadata"]["num_frames_output"] += 1

        for attr in (
            self.basic_time_dependent_attributes + self.player_time_dependent_attributes
        ):
            new_attr = prompt["player"][attr][:, : self.num_timesteps - 1]
            prompt["player"][attr] = torch.cat([new_attr[:, :1], new_attr], dim=1)

        for attr in self.basic_time_dependent_attributes:
            new_attr = prompt["qb"][attr][:, : self.num_timesteps - 1]
            prompt["qb"][attr] = torch.cat([new_attr[:, :1], new_attr], dim=1)

        return prompt

    @property
    def shift_scrimage_line(self) -> "Prompt":
        prompt: Prompt = self.clone()

        player, qb, polyline, ball, target = (
            prompt["player"],
            prompt["qb"],
            prompt["polyline"],
            prompt["ball"],
            prompt["target"],
        )

        target_x = player["x"][player["is_predicted"], -1:] + target["xy"][..., 0]
        for batch_idx in range(ball.num_nodes):
            is_player = player.batch.eq(batch_idx)
            is_qb = qb.batch.eq(batch_idx)
            is_poly = polyline.batch.eq(batch_idx)
            is_target = target.batch.eq(batch_idx)

            x_min, x_max = (
                player["x"][is_player].min().item(),
                player["x"][is_player].max().item(),
            )
            max_shift = min(x_min, XMAX - x_max)
            target_x_min, target_x_max = (
                target_x[is_target].min().item(),
                target_x[is_target].max().item(),
            )
            max_shift = min(max_shift, min(target_x_min, XMAX - target_x_max))
            ball_x = ball["xyh"][batch_idx, 0].item()
            max_shift = min(max_shift, min(ball_x, XMAX - ball_x))
            shift = int((random.random() * 2 - 1) * max_shift)

            polyline["xyh"][is_poly][88:98, 0] += shift
            player["x"][is_player] += shift
            qb["x"][is_qb] += shift
            ball["xyh"][batch_idx, 0] += shift

        return prompt


Prompts: TypeAlias = list[Prompt]

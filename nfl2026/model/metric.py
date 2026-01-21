import torch
from jaxtyping import Float
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from ..data.dataclass import PassResult


class RMSEMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(
        self,
        preds: Float[torch.Tensor, "A T XY"],
        target: Float[torch.Tensor, "A T XY"],
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        self.preds.append(preds.flatten(0, 1))
        self.target.append(target.flatten(0, 1))

    def compute(self) -> Float[torch.Tensor, ""]:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        mse = torch.mean((target - preds) ** 2)
        return torch.sqrt(mse)


class RMSEMetricForEachResult(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("preds_c", default=[], dist_reduce_fx="cat")
        self.add_state("target_c", default=[], dist_reduce_fx="cat")
        self.add_state("preds_i", default=[], dist_reduce_fx="cat")
        self.add_state("target_i", default=[], dist_reduce_fx="cat")
        self.add_state("preds_in", default=[], dist_reduce_fx="cat")
        self.add_state("target_in", default=[], dist_reduce_fx="cat")

    def update(
        self,
        preds: Float[torch.Tensor, "A T XY"],
        target: Float[torch.Tensor, "A T XY"],
        pass_result: int,
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        preds = preds.flatten(0, 1)
        target = target.flatten(0, 1)
        if pass_result == PassResult.C.value:
            self.preds_c.append(preds)
            self.target_c.append(target)
        elif pass_result == PassResult.I:
            self.preds_i.append(preds)
            self.target_i.append(target)
        elif pass_result == PassResult.IN:
            self.preds_in.append(preds)
            self.target_in.append(target)
        else:
            raise ValueError(f"Invalid pass result: {pass_result}")

    def compute(self) -> dict[str, Float[torch.Tensor, ""]]:
        rmse_c = torch.tensor(0.0)
        rmse_i = torch.tensor(0.0)
        rmse_in = torch.tensor(0.0)

        preds_list = []
        target_list = []
        if len(self.preds_c) > 0:
            preds_c = dim_zero_cat(self.preds_c)
            target_c = dim_zero_cat(self.target_c)
            rmse_c = torch.sqrt(torch.mean((target_c - preds_c) ** 2))
            preds_list.append(preds_c)
            target_list.append(target_c)

        if len(self.preds_i) > 0:
            preds_i = dim_zero_cat(self.preds_i)
            target_i = dim_zero_cat(self.target_i)
            rmse_i = torch.sqrt(torch.mean((target_i - preds_i) ** 2))
            preds_list.append(preds_i)
            target_list.append(target_i)

        if len(self.preds_in) > 0:
            preds_in = dim_zero_cat(self.preds_in)
            target_in = dim_zero_cat(self.target_in)
            rmse_in = torch.sqrt(torch.mean((target_in - preds_in) ** 2))
            preds_list.append(preds_in)
            target_list.append(target_in)

        preds = torch.cat(preds_list, dim=0)
        target = torch.cat(target_list, dim=0)
        rmse = torch.sqrt(torch.mean((target - preds) ** 2))
        return {"rmse": rmse, "rmse_c": rmse_c, "rmse_i": rmse_i, "rmse_in": rmse_in}

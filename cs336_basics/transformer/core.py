import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self, in_features: Int, out_features: Int, device: str = None, dtype: str = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight: Float[Tensor, "out_features in_features"] = nn.Parameter(weight)
        sigma: float = 2.0 / (in_features + out_features)
        # sigma: float = 1.0 / (in_features**0.5)
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(
        self, x: Float[Tensor, "batch ... in_features"]
    ) -> Float[Tensor, "batch ... out_features"]:
        out: Float[Tensor, "batch ... out_features"] = einsum(
            x,
            self.weight,
            "batch ... in_features, out_features in_features -> batch ... out_features",
        )
        return out

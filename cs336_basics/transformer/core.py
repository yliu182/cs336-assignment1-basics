import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor


# uv run pytest -k test_linear
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


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: Int,  # size of vocabulary
        embedding_dim: Int,
        device: str = None,
        dtype: str = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight: Float[Tensor, "num_embeddings embedding_dim"] = nn.Parameter(
            weight
        )
        sigma: float = 1.0 / (embedding_dim**0.5)
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(
        self, x: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len embedding_dim"]:
        return torch.index_select(self.weight, dim=0, index=x.reshape(-1)).view(
            *x.size(), -1
        )


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: Int,
        eps: float = 1e-5,
        gains: Float[Tensor, " d_model"] = None,
        device: str = None,
        dtype: str = None,
    ):
        super().__init__()
        if gains is None:
            gains = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(gains)
        self.eps = eps

    def forward(
        self, x: Float[Tensor, "batch_size sequence_length d_model"]
    ) -> Float[Tensor, "batch_size sequence_length d_model"]:

        in_dtype = x.dtype
        x = x.to(torch.float32)
        with torch.autocast(device_type="cuda", enabled=False):
            reverse_rms = torch.rsqrt((x * x).mean(-1) + self.eps).unsqueeze(-1)
            out: Float[Tensor, "batch_size sequence_length d_model"] = (
                x * reverse_rms * self.gain
            )
        return out.to()

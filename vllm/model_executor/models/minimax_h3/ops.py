# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical reference operations for H3's FP16/FP32 execution contract."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        if residual is not None:
            residual = residual + x
            x = residual
        value = x.float()
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        output = (value * self.weight.float()).to(x.dtype)
        return output if residual is None else (output, residual)


class RotaryEmbedding(nn.Module):
    def __init__(self, *, is_neox_style=True, half_head_dim=False):
        super().__init__()
        if not is_neox_style or half_head_dim:
            raise ValueError("H3 uses non-interleaved full-width rotary tables")

    def forward(self, x, cos, sin):
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return x * cos.unsqueeze(-2) + rotated * sin.unsqueeze(-2)


def fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps):
    if (
        q.is_cuda
        and not torch.is_grad_enabled()
        and q.dtype == k.dtype == rope_table.dtype == torch.float16
        and q.ndim == k.ndim == 3
        and q.shape[-1] == k.shape[-1] == 128
        and q.shape[0] == k.shape[0]
        and rope_table.shape == (q.shape[0], 96)
        and q_weight.shape == k_weight.shape == (128,)
        and all(t.device == q.device for t in (k, q_weight, k_weight, rope_table))
        and all(t.stride(-1) == 1 for t in (q, k, q_weight, k_weight, rope_table))
    ):
        from .qk_norm_rope import qk_norm_rope

        return qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps)
    return qk_norm_rope_reference(q, k, q_weight, k_weight, rope_table, eps)


def qk_norm_rope_reference(q, k, q_weight, k_weight, rope_table, eps):
    # This reference retains the normalized FP16 rounding boundary. The CUDA
    # implementation must preserve it, including partial (96/128) rotation.
    def apply(x, weight):
        value = x.float()
        value = (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
            * weight.float()
        ).to(x.dtype)
        half = rope_table.shape[-1] // 2
        cos, sin = rope_table.to(x.dtype).unsqueeze(1).chunk(2, dim=-1)
        first, second = value[..., :half], value[..., half : 2 * half]
        return torch.cat(
            (
                first * cos - second * sin,
                second * cos + first * sin,
                value[..., 2 * half :],
            ),
            dim=-1,
        )

    return apply(q, q_weight), apply(k, k_weight)


def convrot_reference(x, group_size=256):
    """Regular Hadamard (Kronecker power of H4), accumulated in FP32.

    The checkpoint holds W @ H.T. H is symmetric and orthonormal, so the
    activation must receive x @ H before multiplication by the stored W.
    """
    if group_size != 256 or x.shape[-1] % group_size:
        raise ValueError("H3 ConvRot requires complete 256-channel groups")
    value = x.float().reshape(-1, group_size)
    stride = 1
    while stride < group_size:
        parts = value.reshape(-1, group_size // (4 * stride), 4, stride)
        a, b, c, d = parts.unbind(-2)
        value = torch.stack(
            (a + b + c - d, a + b - c + d, a - b + c + d, -a + b + c + d), -2
        )
        stride *= 4
    return (value.reshape(x.shape) / 16).to(x.dtype)


def dequantize_int8_reference(weight, scale):
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise ValueError("W8A16 requires a signed INT8 matrix")
    if scale.dtype != torch.float32 or scale.numel() != weight.shape[0]:
        raise ValueError("W8A16 requires one FP32 scale per output row")
    return (weight.float() * scale.reshape(-1, 1)).to(torch.float16)

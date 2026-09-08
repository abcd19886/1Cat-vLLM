# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H3 D128 RMSNorm and partial RoPE with explicit FP16 rounding boundaries."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _qk_norm_rope_kernel(
    x_ptr,
    weight_ptr,
    rope_ptr,
    output_ptr,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    token_stride: tl.constexpr,
    head_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    columns = tl.arange(0, 128)
    token, head = rows // heads, rows % heads
    offsets = token[:, None] * token_stride + head[:, None] * head_stride
    valid = rows[:, None] < tokens * heads
    x = tl.load(x_ptr + offsets + columns[None, :], valid, 0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(x * x, 1) / 128 + eps)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    normalized = ((x * inverse_rms[:, None]) * weight[None, :]).to(tl.float16)

    # H3 rotates two 48-channel halves and leaves the final 32 channels intact.
    partner = tl.where(
        columns < 48, columns + 48, tl.where(columns < 96, columns - 48, columns)
    )
    paired = tl.load(x_ptr + offsets + partner[None, :], valid, 0).to(tl.float32)
    paired_weight = tl.load(weight_ptr + partner).to(tl.float32)
    paired = ((paired * inverse_rms[:, None]) * paired_weight[None, :]).to(tl.float16)
    rope_column = tl.where(columns < 48, columns, columns - 48)
    rope_offsets = token[:, None] * rope_stride + rope_column[None, :]
    rotated_mask = valid & (columns[None, :] < 96)
    cosine = tl.load(rope_ptr + rope_offsets, rotated_mask, 0).to(tl.float32)
    sine = tl.load(rope_ptr + rope_offsets + 48, rotated_mask, 0).to(tl.float32)

    # The reference rounds normalization and each product to FP16 before
    # addition/subtraction. Fusing these into an FMA changes model output.
    first = (normalized.to(tl.float32) * cosine).to(tl.float16).to(tl.float32)
    second = (paired.to(tl.float32) * sine).to(tl.float16).to(tl.float32)
    rotated = tl.where(columns[None, :] < 48, first - second, first + second).to(
        tl.float16
    )
    output = tl.where(columns[None, :] < 96, rotated, normalized)
    tl.store(output_ptr + rows[:, None] * 128 + columns[None, :], output, valid)


def qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps):
    outputs = []
    for x, weight in ((q, q_weight), (k, k_weight)):
        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        if x.numel():
            _qk_norm_rope_kernel[(triton.cdiv(x.shape[0] * x.shape[1], 4),)](
                x,
                weight,
                rope_table,
                output,
                *x.shape[:2],
                *x.stride()[:2],
                rope_table.stride(0),
                eps,
                4,
                num_warps=4,
                enable_fp_fusion=False,
            )
        outputs.append(output)
    return tuple(outputs)

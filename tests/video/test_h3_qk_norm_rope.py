# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.minimax_h3.ops import (
    fused_qk_norm_rope,
    qk_norm_rope_reference,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def inputs(tokens, heads, amplitude, weight_dtype):
    torch.manual_seed(42)
    # Split projection views have a non-contiguous token stride and offsets.
    storage = torch.randn(tokens, 3, heads, 128, device="cuda", dtype=torch.float16)
    storage *= amplitude
    q, k, _ = storage.unbind(1)
    q[0].zero_()
    weights = torch.randn(2, 128, device="cuda", dtype=weight_dtype)
    angles = torch.randn(tokens, 48, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), -1).half()
    return q, k, *weights, rope, 1e-5


@pytest.mark.parametrize("tokens,heads", [(1, 1), (65, 3), (129, 14), (12323, 14)])
@pytest.mark.parametrize(
    "amplitude,weight_dtype", [(1, torch.float16), (2000, torch.float32)]
)
@torch.inference_mode()
def test_rounding_and_partial_rotation(tokens, heads, amplitude, weight_dtype):
    args = inputs(tokens, heads, amplitude, weight_dtype)
    expected = qk_norm_rope_reference(*args)
    actual = fused_qk_norm_rope(*args)
    for result, reference in zip(actual, expected):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@torch.inference_mode()
def test_graph_replay_reads_changed_inputs_and_rope():
    args = inputs(65, 3, 1000, torch.float16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_qk_norm_rope(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_qk_norm_rope(*args)
    for _ in range(2):
        args[0].normal_()
        args[1].normal_()
        args[4].neg_()
        graph.replay()
        expected = qk_norm_rope_reference(*args)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


@torch.inference_mode()
def test_other_rotary_width_keeps_reference_path():
    args = list(inputs(65, 3, 1, torch.float16))
    args[4] = args[4][:, :64]
    actual = fused_qk_norm_rope(*args)
    expected = qk_norm_rope_reference(*args)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)

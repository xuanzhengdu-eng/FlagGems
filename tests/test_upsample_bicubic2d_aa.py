# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

random.seed(time.time() // 100)


if QUICK_MODE:
    ALIGN_CORNERS_FWD = [False]
    SCALES = [(2, 2)]
    SHAPES_FWD = [(1, 2, 8, 9)]
    FLOAT_DTYPES = [torch.float16]
    PARAMS_BWD = [(1, 3, 16, 16, 8, 8, False)]
else:
    ALIGN_CORNERS_FWD = [False, True]
    SCALES = [(2, 2), (0.5, 0.75)]
    SHAPES_FWD = [
        (1, 1, 3, 5),
        (2, 3, 17, 19),
        (4, 8, 32, 48),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    PARAMS_BWD = [
        (1, 3, 16, 16, 8, 8, False),
        (2, 4, 8, 8, 16, 16, False),
        (1, 3, 32, 32, 10, 10, False),
        # (1, 1, 10, 10, 23, 23, False),  # commented out to reduce CI timeout
        (1, 3, 16, 16, 8, 8, True),
        (1, 3, 8, 8, 16, 16, True),
        (2, 64, 32, 32, 16, 16, False),
        # (1, 3, 7, 11, 13, 5, False),  # commented out to reduce CI timeout
        # (1, 1, 4, 4, 4, 4, False),  # commented out to reduce CI timeout
        # (1, 1, 8, 8, 1, 1, True),  # commented out to reduce CI timeout
        # Extra cases
        (1, 1, 64, 64, 16, 16, False),
        # (1, 1, 64, 64, 128, 128, False),  # commented out to reduce CI timeout
        # (512, 1024, 32, 32, 8, 8, False),  # commented out to reduce CI timeout - very large batch
        # (256, 512, 64, 64, 16, 16, False),  # commented out to reduce CI timeout - very large batch
        # (4, 16, 16, 16, 4, 4, False),  # commented out to reduce CI timeout
        # (4, 16, 4, 4, 16, 16, False),  # commented out to reduce CI timeout
        (4, 16, 64, 128, 32, 64, False),
        # (4, 16, 64, 128, 128, 256, True),  # commented out to reduce CI timeout
        # (1, 1, 4096, 4096, 1024, 1024, False),  # commented out to reduce CI timeout - extreme large
    ]


@pytest.mark.upsample_bicubic2d_aa
@pytest.mark.parametrize("align_corners", ALIGN_CORNERS_FWD)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("shape", SHAPES_FWD)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_upsample_bicubic2d_aa(dtype, shape, scale, align_corners):
    input = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input, True)
    output_size = tuple([int(input.shape[i + 2] * scale[i]) for i in range(2)])
    ref_out = torch._C._nn._upsample_bicubic2d_aa(
        ref_i, output_size=output_size, align_corners=align_corners
    )
    with flag_gems.use_gems():
        res_out = torch._C._nn._upsample_bicubic2d_aa(
            input, output_size=output_size, align_corners=align_corners
        )

    def span(scale):
        support = 2 if (scale >= 1.0) else 2.0 / scale
        interpolate_range = int(support + 0.5) * 2 + 1
        return interpolate_range

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    reduce_dim = span(scale[0]) * span(scale[1])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.upsample_bicubic2d_aa
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "transposed"])
def test_upsample_bicubic2d_aa_layout(layout):
    input = torch.randn((2, 3, 7, 9), device=flag_gems.device)
    if layout == "channels_last":
        input = input.contiguous(memory_format=torch.channels_last)
    elif layout == "transposed":
        input = input.transpose(-1, -2)
    ref_input = utils.to_reference(input)
    ref_out = torch.ops.aten._upsample_bicubic2d_aa(
        ref_input, (11, 13), False, None, None
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_bicubic2d_aa(
            input, (11, 13), False, None, None
        )
    utils.gems_assert_close(result, ref_out, torch.float32, reduce_dim=25)
    expected_format = (
        torch.channels_last if layout == "channels_last" else torch.contiguous_format
    )
    assert result.is_contiguous(memory_format=expected_format)


@pytest.mark.upsample_bicubic2d_aa_vec
@pytest.mark.parametrize("use_scale_factors", [False, True])
def test_upsample_bicubic2d_aa_vec(use_scale_factors):
    input = torch.randn((1, 2, 8, 10), device=flag_gems.device)
    ref_input = utils.to_reference(input)
    args = (None, False, (1.5, 0.7)) if use_scale_factors else ((12, 7), False, None)
    ref_out = torch.ops.aten._upsample_bicubic2d_aa.vec(ref_input, *args)
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_bicubic2d_aa.vec(input, *args)
    utils.gems_assert_close(result, ref_out, torch.float32, reduce_dim=49)


@pytest.mark.upsample_bicubic2d_aa_vec
@pytest.mark.parametrize(
    "output_size,scale_factors", [(None, None), ((12, 7), (1.5, 0.7))]
)
def test_upsample_bicubic2d_aa_vec_requires_one_size(output_size, scale_factors):
    input = torch.randn((1, 2, 8, 10), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten._upsample_bicubic2d_aa.vec(
            input, output_size, False, scale_factors
        )


@pytest.mark.upsample_bicubic2d_aa_out
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_upsample_bicubic2d_aa_out(noncontiguous):
    input = torch.randn((1, 2, 8, 10), device=flag_gems.device)
    ref_input = utils.to_reference(input)
    if noncontiguous:
        out = torch.empty((1, 2, 7, 12), device=flag_gems.device).transpose(-1, -2)
    else:
        out = torch.empty(0, device=flag_gems.device)
    ref_out = torch.ops.aten._upsample_bicubic2d_aa(
        ref_input, (12, 7), False, None, None
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_bicubic2d_aa.out(
            input, (12, 7), False, None, None, out=out
        )
    assert result is out
    utils.gems_assert_close(result, ref_out, torch.float32, reduce_dim=25)


def upsample_bicubic2d_aa_backward_call(grad, input_size, align_corners):
    orig_shape = tuple(input_size)
    n = 1
    for s in orig_shape[:-2]:
        n *= s
    c = orig_shape[-2] if len(orig_shape) >= 2 else 1
    in_h = orig_shape[-2] if len(orig_shape) >= 3 else 1
    in_w = orig_shape[-1]
    if len(orig_shape) >= 4:
        c = orig_shape[-3]
        in_h = orig_shape[-2]
        in_w = orig_shape[-1]
        n = 1
        for s in orig_shape[:-3]:
            n *= s
    else:
        # For 4D input: (N, C, H, W)
        n, c, in_h, in_w = orig_shape

    shape_4d = (n, c, in_h, in_w)
    out_h = grad.shape[-2]
    out_w = grad.shape[-1]

    grad_4d = grad.reshape(n, c, out_h, out_w)

    out = torch.ops.aten._upsample_bicubic2d_aa_backward(
        grad_4d,
        [out_h, out_w],
        list(shape_4d),
        align_corners,
        None,
        None,
    )

    return out.reshape(orig_shape)


@pytest.mark.upsample_bicubic2d_aa_backward
@pytest.mark.parametrize("N,C,H_in,W_in,H_out,W_out,align_corners", PARAMS_BWD)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_upsample_bicubic2d_aa_backward(
    N, C, H_in, W_in, H_out, W_out, align_corners, dtype
):
    shape = (N, C, H_in, W_in)

    grad_shape = (N, C, H_out, W_out)

    res_grad = torch.randn(
        grad_shape,
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_grad = utils.to_reference(res_grad)

    ref_out = upsample_bicubic2d_aa_backward_call(
        ref_grad,
        shape,
        align_corners,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = upsample_bicubic2d_aa_backward_call(
            res_grad.to(dtype),
            shape,
            align_corners,
        )

    assert res_out.shape == shape

    # dtype-specific tolerance
    if dtype == torch.float32:
        atol = 1e-4
    elif dtype == torch.float16:
        atol = 3e-3
    else:  # bfloat16
        atol = 2e-2

    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)

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

import pytest
import torch

import flag_gems

from . import base, consts


class WeightNormInterfaceBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(64, 64), (1024, 1024), (10000, 256)]
    DEFAULT_SHAPE_DESC = "input shape; both first and last dimensions are benchmarked"

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.DEFAULT_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(self.set_more_shapes())

    def set_more_shapes(self):
        return [(4096, 4096), (64, 512, 512)]


def weight_norm_interface_input_fn(shape, dtype, device):
    v = torch.randn(shape, dtype=dtype, device=device)
    for dim in dict.fromkeys((0, len(shape) - 1)):
        g = torch.randn(shape[dim], dtype=dtype, device=device)
        yield v, g, dim


def weight_norm_interface_out_input_fn(shape, dtype, device):
    for v, g, dim in weight_norm_interface_input_fn(shape, dtype, device):
        out0 = torch.empty(0, dtype=dtype, device=device)
        out1 = torch.empty(0, dtype=torch.float32, device=device)
        yield v, g, dim, {"out0": out0, "out1": out1}


@pytest.mark.weight_norm_interface
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface():
    bench = WeightNormInterfaceBenchmark(
        op_name="weight_norm_interface",
        input_fn=weight_norm_interface_input_fn,
        torch_op=torch.ops.aten._weight_norm_interface.default,
    )
    bench.set_gems(flag_gems.weight_norm_interface)

    bench.run()


@pytest.mark.weight_norm_interface_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface_out():
    bench = WeightNormInterfaceBenchmark(
        op_name="weight_norm_interface_out",
        input_fn=weight_norm_interface_out_input_fn,
        torch_op=torch.ops.aten._weight_norm_interface.out,
    )
    bench.set_gems(flag_gems.weight_norm_interface_out)

    bench.run()


def weight_norm_interface_backward_input_fn(shape, dtype, device):
    dim = 0
    w_grad = torch.randn(shape, dtype=dtype, device=device)
    saved_v = torch.randn(shape, dtype=dtype, device=device)
    saved_g = torch.randn(shape[dim], dtype=dtype, device=device)
    saved_norms = torch.randn(shape[dim], dtype=dtype, device=device)
    yield w_grad, saved_v, saved_g, saved_norms, dim


@pytest.mark.weight_norm_interface_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface_backward():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="weight_norm_interface_backward",
        input_fn=weight_norm_interface_backward_input_fn,
        torch_op=torch.ops.aten._weight_norm_interface_backward,
        # NOTE: torch.ops.aten._weight_norm_interface_backward only supports float32,
        # using fp16/bf16 causes "expected scalar type Float but found Half" error.
        # Original: dtypes=consts.FLOAT_DTYPES,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.weight_norm_interface_backward)

    bench.run()

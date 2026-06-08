# Copyright 2025 The Torch-Spyre Authors.
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

import torch
from torch_spyre._inductor import config
from torch.testing import FileCheck
from torch._inductor.exc import InductorError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    run_and_get_code,
)


class TestSpyreConfig(InductorTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_config_default(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")

        comp_fn = torch.compile(fn)
        out, source_codes = run_and_get_code(comp_fn, x)
        # print("test_config_default")
        # print(source_codes[0])
        FileCheck().check("sdsc_fused_abs").check(
            f"sympify('c0'): (sympify('256'), {config.sencores})"
        ).run(source_codes[0])

    @config.patch({"sencores": 64})
    def test_config_too_many_sencores(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")

        with self.assertRaisesRegex(
            InductorError,
            "Unsupported: Spyre backend does not support: invalid SENCORES value 64",
        ):
            comp_fn = torch.compile(fn)
            comp_fn(x)

    @config.patch({"sencores": 16})
    def test_sencores_16(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")
        cfn = torch.compile(fn, dynamic=False)
        out, source_codes = run_and_get_code(cfn, x)
        # print("test_sencores 16")
        # print(source_codes[0])
        FileCheck().check("sdsc_fused_abs").check(
            f"sympify('c0'): (sympify('256'), {config.sencores})"
        ).run(source_codes[0])

    @config.patch({"sencores": 32})
    def test_symbolic_batch_dim_pointwise_split(self):
        """Symbolic batch dim must split by ``granularity`` not ``max_size`` (#2287).

        ``[s, 128]`` fp16 with ``s in [64, 1024]`` (granularity = 64). The planner picks the largest
        divisor of granularity ≤ SENCORES = 32, so the batch dim absorbs all
        32 cores and the static stick dim gets split 1.
        """
        fn = torch.add
        x = torch.randn((1024, 128), dtype=torch.float16)
        y = torch.randn_like(x)
        torch._dynamo.mark_dynamic(x, 0, min=64, max=1024)
        torch._dynamo.mark_dynamic(y, 0, min=64, max=1024)
        comp_fn = torch.compile(fn, dynamic=True)
        _, source_codes = run_and_get_code(comp_fn, x.to("spyre"), y.to("spyre"))
        # Iteration space embeds (size_expr, split). The symbolic batch dim's
        # split must equal SENCORES=32; the static stick dim's split must be 1.
        FileCheck().check("sdsc_fused_add").check(", 32)").check(", 1)").run(
            source_codes[0]
        )

    @config.patch({"sencores": 32})
    def test_symbolic_batch_with_static_dim_leftover(self):
        """Symbolic dim caps at granularity; static dim absorbs the leftover (#2287).

        ``[s, 1024]`` fp16 with ``s in [4, 64]`` (granularity = 4). The symbolic
        batch dim takes the largest divisor of granularity ≤ 32 = 4, leaving 8 cores for the
        static stick dim (1024 / 64 = 16 sticks → split 8).
        """
        fn = torch.add
        x = torch.randn((64, 1024), dtype=torch.float16)
        y = torch.randn_like(x)
        torch._dynamo.mark_dynamic(x, 0, min=4, max=64)
        torch._dynamo.mark_dynamic(y, 0, min=4, max=64)
        comp_fn = torch.compile(fn, dynamic=True)
        _, source_codes = run_and_get_code(comp_fn, x.to("spyre"), y.to("spyre"))
        # Symbolic batch dim split = 4 (largest divisor of granularity=4 ≤ 32);
        # static stick dim split = 8 (largest divisor of 16 sticks ≤ 8).
        FileCheck().check("sdsc_fused_add").check(", 4)").check(", 8)").run(
            source_codes[0]
        )

    # Need a test where changing dxp_lx_frac_avail changes the generated OpSpec
    # @config.patch({"dxp_lx_frac_avail": 0.01, "lx_planning": True})
    # def test_config_dxp_lx_frac_avail(self):
    #    fn = torch.abs
    #    x = torch.randn((256, 128, 512)).to("spyre")
    #
    #    comp_fn = torch.compile(fn)
    #    out, source_codes = run_and_get_code(comp_fn, x)
    #    #print("test_conf_dxp_lx_frac_avail")
    #    #print(source_codes[0])

    # Need a test where setting lx_planning to True generates a different OpSpec
    # @config.patch({'lx_planning': True})
    # def test_config_lx_planning(self):
    #    fn = torch.abs
    #    x = torch.randn((256, 128, 512)).to("spyre")
    #
    #    comp_fn = torch.compile(fn)
    #    out, source_codes = run_and_get_code(comp_fn, x)
    #    #print(source_codes[0])

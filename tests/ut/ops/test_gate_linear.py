#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import unittest
from unittest import mock
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear


def _cpu_unquantized_apply(layer, x, bias=None):
    """Stand in for AscendUnquantizedLinearMethod.apply on CPU.

    Production apply dispatches vllm::unquantized_gemm (PrivateUse1 / NPU only).
    """
    return F.linear(x, layer.weight, bias)


class TestAscendGateLinear(TestBase):
    def setUp(self):
        super().setUp()

        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 1
        self.mock_group.rank_in_group = 0

        self.patches = [
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
            patch(
                "vllm_ascend.ops.linear_op.get_tp_group",
                return_value=self.mock_group,
            ),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

        super().tearDown()

    def test_forward_keeps_router_logits_fp32(self):
        gate = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=torch.float32,
            prefix="test.gate",
        )

        self.assertEqual(gate.weight.dtype, torch.float32)
        self.assertEqual(gate.out_dtype, torch.float32)

        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        with patch.object(gate.quant_method, "apply", side_effect=_cpu_unquantized_apply):
            output, output_bias = gate(hidden_states)

        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape, (2, 4))
        self.assertIsNone(output_bias)

    def test_forward_bf16_router_gemm_fast_path(self):
        """bf16 weight + bf16 x + fp32 out -> tier 4 torch.mm(out_dtype=fp32)."""
        gate = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=torch.float32,
            prefix="test.gate",
        )
        gate.weight.data = gate.weight.data.to(torch.bfloat16)
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)

        with patch("vllm_ascend.ops.fused_moe.gate_linear.torch.mm") as mm:
            mm.return_value = torch.randn(2, 4, dtype=torch.float32)
            output, output_bias = gate(hidden_states)

        self.assertIs(output, mm.return_value)
        self.assertIsNone(output_bias)
        mm.assert_called_once()
        args, kwargs = mm.call_args
        self.assertIs(args[0], hidden_states)
        self.assertEqual(args[1].data_ptr(), gate.weight.data_ptr())
        self.assertEqual(kwargs["out_dtype"], torch.float32)

    def test_none_out_dtype_defaults_to_fp32(self):
        """out_dtype=None (upstream dsv2/glm5next wiring) defaults to fp32:
        bf16 weights take tier 4 directly; fp32 weights cast x instead of
        raising a dtype mismatch."""
        gate = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=None,
            prefix="test.gate",
        )
        self.assertEqual(gate.out_dtype, torch.float32)
        gate.weight.data = gate.weight.data.to(torch.bfloat16)
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)

        with patch("vllm_ascend.ops.fused_moe.gate_linear.torch.mm") as mm:
            mm.return_value = torch.randn(2, 4, dtype=torch.float32)
            output, output_bias = gate(hidden_states)
        mm.assert_called_once()
        self.assertIs(output, mm.return_value)
        self.assertIsNone(output_bias)

        # fp32 weight + bf16 x: tier 5 casts x to the weight dtype.
        gate2 = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=None,
            prefix="test.gate2",
        )
        with patch.object(gate2.quant_method, "apply", side_effect=_cpu_unquantized_apply):
            output2, _ = gate2(hidden_states)
        self.assertEqual(output2.dtype, torch.float32)

    def test_params_follow_model_dtype_unless_forced_fp32(self):
        """Without force_fp32_compute, params follow the model (default)
        dtype, mirroring upstream; with it, the weight is stored fp32."""
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            gate = AscendGateLinear(
                input_size=16,
                output_size=4,
                bias=False,
                out_dtype=torch.float32,
                prefix="test.gate",
            )
            self.assertEqual(gate.weight.dtype, torch.bfloat16)

            forced = AscendGateLinear(
                input_size=16,
                output_size=4,
                bias=False,
                out_dtype=torch.float32,
                force_fp32_compute=True,
                prefix="test.gate.forced",
            )
            self.assertEqual(forced.weight.dtype, torch.float32)
        finally:
            torch.set_default_dtype(default_dtype)

    def test_forward_fallback_casts_output_to_out_dtype(self):
        """Tier 5: x is cast to the weight dtype and the output to out_dtype."""
        gate = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=torch.bfloat16,
            prefix="test.gate",
        )
        # Weight stays fp32 (test default dtype); bf16 x -> fp32 compute,
        # then the output is cast to the requested bf16 out_dtype.
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        with patch.object(gate.quant_method, "apply", side_effect=_cpu_unquantized_apply):
            output, output_bias = gate(hidden_states)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (2, 4))
        self.assertIsNone(output_bias)

    def test_model_can_resolve_default_output_dtype_once(self):
        gate = AscendGateLinear(16, 4, params_dtype=torch.bfloat16)
        self.assertEqual(gate.out_dtype, torch.float32)
        gate.set_out_dtype(torch.bfloat16)
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        with (
            patch("vllm_ascend.ops.fused_moe.gate_linear.torch.mm") as mm,
            patch.object(gate.quant_method, "apply", side_effect=_cpu_unquantized_apply),
        ):
            output, _ = gate(hidden_states)
        mm.assert_not_called()
        self.assertEqual(output.dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "already been set"):
            gate.set_out_dtype(torch.float32)
        explicit_gate = AscendGateLinear(16, 4, out_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "already been set"):
            explicit_gate.set_out_dtype(torch.bfloat16)

    def test_tensor_only_return_for_fast_path_and_fallback(self):
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        for params_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(params_dtype=params_dtype):
                gate = AscendGateLinear(16, 4, params_dtype=params_dtype, return_bias=False)
                with (
                    patch("vllm_ascend.ops.fused_moe.gate_linear.torch.mm") as mm,
                    patch.object(gate.quant_method, "apply", side_effect=_cpu_unquantized_apply),
                ):
                    mm.return_value = torch.randn(2, 4, dtype=torch.float32)
                    output = gate(hidden_states)
                self.assertIsInstance(output, torch.Tensor)
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(mm.call_count, int(params_dtype == torch.bfloat16))

    def test_bias_bypasses_fast_path_and_honors_skip_bias_add(self):
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        for skip_bias_add in (False, True):
            with self.subTest(skip_bias_add=skip_bias_add):
                gate = AscendGateLinear(16, 4, bias=True, params_dtype=torch.bfloat16, skip_bias_add=skip_bias_add)
                gate.weight.data.fill_(1)
                gate.bias.data.fill_(2)
                with (
                    patch("vllm_ascend.ops.fused_moe.gate_linear.torch.mm") as mm,
                    patch.object(gate.quant_method, "apply", side_effect=_cpu_unquantized_apply),
                ):
                    output, output_bias = gate(hidden_states)
                mm.assert_not_called()
                expected = F.linear(hidden_states, gate.weight, None if skip_bias_add else gate.bias)
                torch.testing.assert_close(output, expected.float())
                self.assertIs(output_bias, gate.bias if skip_bias_add else None)

    def test_quantized_gate_without_plain_weight_uses_quant_method(self):
        quant_method = mock.MagicMock()
        quant_config = mock.MagicMock()
        quant_config.get_quant_method.return_value = quant_method
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        quant_method.apply.return_value = torch.randn(2, 4, dtype=torch.bfloat16)
        for return_bias in (False, True):
            with self.subTest(return_bias=return_bias):
                gate = AscendGateLinear(16, 4, quant_config=quant_config, return_bias=return_bias)
                self.assertFalse(hasattr(gate, "weight"))
                self.assertFalse(gate.is_unquantized)
                with patch("vllm_ascend.ops.fused_moe.gate_linear.torch.mm") as mm:
                    result = gate(hidden_states)
                mm.assert_not_called()
                self.assertIs(quant_method.apply.call_args.args[1], hidden_states)
                output = result[0] if return_bias else result
                torch.testing.assert_close(output, quant_method.apply.return_value.float())
                if return_bias:
                    self.assertIsNone(result[1])


if __name__ == "__main__":
    unittest.main()

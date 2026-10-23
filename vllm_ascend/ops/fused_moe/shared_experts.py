#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import tensor_model_parallel_all_gather, tensor_model_parallel_reduce_scatter
from vllm.logger import logger
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.lora.fused_moe import has_lora
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    npu_stream_switch,
    shared_experts_calculation_stream,
)


# ---------------------------------------------------------------------------
# W3-F3 (2026-10-16): shared scalar-gate pack（L0 direct/whole 变体, N=513/640）
# 共享层（Qwen3NextMLP; 含 gate_up_proj/down_proj/expert_gate）的 pack = cat([gate_up_w, scalar_gate_w], dim=0)
# 或 640 等宽 pad。同输入三投影合一（L0 bitwise 全 shape PASS）。
# env `VLLM_HUST_SHARED_GATE_PACK`: 513direct|513whole|640direct|640whole, 默认 0 不启用。
# 本代码只在 worker hook 阶段（非 dynamo traced 区）里被调用; pack 结果写 layer._f3_wpack。
# ---------------------------------------------------------------------------
_F3_ALLOWED = ("513direct", "513whole", "640direct", "640whole")


def _f3_parse_mode(raw: str) -> tuple[int, str]:
    """`513direct` -> (513, 'direct'); 非法返回 (0, 'direct')"""
    v = (raw or "").strip().lower()
    if v not in _F3_ALLOWED:
        return 0, "direct"
    digits = "".join(c for c in v if c.isdigit())
    mode = "direct" if "direct" in v else "whole"
    try:
        return int(digits), mode
    except ValueError:
        return 0, "direct"


def pack_shared_gate_weight(shared_layer, mode_str: str) -> None:
    """worker hook 用 helper: 对 shared layer（Qwen3NextMLP 实例）构建 packed weight.

    要求 shared_layer 有 gate_up_proj / down_proj / expert_gate 权重。
    写 shared_layer._f3_wpack 与 _f3_pack_mode; 幂等（重复调用覆盖）。
    fail-closed: 权重缺位/dtype/设备不匹配 → raise。
    """
    n, mode = _f3_parse_mode(mode_str)
    if n == 0:
        return
    eg = getattr(shared_layer, "expert_gate", None)
    if eg is None or getattr(eg, "weight", None) is None:
        raise RuntimeError(
            "W3F3: shared layer has no expert_gate weight; "
            "shared scalar-gate pack not applicable"
        )
    w13 = shared_layer.gate_up_proj.weight       # [2*intermediate, hidden]（本 rank shard）
    w_s = eg.weight                              # [1, hidden]
    if w13.dtype != w_s.dtype:
        raise RuntimeError(
            f"W3F3: dtype mismatch gate_up_proj={w13.dtype} expert_gate={w_s.dtype}"
        )
    if w13.device != w_s.device:
        raise RuntimeError(
            f"W3F3: device mismatch gate_up_proj={w13.device} expert_gate={w_s.device}"
        )
    if n == 513:
        pack = torch.cat([w13, w_s], dim=0)
    else:
        pad_rows = n - (w13.shape[0] + w_s.shape[0])
        if pad_rows < 0:
            raise RuntimeError(f"W3F3: target width {n} < rows+1={w13.shape[0]+w_s.shape[0]}")
        pad = torch.zeros(pad_rows, w13.shape[1], dtype=w13.dtype, device=w13.device)
        pack = torch.cat([w13, w_s, pad], dim=0)
    shared_layer._f3_wpack = pack
    shared_layer._f3_pack_mode = mode

# CANN uses 36 to select FP8 E4M3FN output for situ_mx_quant.
SITU_MX_DST_TYPE_E4M3FN = 36


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    after_routed_experts: torch.npu.Event | None = field(default=None)
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_gmm2: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)
    after_routed_finalize: torch.npu.Event | None = field(default=None)


class SharedExpertParallelMode(Enum):
    """Effective activation and weight layout for a shared-expert forward."""

    TENSOR_PARALLEL = auto()  # Full activations, TP-sharded weights.
    SHARED_EXPERT_DATA_PARALLEL_ONLY = auto()  # Full activations, replicated weights (DP only).
    SEQUENCE_PARALLEL_ONLY = auto()  # Sharded activations, TP-sharded weights (SP only).
    SEQUENCE_PARALLEL_SEDP = auto()  # Sharded activations, replicated weights (SP + DP).


class AscendSharedExperts:
    """Ascend-owned shared expert executor.

    Keep the original shared expert module registered on ``AscendMoERunner``
    for checkpoint compatibility while moving split/overlap execution details
    out of the runner.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        quant_type: QuantType,
        quant_method: FusedMoEMethodBase,
    ):
        self.layer = layer
        self.moe_config = moe_config
        self.hidden_size = moe_config.hidden_dim
        self.shared_expert_input_size = getattr(
            layer.gate_up_proj,
            "input_size",
            self.hidden_size,
        )
        self.in_dtype = moe_config.in_dtype
        self.swiglu_limit = 0.0 if moe_config.swiglu_limit is None else moe_config.swiglu_limit
        self.swiglu_alpha = 1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha
        self.swiglu_beta = 0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta
        self.is_sequence_parallel = moe_config.is_sequence_parallel
        self.situ_activation = layer.act_fn if isinstance(layer.act_fn, SituAndMul) else None
        self.quant_type = quant_type
        self.lora_context = None
        ascend_config = get_ascend_config()
        self.multistream_overlap = ascend_config.multistream_overlap_shared_expert
        self.weights_replicated = ascend_config.enable_shared_expert_dp

        if self.multistream_overlap:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection, activation,
            # then down projection) yields identical results to integrated
            # computation after weight loading.
            original_process_weights = quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self.validate_consistency()
                return result

            quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context

    def validate_consistency(self):
        """Validate that split shared expert computation matches integrated computation."""
        test_input = (
            torch.rand(
                10,
                self.shared_expert_input_size,
                device="npu",
                dtype=self.in_dtype,
            )
            * 2
            - 1
        )  # Random input for testing, scoped to [-1, 1]

        integrated_out = self.layer(test_input)
        part1_out = self.part1(test_input)
        shared_act = self.apply_activation(part1_out)
        split_out = self.part2(test_input, shared_act)

        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error(
                "[fused_moe/layer] Shared expert split computation validation failed."
                " The split-path computation does not match the integrated-path result."
                " max_abs_diff=%s, integrated_sum=%s, integrated_norm=%s,"
                " split_sum=%s, split_norm=%s, hidden_size=%s, dtype=%s.",
                diff.max().item(),
                integrated_out.sum().item(),
                integrated_out.norm().item(),
                split_out.sum().item(),
                split_out.norm().item(),
                self.shared_expert_input_size,
                self.in_dtype,
            )
            raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
        logger.info_once(
            "[fused_moe/layer] Shared expert split computation validation passed."
            " Integrated and split-path results are consistent."
        )

    def part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self.layer.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def apply_activation(self, shared_gate_up: torch.Tensor):
        return self.layer.act_fn(shared_gate_up)  # type: ignore

    def part2(self, hidden_states: torch.Tensor, shared_act: torch.Tensor):
        shared_out, _ = self.layer.down_proj(shared_act)  # type: ignore

        # Qwen3-Next specific gating mechanism
        if hasattr(self.layer, "expert_gate") and self.layer.expert_gate is not None:
            gate_out, _ = self.layer.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def parallel_mode(self) -> SharedExpertParallelMode:
        """Resolve the effective activation/weight layout for this forward."""
        # EP rewrites FusedMoEParallelConfig.tp_size to 1 because each routed
        # expert is local. Shared-expert linears still span the physical TP
        # group, so their layout must be derived from that group instead.
        tp_size = self.moe_config.tp_group.world_size
        if tp_size <= 1:
            return SharedExpertParallelMode.TENSOR_PARALLEL

        if self.moe_config.is_sequence_parallel:
            # SP has already sharded the token dimension before entering the
            # runner. Replicated weights (SP+DP) compute directly on the
            # shard; TP-sharded weights (SP-only) gather the shard first and
            # reduce-scatter the output back.
            if self.weights_replicated:
                return SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP
            return SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
        if self.weights_replicated:
            return SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY
        return SharedExpertParallelMode.TENSOR_PARALLEL

    def _prepare_local_dp_input(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        original_num_tokens = hidden_states.shape[0]
        # See parallel_mode(): moe_config.tp_size describes routed experts in
        # EP, while this token split follows the shared-expert TP group.
        tp_group = self.moe_config.tp_group
        tp_size = tp_group.world_size
        pad_size = (tp_size - original_num_tokens % tp_size) % tp_size
        if pad_size > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
        hidden_states = torch.tensor_split(
            hidden_states,
            tp_size,
            dim=0,
        )[tp_group.rank_in_group]
        return hidden_states, (original_num_tokens, pad_size)

    def _finalize_local_dp_output(
        self,
        shared_out: torch.Tensor,
        metadata: tuple[int, int],
    ) -> torch.Tensor:
        original_num_tokens, pad_size = metadata
        shared_out = self.moe_config.tp_group.all_gather(shared_out, dim=0)
        if pad_size > 0:
            shared_out = shared_out[:original_num_tokens]
        return shared_out

    def _gather_sp_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gather SP-sharded activations to the full sequence via TP all-gather + unpad.

        SP shards the token dimension across the TP group as [ceil(T/TP), H]
        per rank (``sequence_parallel_chunk`` pads T to a multiple of TP).
        TP-sharded shared-expert weights need the full activations, so
        all-gather the shards back to [T_padded, H] and drop the padding rows.
        """
        gathered = tensor_model_parallel_all_gather(hidden_states, dim=0)
        return gathered[: _EXTRA_CTX.num_tokens]

    def _pad_and_reduce_scatter(self, shared_out: torch.Tensor) -> torch.Tensor:
        """Pad the full output to a TP multiple, then reduce-scatter to the SP shard.

        Called on exit of the SP-only path (TP-sharded weights) to convert the
        full [T, H] output back to this rank's SP shard [ceil(T/TP), H]. The
        reduce-scatter also sums the TP-partial down-projection results
        (down_proj is built with reduce_results=False), replacing the usual
        TP all-reduce.
        """
        tp_size = self.moe_config.tp_group.world_size
        original_num_tokens = shared_out.shape[0]
        pad_size = (tp_size - original_num_tokens % tp_size) % tp_size
        if pad_size > 0:
            shared_out = F.pad(shared_out, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(shared_out, dim=0)

    def prepare_input_before_routed_experts(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.npu.Event | None]:
        """Start the SP-only input all-gather on the shared-expert stream."""
        if not (self.multistream_overlap and self.parallel_mode() is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY):
            return hidden_states, None

        input_ready = torch.npu.current_stream().record_event()
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=True):
            torch.npu.current_stream().wait_event(input_ready)
            hidden_states = self._gather_sp_input(hidden_states)
            all_gather_done = torch.npu.current_stream().record_event()
        return hidden_states, all_gather_done

    def forward(
        self,
        hidden_states: torch.Tensor,
        fused_moe_evts: FusedMoEEvents,
        input_is_gathered: bool = False,
    ):
        mode = self.parallel_mode()
        local_dp_metadata = None
        down_projection_ready = (
            fused_moe_evts.after_routed_finalize
            if self.multistream_overlap and mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
            else fused_moe_evts.before_combine
        )

        def maybe_wait_event(evt: torch.npu.Event | None):
            if evt is not None:
                torch.npu.current_stream().wait_event(evt)

        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
            if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
                # Full activations + replicated weights: shard tokens locally,
                # run the MLP, then gather its complete output.
                maybe_wait_event(fused_moe_evts.before_routed_experts)
                hidden_states, local_dp_metadata = self._prepare_local_dp_input(hidden_states)
            elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY:
                # Sharded activations + TP-sharded weights: gather the SP
                # shard to full activations before the MLP; the output is
                # padded and reduce-scattered back below.
                if not input_is_gathered:
                    maybe_wait_event(fused_moe_evts.before_routed_experts)
                    hidden_states = self._gather_sp_input(hidden_states)
            # Only used for int quantization
            has_quantized_shared_without_lora = (
                not has_lora(self.lora_context)
                and hasattr(self.layer.gate_up_proj, "weight_scale")
                and hasattr(self.layer.down_proj, "weight_scale")
            )
            if has_quantized_shared_without_lora and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.after_routed_experts)
                hidden_states = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.layer.gate_up_proj.weight,
                    self.layer.gate_up_proj.weight_scale,
                    pertoken_scale=None,
                    bias=None,
                    output_dtype=torch.int32,
                )
                # Execute activation concurrently with gmm2.

                maybe_wait_event(fused_moe_evts.before_gmm2)
                if self.situ_activation is not None:
                    quantized_x, swiglu_out_scale = torch.ops._C_ascend.dequant_situ_quant(
                        x=hidden_states,
                        weight_scale=self.layer.gate_up_proj.weight_scale_fp32,
                        activation_scale=pertoken_scale,
                        bias=None,
                        quant_scale=None,
                        quant_offset=None,
                        group_index=None,
                        beta=self.situ_activation.beta,
                        linear_beta=self.situ_activation.linear_beta or 0.0,
                        activate_left=True,
                        quant_mode="dynamic",
                    )
                else:
                    quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                        x=hidden_states,
                        weight_scale=self.layer.gate_up_proj.weight_scale_fp32,
                        activation_scale=pertoken_scale,
                        bias=None,
                        quant_scale=None,
                        quant_offset=None,
                        group_index=None,
                        activate_left=True,
                        quant_mode=1,
                        swiglu_mode=1,
                        clamp_limit=self.swiglu_limit,
                        **(
                            {}
                            if not get_current_hardware_profile().supports(HardwareCapability.FUSED_SWIGLU_TUNING_ARGS)
                            else {"glu_alpha": self.swiglu_alpha, "glu_bias": self.swiglu_beta}
                        ),
                    )
                maybe_wait_event(down_projection_ready)
                shared_out = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.layer.down_proj.weight,
                    self.layer.down_proj.weight_scale,
                    pertoken_scale=swiglu_out_scale,
                    bias=None,
                    output_dtype=original_dtype,
                )
            elif has_quantized_shared_without_lora and self.quant_type == QuantType.W4A8MXFP:
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=torch.float8_e4m3fn
                )
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                hidden_states = self.layer.gate_up_proj((quantized_x, pertoken_scale))[0]
                # Execute activation concurrently with gmm2.
                maybe_wait_event(fused_moe_evts.before_gmm2)
                if self.situ_activation is not None:
                    quantized_x, swiglu_out_scale = torch.ops._C_ascend.situ_mx_quant(
                        x=hidden_states,
                        beta=self.situ_activation.beta,
                        linear_beta=self.situ_activation.linear_beta or 0.0,
                        activate_left=True,
                        dst_type=SITU_MX_DST_TYPE_E4M3FN,
                    )
                else:
                    quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                        hidden_states,
                        topk_weight=None,
                        group_index=None,
                        dst_type=torch.float8_e4m3fn,
                        quant_mode=2,
                        clamp_value=self.swiglu_limit,
                    )
                maybe_wait_event(down_projection_ready)
                shared_out = self.layer.down_proj((quantized_x, swiglu_out_scale))[0]
            else:
                # Ensure the shared experts wait for hidden_states to be ready.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                # Execute the gate projection concurrently with dispatch.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                _f3_w = getattr(self.layer, "_f3_wpack", None)
                _f3_mode = getattr(self.layer, "_f3_pack_mode", "direct")
                if _f3_w is not None:
                    # W3-F3: 一次 pack 投影, scalar gate 切列 sigmoid（L0 direct/whole 变体,
                    # bitwise 已在 L0 全 shape 通过; L3 真模型须再验一次）
                    y = F.linear(hidden_states, _f3_w)
                    if _f3_mode == "whole":
                        y = y.contiguous()
                    _a, _b = y[..., :256], y[..., 256:512]
                    _s_col = y[..., 512:513]
                    _z = F.silu(_a) * _b
                    maybe_wait_event(fused_moe_evts.before_gmm2)
                    maybe_wait_event(down_projection_ready)
                    _down_out, _ = self.layer.down_proj(_z)
                    shared_out = torch.sigmoid(_s_col) * _down_out
                else:
                    part1_out = self.part1(hidden_states)
                    # Execute activation concurrently with routed GMM2.
                    maybe_wait_event(fused_moe_evts.before_gmm2)
                    shared_act = self.apply_activation(part1_out)
                    # Execute the down projection concurrently with combine.
                    maybe_wait_event(down_projection_ready)
                    shared_out = self.part2(hidden_states, shared_act)

        if self.multistream_overlap and mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY:
            # Keep the shared-expert output collective on the auxiliary stream,
            # but start it only after routed dispatch/combine/finalize
            # communication has completed.
            with npu_stream_switch(shared_experts_calculation_stream(), enabled=True):
                maybe_wait_event(fused_moe_evts.after_routed_finalize)
                shared_out = self._pad_and_reduce_scatter(shared_out)

        # Make sure the default stream waits for the shared experts stream to finish.
        if self.multistream_overlap:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

        if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            assert local_dp_metadata is not None
            shared_out = self._finalize_local_dp_output(shared_out, local_dp_metadata)
        elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY and not self.multistream_overlap:
            shared_out = self._pad_and_reduce_scatter(shared_out)
        return shared_out

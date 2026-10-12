# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import os

import torch
from safetensors.torch import save_file
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.model_loader import ShardedStateLoader


def save_sharded_state(
    model: torch.nn.Module,
    path: str,
    pattern: str | None = None,
    max_size: int | None = None,
    *,
    restore_unquantized_moe_layout: bool = False,
) -> None:
    """Save a generic-loadable TP shard from an Ascend runtime model.

    The unquantized Ascend fused-MoE post-load hook transposes ``w13_weight``
    and ``w2_weight`` into its kernel layout. Generic sharded-state loading
    runs that hook again, so export must first restore the initialization
    layout. Other parameters are serialized without transformation.
    """
    if pattern is None:
        pattern = ShardedStateLoader.DEFAULT_PATTERN

    rank = get_tensor_model_parallel_rank()
    part_idx = 0
    total_size = 0
    state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
    state_dict_part: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if restore_unquantized_moe_layout and tensor.ndim == 3 and key.endswith(("w13_weight", "w2_weight")):
            tensor = tensor.transpose(1, 2).contiguous()

        param_size = tensor.nelement() * tensor.element_size()
        if state_dict_part and max_size is not None and total_size + param_size > max_size:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(state_dict_part, os.path.join(path, filename))
            part_idx += 1
            total_size = 0
            state_dict_part = {}

        state_dict_part[key] = tensor
        total_size += param_size

    if state_dict_part:
        filename = pattern.format(rank=rank, part=part_idx)
        save_file(state_dict_part, os.path.join(path, filename))

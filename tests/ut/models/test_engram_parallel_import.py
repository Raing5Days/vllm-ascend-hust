# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "missing_module",
    [None, "vllm.models.deepseek_v41.nvidia", "vllm.models.deepseek_v41.nvidia.engram"],
)
def test_engram_shard_rank_import_prefers_nvidia_with_common_fallback(monkeypatch, missing_module):
    nvidia_rank, common_rank = object(), object()
    imports = []
    original_import = builtins.__import__

    def import_engram(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.models.deepseek_v41.nvidia.engram":
            imports.append("nvidia")
            if missing_module is not None:
                raise ModuleNotFoundError(name=missing_module)
            return SimpleNamespace(engram_head_shard_rank=nvidia_rank)
        if name == "vllm.models.deepseek_v41.common.engram" and "engram_head_shard_rank" in fromlist:
            imports.append("common")
            return SimpleNamespace(engram_head_shard_rank=common_rank)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_engram)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/parallel.py"
    module = runpy.run_path(str(path))
    assert module["engram_head_shard_rank"] is (common_rank if missing_module else nvidia_rank)
    assert imports == (["nvidia", "common"] if missing_module else ["nvidia"])


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError(name="unrelated_dependency"),
        ModuleNotFoundError(name="vllm.models.deepseek_v41"),
        ImportError("missing helper", name="vllm.models.deepseek_v41.nvidia.engram"),
    ],
)
def test_engram_shard_rank_import_preserves_unrelated_errors(monkeypatch, error):
    original_import = builtins.__import__

    def import_engram(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.models.deepseek_v41.nvidia.engram":
            raise error
        if name == "vllm.models.deepseek_v41.common.engram" and "engram_head_shard_rank" in fromlist:
            pytest.fail("Unrelated import errors must not trigger the fallback")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_engram)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/parallel.py"
    with pytest.raises(type(error)) as caught:
        runpy.run_path(str(path))
    assert caught.value is error

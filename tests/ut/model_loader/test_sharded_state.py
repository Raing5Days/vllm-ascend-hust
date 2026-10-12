from unittest.mock import patch

import torch

from vllm_ascend.model_loader.sharded_state import save_sharded_state


class _Experts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w13_weight = torch.nn.Parameter(torch.arange(24).reshape(2, 3, 4).float())
        self.w2_weight = torch.nn.Parameter(torch.arange(24, 48).reshape(2, 3, 4).float())


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _Experts()
        self.dense_weight = torch.nn.Parameter(torch.arange(6).reshape(2, 3).float())


@patch(
    "vllm_ascend.model_loader.sharded_state.get_tensor_model_parallel_rank",
    return_value=1,
)
@patch("vllm_ascend.model_loader.sharded_state.save_file")
def test_save_sharded_state_restores_unquantized_moe_layout(mock_save_file, _mock_rank):
    model = _Model()

    save_sharded_state(
        model,
        "/checkpoint",
        restore_unquantized_moe_layout=True,
    )

    state_dict, filename = mock_save_file.call_args.args
    assert filename == "/checkpoint/model-rank-1-part-0.safetensors"
    assert state_dict["experts.w13_weight"].shape == (2, 4, 3)
    assert state_dict["experts.w2_weight"].shape == (2, 4, 3)
    torch.testing.assert_close(
        state_dict["experts.w13_weight"],
        model.experts.w13_weight.transpose(1, 2),
    )
    torch.testing.assert_close(state_dict["dense_weight"], model.dense_weight)


@patch(
    "vllm_ascend.model_loader.sharded_state.get_tensor_model_parallel_rank",
    return_value=0,
)
@patch("vllm_ascend.model_loader.sharded_state.save_file")
def test_save_sharded_state_preserves_runtime_layout_when_not_requested(mock_save_file, _mock_rank):
    model = _Model()

    save_sharded_state(model, "/checkpoint")

    state_dict, _filename = mock_save_file.call_args.args
    assert state_dict["experts.w13_weight"].shape == (2, 3, 4)
    assert state_dict["experts.w2_weight"].shape == (2, 3, 4)


@patch(
    "vllm_ascend.model_loader.sharded_state.get_tensor_model_parallel_rank",
    return_value=0,
)
@patch("vllm_ascend.model_loader.sharded_state.save_file")
def test_save_sharded_state_splits_only_nonempty_parts(mock_save_file, _mock_rank):
    model = _Model()

    save_sharded_state(model, "/checkpoint", max_size=1)

    assert mock_save_file.call_count == 3
    assert all(call.args[0] for call in mock_save_file.call_args_list)
    assert [call.args[1] for call in mock_save_file.call_args_list] == [
        "/checkpoint/model-rank-0-part-0.safetensors",
        "/checkpoint/model-rank-0-part-1.safetensors",
        "/checkpoint/model-rank-0-part-2.safetensors",
    ]

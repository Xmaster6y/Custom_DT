"""
Test translate.
"""

import chess
import torch

import src.utils.translate as translate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(DEVICE)


class TestEncodeSeq:
    def test_simple_seq(self, simple_seq):
        move_indices, _, _ = translate.encode_seq(simple_seq)
        assert len(move_indices) == 12
        assert move_indices[0] == chess.D2 + 64 * chess.D4  # d2 -> d4


class TestInputFormat:
    def test_simple_seq(self, simple_seq, default_64_model):
        move_indices, board_tensors, end_rewards = translate.encode_seq(
            simple_seq,
            translate.board_to_64tensor,
        )
        input_dict = translate.format_inputs(
            move_indices,
            board_tensors,
            end_rewards,
            act_dim=4672,
            state_dim=64,
            device=DEVICE,
            window_size=10,
            generator=None,
            return_dict=True,
        )
        assert input_dict["actions"].shape == (1, 10, 4672)
        _ = default_64_model(**input_dict)

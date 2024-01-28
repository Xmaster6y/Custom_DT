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
            sequence=simple_seq,
            act_dim=4672,
            state_dim=64,
            device=DEVICE,
            window_size=10,
            generator=None,
            return_dict=True,
        )
        assert input_dict["actions"].shape == (1, 10, 4672)
        _ = default_64_model(**input_dict)


class TestBoardTensorToFEN:
    def test_initial_board_to_FEN(self):
        board = chess.Board()
        tensor = translate.board_to_72tensor(board)
        assert translate.complete_tensor_to_fen(tensor) == chess.STARTING_FEN

    def test_default_dataset_to_FEN(self, default_dataset_boards):
        for board in default_dataset_boards:
            assert translate.complete_tensor_to_fen(translate.board_to_72tensor(board)) == board.fen()

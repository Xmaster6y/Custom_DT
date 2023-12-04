"""
File to test the encodings for the Leela Chess Zero engine.
"""

import random

import chess
import pytest
import torch

from src.utils import leela_encodings
from src.utils.leela_constants import ACT_DIM, STATE_DIM


@pytest.fixture(scope="module")
def simple_seq():
    return "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"


@pytest.fixture(scope="module")
def encoded_simple_seq(simple_seq):
    return leela_encodings.encode_seq(
        simple_seq,
        board_to_tensor=leela_encodings.board_to_tensor,
        move_to_index=leela_encodings.encode_move,
    )


class TestStability:
    def test_encode_decode(self):
        """
        Test that encoding and decoding a move is the identity.
        """
        board = chess.Board()
        seed = 42
        random.seed(seed)
        us, them = chess.WHITE, chess.BLACK
        for _ in range(20):
            move = random.choice(list(board.legal_moves))
            encoded_move = leela_encodings.encode_move(move, (us, them))
            decoded_move = leela_encodings.decode_move(encoded_move, (us, them), board)
            assert move == decoded_move
            board.push(move)
            us, them = them, us


class TestEncoding:
    def test_board_to_tensor(self, encoded_simple_seq):
        """
        Test that the board to tensor function works.
        """
        white_board_tensors = encoded_simple_seq["board_tensors"][chess.WHITE]
        black_board_tensors = encoded_simple_seq["board_tensors"][chess.BLACK]
        for board_tensor in white_board_tensors[:3]:  # No capture during first 3 moves
            planes = board_tensor.sum(dim=(1, 2))
            assert planes.shape == (20,)
            assert (planes[:12] == torch.tensor([8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1])).all()
            assert (planes[13:17] == torch.full((4,), 64.0)).all()
            assert planes[17] == 0

        for board_tensor in black_board_tensors[:3]:
            planes = board_tensor.sum(dim=(1, 2))
            assert planes.shape == (20,)
            assert (planes[:12] == torch.tensor([8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1])).all()
            assert (planes[13:17] == torch.full((4,), 64.0)).all()
            assert planes[17] == 64


class TestFormting:
    def test_encode_seq(self, encoded_simple_seq):
        """
        Test that encoding a sequence of moves is correct.
        """
        move_indices = encoded_simple_seq["move_indices"]
        board_tensors = encoded_simple_seq["board_tensors"]
        assert len(move_indices[chess.WHITE]) == len(board_tensors[chess.BLACK]) == 6
        assert move_indices[chess.WHITE][0] == 293
        assert move_indices[chess.BLACK][0] == 322  # Flipped rows and columns

    def test_format_inputs_one_player(self, encoded_simple_seq):
        """
        Test that formatting a sequence of moves is correct.
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        inputs = leela_encodings.format_inputs(
            encoded_simple_seq,
            window_size=6,
            one_player=True,
            generator=generator,
            device=torch.device("cpu"),
        )
        assert inputs["states"].shape == (1, 6, STATE_DIM)
        assert inputs["actions"].shape == (1, 6, ACT_DIM)
        assert inputs["returns_to_go"].shape == (1, 6, 1)
        assert (inputs["timesteps"] == torch.tensor([0, 2, 4, 6, 8, 10])).all()

    def test_format_inputs_two_player(self, encoded_simple_seq):
        """
        Test that formatting a sequence of moves is correct.
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        inputs = leela_encodings.format_inputs(
            encoded_simple_seq,
            window_size=6,
            one_player=False,
            generator=generator,
            device=torch.device("cpu"),
        )
        assert inputs["states"].shape == (1, 12, STATE_DIM)
        assert inputs["actions"].shape == (1, 12, ACT_DIM)
        assert inputs["returns_to_go"].shape == (1, 12, 1)
        assert (inputs["timesteps"] == torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])).all()

        states = inputs["states"].reshape(1, 12, -1, 8, 8).squeeze(0)
        planes = states.sum(dim=(2, 3))
        assert (planes[:6, :12] == torch.tensor([8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1])).all()
        assert (planes[:, 13:17] == torch.full((12, 4), 64.0)).all()
        assert (planes[:, 17] == torch.tensor([0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64])).all()

"""
File to test the encodings for the Leela Chess Zero engine.
"""

import random

import chess
import pytest
import torch

from src.metric.stockfish import StockfishMetric
from src.utils import leela_encodings, translate
from src.utils.leela_constants import ACT_DIM, STATE_DIM


@pytest.fixture(scope="module")
def simple_seq():
    return "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"


@pytest.fixture(scope="module")
def stockfish_metric():
    return StockfishMetric()


@pytest.fixture(scope="module")
def encoded_simple_seq(simple_seq, stockfish_metric):
    def position_evaluator(board, us_them):
        player = "white" if us_them[0] == chess.WHITE else "black"
        return stockfish_metric.eval_board(board, player=player)

    return leela_encodings.encode_seq(
        simple_seq,
        board_to_tensor=leela_encodings.board_to_tensor,
        move_to_index=leela_encodings.encode_move,
        position_evaluator=position_evaluator,
    )


@pytest.fixture(scope="module")
def encoded_deprec_simple_seq(simple_seq):
    return translate.encode_seq(
        simple_seq,
        board_to_tensor=translate.board_to_64tensor,
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


class TestFormatting:
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


class TestRewardEquivalence:
    """
    Tests that the reward representation is the same for both the Leela encodings and the deprecated encoding
    """

    def test_reward_equivalence_one_player(
        self, encoded_simple_seq, encoded_deprec_simple_seq, simple_seq, stockfish_metric
    ):
        """
        Test that the reward representation is the same for both the Leela encodings
            and the deprecated encoding for a single player game
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        leela_inputs = leela_encodings.format_inputs(
            encoded_simple_seq,
            window_size=6,
            one_player=True,
            generator=generator,
            device=torch.device("cpu"),
            shaping_rewards=True,
        )
        deprec_inputs = translate.format_inputs(
            encoded_deprec_simple_seq[0],
            encoded_deprec_simple_seq[1],
            encoded_deprec_simple_seq[2],
            sequence=simple_seq,
            act_dim=4672,
            state_dim=64,
            device=torch.device("cpu"),
            window_size=11,
            generator=generator,
            return_dict=True,
            one_player=True,
            shaping_rewards=True,
            stockfish_metric=stockfish_metric,
        )
        assert torch.all(leela_inputs["timesteps"] == deprec_inputs["timesteps"]).item()
        assert torch.all(leela_inputs["returns_to_go"] == deprec_inputs["returns_to_go"]).item()

    def test_reward_equivalence_two_player(
        self, encoded_simple_seq, encoded_deprec_simple_seq, simple_seq, stockfish_metric
    ):
        """
        Test that the reward representation is the same for both the Leela encodings
            and the deprecated encoding for a two player game
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        leela_inputs = leela_encodings.format_inputs(
            encoded_simple_seq,
            window_size=5,
            one_player=False,
            generator=generator,
            device=torch.device("cpu"),
            shaping_rewards=True,
        )
        deprec_inputs = translate.format_inputs(
            encoded_deprec_simple_seq[0],
            encoded_deprec_simple_seq[1],
            encoded_deprec_simple_seq[2],
            sequence=simple_seq,
            act_dim=4672,
            state_dim=64,
            device=torch.device("cpu"),
            window_size=11,
            generator=generator,
            return_dict=True,
            one_player=False,
            shaping_rewards=True,
            stockfish_metric=stockfish_metric,
        )
        assert torch.all(leela_inputs["timesteps"] == deprec_inputs["timesteps"][0, :-1]).item()
        assert torch.all(leela_inputs["returns_to_go"] == deprec_inputs["returns_to_go"][:, :-1]).item()

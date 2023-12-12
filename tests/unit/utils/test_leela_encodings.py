"""
File to test the encodings for the Leela Chess Zero engine.
"""

import random

import chess
import pytest
import torch

from src.utils import leela_encodings
from src.utils.leela_constants import ACT_DIM, STATE_DIM

torch.set_printoptions(sci_mode=False)


@pytest.fixture(scope="module")
def simple_seq():
    return "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"


@pytest.fixture(scope="module")
def complex_seq():
    return """1. e3 Nf6 2. Nf3 Nc6 3. Bb5 e5 4. d3 Na5 5. Nxe5 c6 6. Ba4 Qc7
            7. Ng4 b5 8. Bb3 Be7 9. Bd2 h5 10. Nxf6+ Bxf6 11. Bc3 Qd8 12. O-O b4
            13. Bd4 d5 14. Ba4 h4 15. Qf3 h3 16. g3 Bxd4 17. exd4 O-O 18. Qf4 f6
            19. a3 b3 20. cxb3 Qb6 21. b4 Nb7 22. g4 Bd7 23. Nc3 Rad8 24. Rae1 a6
            25. Re7 Rf7 26. Rfe1 Kf8 27. Rxf7+ Kxf7 28. g5 a5 29. g6+ Kxg6 30. Re7 axb4
            31. Qg3+ Kf5 32. Bd1 Qxd4 33. Qxh3+ Kg5 34. Qh5+ Kf4 35. Ne2#"""


@pytest.fixture(scope="function")
def encoded_simple_seq(simple_seq, stockfish_metric_fix_compose):
    def position_evaluator(board, us_them):
        player = "white" if us_them[0] else "black"
        return stockfish_metric_fix_compose.eval_board(board, player=player)

    encoded_seq = leela_encodings.encode_seq(
        simple_seq,
        board_to_tensor=leela_encodings.board_to_tensor,
        move_to_index=leela_encodings.encode_move,
        position_evaluator=position_evaluator,
    )
    return encoded_seq


@pytest.fixture(scope="function")
def encoded_complex_seq(complex_seq, stockfish_metric_fix_compose):
    def position_evaluator(board, us_them):
        player = "white" if us_them[0] else "black"
        return stockfish_metric_fix_compose.eval_board(board, player=player)

    encoded_seq = leela_encodings.encode_seq(
        complex_seq,
        board_to_tensor=leela_encodings.board_to_tensor,
        move_to_index=leela_encodings.encode_move,
        position_evaluator=position_evaluator,
    )
    return encoded_seq


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


class TestRewardRepresentation:
    """
    Tests that the reward representation for Leela encodings is as expected.
    """

    def test_reward_two_player_complex(self, encoded_complex_seq, complex_seq, stockfish_metric):
        """
        Test that the reward representation for the Leela encodings is as expected
            for a complex game sequence
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        leela_inputs = leela_encodings.format_inputs(
            encoded_complex_seq,
            window_size=34,
            one_player=False,
            generator=generator,
            device=torch.device("cpu"),
            shaping_rewards=True,
        )
        board = chess.Board()

        def eval_seq(board, sequence, player, stockfish_metric):
            evaluations = []
            board = chess.Board()
            evaluations.append(stockfish_metric.eval_board(board, player))
            for alg_move in sequence.split():
                if alg_move.endswith("."):
                    continue
                board.push_san(alg_move)
                evaluation = stockfish_metric.eval_board(board, player)
                evaluations.append(evaluation)
            return evaluations[:-1]

        complex_eval = torch.tensor(eval_seq(board, complex_seq, player="both", stockfish_metric=stockfish_metric))

        alternating = torch.ones(69)
        alternating[1::2] = -1
        two_player_game = alternating + complex_eval  # Evaluating moves instead of positions
        assert torch.all(leela_inputs["returns_to_go"].squeeze() == two_player_game[:-1]).item()

    def test_reward_two_player_simple(self, encoded_simple_seq, simple_seq, stockfish_metric):
        """
        Test that the reward representation for the Leela encodings is as expected
            for a simple game sequence
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        leela_inputs = leela_encodings.format_inputs(
            encoded_simple_seq,
            window_size=6,
            one_player=False,
            generator=generator,
            device=torch.device("cpu"),
            shaping_rewards=True,
        )
        board = chess.Board()

        def eval_seq(board, sequence, player, stockfish_metric):
            evaluations = []
            board = chess.Board()
            evaluations.append(stockfish_metric.eval_board(board, player))
            for alg_move in sequence.split():
                if alg_move.endswith("."):
                    continue
                board.push_san(alg_move)
                evaluation = stockfish_metric.eval_board(board, player)
                evaluations.append(evaluation)
            return evaluations[:-1]

        simple_eval = torch.tensor(eval_seq(board, simple_seq, player="both", stockfish_metric=stockfish_metric))

        rtg = torch.ones(12) / 2
        two_player_game = rtg + simple_eval  # Evaluating moves instead of positions
        assert torch.all(leela_inputs["returns_to_go"].squeeze() == two_player_game).item()

    def test_reward_one_player_complex(self, encoded_complex_seq, complex_seq, stockfish_metric):
        """
        Test that the reward representation for the Leela encodings is as expected
            for a complex game sequence for one player game
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        leela_inputs = leela_encodings.format_inputs(
            encoded_complex_seq,
            window_size=34,
            one_player=True,
            generator=generator,
            device=torch.device("cpu"),
            shaping_rewards=True,
        )
        board = chess.Board()

        def eval_seq(board, sequence, player, stockfish_metric):
            evaluations = []
            board = chess.Board()
            evaluations.append(stockfish_metric.eval_board(board, player))
            for alg_move in sequence.split():
                if alg_move.endswith("."):
                    continue
                board.push_san(alg_move)
                evaluation = stockfish_metric.eval_board(board, player)
                evaluations.append(evaluation)
            return evaluations[:-1]

        complex_eval = torch.tensor(eval_seq(board, complex_seq, player="both", stockfish_metric=stockfish_metric))

        alternating = torch.ones(69)
        alternating[1::2] = -1
        two_player_game = alternating + complex_eval  # Evaluating moves instead of positions
        one_player_game = two_player_game[::2]
        assert torch.all(leela_inputs["returns_to_go"].squeeze() == one_player_game[:-1]).item()

    def test_reward_one_player_simple(self, encoded_simple_seq, simple_seq, stockfish_metric):
        """
        Test that the reward representation for the Leela encodings is as expected
            for a simple game sequence for one player game
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
        board = chess.Board()

        def eval_seq(board, sequence, player, stockfish_metric):
            evaluations = []
            board = chess.Board()
            evaluations.append(stockfish_metric.eval_board(board, player))
            for alg_move in sequence.split():
                if alg_move.endswith("."):
                    continue
                board.push_san(alg_move)
                evaluation = stockfish_metric.eval_board(board, player)
                evaluations.append(evaluation)
            return evaluations[:-1]

        simple_eval = torch.tensor(eval_seq(board, simple_seq, player="both", stockfish_metric=stockfish_metric))

        rtg = torch.ones(12) / 2
        two_player_game = rtg + simple_eval  # Evaluating moves instead of positions
        one_player_game = two_player_game[::2]
        assert torch.all(leela_inputs["returns_to_go"].squeeze() == one_player_game).item()

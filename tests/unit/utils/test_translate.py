"""
Unit tests for the translate module.
"""

import chess
import torch

import src.utils.translate as translate


class TestBoardTensorisation:
    def test_initial_board_to_64tensor(self):
        board = chess.Board()
        tensor = translate.board_to_64tensor(board)
        assert tensor.shape == (64,)
        init_board = torch.zeros(64)
        first_line = [4, 2, 3, 5, 6, 3, 2, 4]
        init_board[:8] = torch.tensor(first_line)
        init_board[8:16] = 1
        init_board[-16:-8] = -1
        init_board[-8:] = -torch.tensor(first_line)
        assert torch.all(tensor == init_board)

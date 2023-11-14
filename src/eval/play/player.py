"""
File implementing the player class.
"""

from typing import Optional

import chess
import torch


class Player:
    is_human: bool
    model: Optional[torch.nn.Module]

    def __init__(self) -> None:
        self.is_human = True
        self.model = None

    @classmethod
    def from_model(cls, model: torch.nn.Module):
        """
        Create a player from a model.
        """
        player = cls()
        player.is_human = False
        player.model = model
        return player

    def play(self, board: chess.Board, **kwargs) -> chess.Move:
        """
        Play a move.
        """
        if self.is_human:
            return self._play_human(board)
        else:
            return self._play_model(board, **kwargs)

    def _play_human(self, board: chess.Board) -> chess.Move:
        """
        Play a move as a human.
        """
        print(board)
        move = input("Enter move: ")
        return chess.Move.from_uci(move)

    def _play_model(self, board: chess.Board, **kwargs) -> chess.Move:
        """
        Play a move as a model.
        """
        pass

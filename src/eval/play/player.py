"""
File implementing the player class.
"""

from typing import Optional

import chess
import torch

from src.eval.play.inference import InferenceWrapper


def pretty_print(board: chess.Board) -> str:
    """
    Pretty print a board.
    """
    repr = str(board)
    repr = "\n".join([f"[{8-i}] {line}" for i, line in enumerate(repr.split("\n"))])
    repr += "\n    a b c d e f g h"
    print(repr)


class Player:
    is_human: bool
    inference_wrapper: Optional[InferenceWrapper]

    def __init__(self, handle_illegal: bool = False) -> None:
        self.is_human = True
        self.inference_wrapper = None
        self.format_kwargs = None
        self.handle_illegal = handle_illegal

    @classmethod
    def from_inference_wrapper(
        cls,
        inference_wrapper: InferenceWrapper,
        handle_illegal: bool = False,
    ):
        """
        Create a player from an inference_wrapper.
        """
        player = cls(handle_illegal=handle_illegal)
        player.is_human = False
        player.inference_wrapper = inference_wrapper
        return player

    @classmethod
    def random_policy(cls):
        """
        Create a player with a random policy.
        """
        player = cls()
        player.is_human = False
        player.inference_wrapper = None
        return player

    def play(self, board: chess.Board) -> chess.Move:
        """
        Play a move.
        """
        if self.is_human:
            return self._play_human(board)
        else:
            return self._play_inference_wrapper(board)

    def _play_human(self, board: chess.Board) -> chess.Move:
        """
        Play a move as a human.
        """
        pretty_print(board)
        move = input("Enter move: ")
        try:
            move = chess.Move.from_uci(move)
        except chess.InvalidMoveError:
            print("Invalid move.")
            return self._play_human(board)
        if self.handle_illegal and move not in board.legal_moves:
            print("Illegal move.")
            return self._play_human(board)
        return move

    def _play_inference_wrapper(self, board: chess.Board) -> chess.Move:
        """
        Play a move as a inference_wrapper.
        """
        if self.inference_wrapper is None:
            n_moves = board.legal_moves.count()
            move_idx = torch.randint(n_moves, (1,)).item()
            return list(board.legal_moves)[move_idx]
        else:
            mv = self.inference_wrapper(board)
            if mv not in board.legal_moves:
                print("Illegal move.")
                raise NotImplementedError("Not implemented yet.")
            return mv

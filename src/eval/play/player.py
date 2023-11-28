"""
File implementing the player class.
"""

from typing import Optional

import chess
import torch

from src.utils import translate


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
    model: Optional[torch.nn.Module]

    def __init__(self, handle_illegal: bool = False) -> None:
        self.is_human = True
        self.model = None
        self.format_kwargs = None
        self.handle_illegal = handle_illegal

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        handle_illegal: bool = False,
        format_kwargs: Optional[dict] = None,
    ):
        """
        Create a player from a model.
        """
        player = cls(handle_illegal=handle_illegal)
        player.is_human = False
        player.model = model
        if format_kwargs is not None:
            player.format_kwargs = format_kwargs
        return player

    @classmethod
    def random_policy(cls):
        """
        Create a player with a random policy.
        """
        player = cls()
        player.is_human = False
        player.model = None
        return player

    def play(self, board: chess.Board) -> chess.Move:
        """
        Play a move.
        """
        if self.is_human:
            return self._play_human(board)
        else:
            return self._play_model(board)

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

    def _play_model(self, board: chess.Board) -> chess.Move:
        """
        Play a move as a model.
        """
        if self.model is None:
            n_moves = board.legal_moves.count()
            move_idx = torch.randint(n_moves, (1,)).item()
            return list(board.legal_moves)[move_idx]
        else:
            move_indices, board_tensors, end_rewards = translate.encode_seq(
                board.move_stack,
                board_to_tensor=translate.board_to_64tensor,
            )
            input_dict = translate.format_inputs(
                move_indices,
                board_tensors,
                end_rewards,
                device=torch.device("cpu"),
                discount=0.99,
                **self.format_kwargs,
            )
            input_dict = {key: input_dict[key].unsqueeze(0) for key in input_dict}
            output_dict = self.model(**input_dict)
            logits = output_dict["action_preds"][0, -1, :]
            move_idx = torch.argmax(logits).item()
            mv = translate.decode_move(move_idx)
            if mv not in board.legal_moves:
                print("Illegal move.")
                raise NotImplementedError("Not implemented yet.")
            return mv

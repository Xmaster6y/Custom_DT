"""
File implementing the game class.
"""

import chess

from src.eval.play import Player


class Game:
    """
    Class implementing a game.
    """

    def __init__(self, white: Player, black: Player) -> None:
        self.white = white
        self.black = black
        self.board = chess.Board()
        self.is_ai_playing = not white.is_human or not black.is_human

    def play(self) -> None:
        """
        Play a game.
        """
        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                move = self.white.play(self.board, move_stack=self.board.move_stack)
            else:
                move = self.black.play(self.board, move_stack=self.board.move_stack)
            if move not in self.board.legal_moves:
                print(f"Player {self.board.turn} made an illegal move: {move}")
                break
            self.board.push(move)
        print(self.board.result())

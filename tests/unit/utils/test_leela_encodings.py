"""
File to test the encodings for the Leela Chess Zero engine.
"""

import random

import chess

from src.utils.leela_encodings import decode_move, encode_move


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
            encoded_move = encode_move(move, (us, them))
            decoded_move = decode_move(encoded_move, (us, them), board)
            assert move == decoded_move
            board.push(move)
            us, them = them, us

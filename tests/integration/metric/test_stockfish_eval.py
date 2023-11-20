"""
Test translate.
"""

import os
import pathlib
import sys

import chess
import chess.engine
import torch

import src.metric.stockfish as stockfish

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)

cwd = os.getcwd()
sys.path.append(cwd)


class TestStockfishEval:
    def test_simple_boards(self, simple_boards):
        stockfish_root = list(pathlib.Path(f"{cwd}/stockfish-source/stockfish/").glob("*.exe"))[0]
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_root)
        for player in ["white", "black", "both"]:
            for evaluation_depth in range(4, 13):
                evaluations = stockfish.stockfish_eval(simple_boards, engine, player, evaluation_depth)
                assert len(evaluations) == 13
                assert isinstance(evaluations[0], float)
                if player in ["white", "both"]:
                    assert evaluations[0] > 0.0
                elif player == "black":
                    assert evaluations[0] < 0.0
                for eval in evaluations:
                    assert eval >= -1.0
                    assert eval <= 1.0

"""
Test stockfish eval.
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


class TestStockfishEval:
    def test_simple_boards(self, simple_seq, stockfish_metric):
        for player in ["white", "black", "both"]:
            for evaluation_depth in range(4, 13):
                evaluations = stockfish_metric.eval_sequence(simple_seq, player, evaluation_depth)
                assert len(evaluations) == 12
                assert isinstance(evaluations[0], float)
                if player in ["white", "both"]:
                    assert evaluations[0] > 0.0
                elif player == "black":
                    assert evaluations[0] < 0.0
                for eval in evaluations:
                    assert eval >= -1.0
                    assert eval <= 1.0

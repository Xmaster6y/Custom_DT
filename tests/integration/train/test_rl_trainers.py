"""
Test RL Trainers
"""

import os

import torch

from src.metric.stockfish import StockfishMetric
from src.train_rl.utils.rl_trainers import DecisionTransformerREINFORCETrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(DEVICE)


class TestDecisionTransformerREINFORCETrainer:
    def test_training_deprec_encodings(self, RL_deprec_encoding_model, RL_deprec_encoding_trainer_cfg):
        stockfish_metric = StockfishMetric()
        RL_deprec_encoding_trainer_cfg.stockfish_metric = stockfish_metric

        trainer = DecisionTransformerREINFORCETrainer(
            cfg=RL_deprec_encoding_trainer_cfg, model=RL_deprec_encoding_model, device=DEVICE
        )
        trainer.train()
        stockfish_metric.engine.quit()

        cwd = os.getcwd()
        assert os.path.exists(f"{cwd}\\logging\\test_training.txt")
        assert os.path.exists(f"{cwd}\\figures\\test_training.png")
        assert os.path.exists(f"{cwd}\\weights\\test_training_checkpt_e4.pt")
        assert os.path.exists(f"{cwd}\\weights\\test_training_checkpt_e9.pt")
        os.remove(f"{cwd}\\logging\\test_training.txt")
        os.remove(f"{cwd}\\figures\\test_training.png")
        os.remove(f"{cwd}\\weights\\test_training_checkpt_e4.pt")
        os.remove(f"{cwd}\\weights\\test_training_checkpt_e9.pt")

"""
File implementing a match making system.
"""

import torch

from src.eval.play import Game, Player
from src.models.decision_transformer import DecisionTransformerModel

model = DecisionTransformerModel.from_pretrained("weights/dt_64_10_1_1e-05/checkpoint-5250")
format_kwargs = {
    "act_dim": 4096,
    "state_dim": 768,
    "window_size": 10,
    "generator": torch.Generator().manual_seed(42),
    "return_dict": True,
    "return_labels": False,
}
white = Player.from_model(model, format_kwargs=format_kwargs)
black = Player()
game = Game(white, black)
game.play()

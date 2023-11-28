"""
File implementing a match making system.
"""

import torch

from src.eval.play import Game, InferenceConfig, InferenceWrapper, Player
from src.models.decision_transformer import DecisionTransformerModel

model = DecisionTransformerModel.from_pretrained("weights/dt_64_10_1_1e-05/checkpoint-5250")
model.eval()
inference_config = InferenceConfig(
    two_player=False,
    window_size=10,
    shaping_rewards=False,
    device=torch.device("cpu"),
    end_rewards=(1.0, -1.0),
)
inference_wrapper = InferenceWrapper(model, inference_config)

white = Player.from_inference_wrapper(inference_wrapper)
black = Player()
game = Game(white, black)
game.play()

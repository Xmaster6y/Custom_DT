"""
File implementing a match making system.
"""

import argparse
import warnings

import chess
import torch

from src.eval.play import Game, InferenceConfig, InferenceWrapper, LeelaInferenceWrapper, Player
from src.metric.stockfish import StockfishMetric
from src.models.decision_transformer import DecisionTransformerModel

parser = argparse.ArgumentParser("leela")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--leela", action="store_true", default=True)

args = parser.parse_args()

model = DecisionTransformerModel.from_pretrained(args.checkpoint)
model.eval()

try:
    stockfish_metric = StockfishMetric()

    def position_evaluator(board, us_them):
        player = "white" if us_them[0] == chess.WHITE else "black"
        return stockfish_metric.eval_board(board, player=player, evaluation_depth=5)

    inference_config = InferenceConfig(
        two_player=False,
        window_size=20,
        shaping_rewards=True,
        device=torch.device("cpu"),
        end_rewards=(1.0, -1.0),
        position_evaluator=position_evaluator,
    )

    if args.leela:
        inference_wrapper = LeelaInferenceWrapper(model, inference_config, chess.WHITE)
    else:
        warnings.warn("You should use leela.", DeprecationWarning)
        inference_wrapper = InferenceWrapper(model, inference_config)

    white = Player.from_inference_wrapper(inference_wrapper)
    black = Player()
    game = Game(white, black)
    game.play()
finally:
    stockfish_metric.engine.quit()

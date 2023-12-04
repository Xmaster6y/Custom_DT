"""
Model training using self-supervised learning.
"""
import argparse

import chess
import torch
from transformers import TrainingArguments

from src.metric.stockfish import StockfishMetric
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.utils.dataset import LeelaChessDataset
from src.utils.leela_constants import ACT_DIM, STATE_DIM
from src.utils.trainer import DecisionTransformerTrainer, compute_metrics

parser = argparse.ArgumentParser("leela")
# Meta
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--training", action=argparse.BooleanOptionalAction, default=False)
# Config
parser.add_argument("--window-size", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--layers", type=int, default=6)
parser.add_argument("--heads", type=int, default=4)
# Training
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--n-epochs", type=int, default=1)
parser.add_argument("--logging-steps-ratio", type=float, default=0.01)
parser.add_argument("--eval-steps-ratio", type=float, default=0.1)
parser.add_argument("--train-batch-size", type=int, default=50)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--eval-batch-size", type=int, default=500)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--one-player", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--use-stockfish-eval", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--stockfish-eval-depth", type=int, default=6)
parser.add_argument("--resume-from-checkpoint", action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.name is None:
    NAME = (
        f"leela{'_sf' if args.use_stockfish_eval else ''}"
        f"{'_1p' if args.one_player else '_2p'}"
        f"L{args.layers}_H{args.heads}"
        f"_{args.window_size}_{args.train_batch_size}_{args.lr}"
    )
else:
    NAME = args.name
if args.debug:
    OUTPUT_DIR = "weights/debug"
    LOGGING_DIR = "logging/debug"
else:
    OUTPUT_DIR = f"weights/{NAME}"
    LOGGING_DIR = f"logging/{NAME}"


if args.use_stockfish_eval:
    stockfish_metric = StockfishMetric()

    def move_evaluator(board, us_them):
        player = "white" if us_them[0] == chess.WHITE else "black"
        return stockfish_metric.eval_board(board, player=player, evaluation_depth=args.stockfish_eval_depth)

else:
    move_evaluator = None

eval_generator = torch.Generator()
eval_generator.manual_seed(args.seed)
eval_dataset = LeelaChessDataset(
    file_name="data/chess_games_base/test_stockfish_5000.jsonl",
    move_evaluator=move_evaluator,
    window_size=args.window_size,
    generator=eval_generator,
    eval_mode=True,
    one_player=args.one_player,
)
eval_dataset_len = len(eval_dataset)

train_generator = torch.Generator()
train_generator.manual_seed(args.seed)
if args.debug:
    train_dataset_file = "data/chess_games_base/test_stockfish_5000.jsonl"
else:
    train_dataset_file = "data/chess_games_base/train_stockfish_262k.jsonl"
train_dataset = LeelaChessDataset(
    file_name=train_dataset_file,
    move_evaluator=move_evaluator,
    window_size=args.window_size,
    generator=train_generator,
    eval_mode=False,
    one_player=args.one_player,
)
train_dataset_len = len(train_dataset)

conf = DecisionTransformerConfig(
    state_dim=STATE_DIM,
    act_dim=ACT_DIM,
    n_layers=args.layers,
    n_heads=args.heads,
    hidden_size=64 * args.heads,
)
model = DecisionTransformerModel(conf)


trainer_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOGGING_DIR,
    overwrite_output_dir=args.debug or args.overwrite,
    logging_strategy="steps",
    logging_steps=int(args.logging_steps_ratio * train_dataset_len / args.train_batch_size),
    prediction_loss_only=False,
    evaluation_strategy="steps",
    eval_steps=int(args.eval_steps_ratio * train_dataset_len / args.train_batch_size),
    save_strategy="steps",
    save_steps=int(args.eval_steps_ratio * train_dataset_len / args.train_batch_size),
    per_device_eval_batch_size=args.eval_batch_size,
    per_device_train_batch_size=args.train_batch_size,
    num_train_epochs=args.n_epochs,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    run_name="latest",
)

trainer = DecisionTransformerTrainer(
    model=model,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
if args.training:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    evaluation = trainer.evaluate()
    print(evaluation)

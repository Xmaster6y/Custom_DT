"""
Model training using self-supervised learning/offline reinforcement learning using Leela encodings.

This script trains a Decision Transformer model on the dataset created using Leela encodings.
The training can be initiated using the command line.

Arguments:
    --debug: if True, the training is done on a small dataset.
    --training: if True, the model is trained. Otherwise, the model is evaluated.
    --window-size: the size of the move window to model for each sequence in the dataset.
    --seed: the seed used to generate the training examples.
    --layers: the number of layers in the Decision Transformer model.
    --heads: the number of attention heads in the Decision Transformer model.
    --name: the name of the training. If None, the name is automatically generated.
    --overwrite: if True, the log directory is overwritten.
    --n-epochs: the number of epochs used for training.
    --logging-steps-ratio: the ratio of steps between each logging.
    --eval-steps-ratio: the ratio of steps between each evaluation.
    --train-batch-size: the size of the training batch.
    --gradient-accumulation-steps: the number of gradient accumulation steps.
    --eval-batch-size: the size of the evaluation batch.
    --lr: the learning rate.
    --one-player: if True, the training examples are generated from one player's perspective.
    --use-stockfish-eval: if True, the training examples are augmeted with Stockfish evaluation shaping rewards.
    --stockfish-eval-depth: the depth of the Stockfish evaluation.
    --resume-from-checkpoint: if True, the training is resumed from the latest checkpoint.
    --checkpoint-path: the path to the checkpoint from which the training is resumed.
    --output-root: the root folder where the weights and logs are saved.


Typical usage example:

```bash
>>> python src/train/leela.py --debug True --window-size 10
```
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
parser.add_argument("--checkpoint-path", type=str, default=None)
parser.add_argument("--output-root", type=str, default="")

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
    OUTPUT_DIR = f"{args.output_root}weights/debug"
    LOGGING_DIR = f"{args.output_root}logging/debug"
else:
    OUTPUT_DIR = f"{args.output_root}weights/{NAME}"
    LOGGING_DIR = f"{args.output_root}logging/{NAME}"


if args.use_stockfish_eval:
    stockfish_metric = StockfishMetric()

    def position_evaluator(board, us_them):
        global stockfish_metric
        player = "white" if us_them[0] == chess.WHITE else "black"
        return stockfish_metric.eval_board(board, player=player, evaluation_depth=args.stockfish_eval_depth)

else:
    position_evaluator = None

try:  # To be sure to close stockfish engine if an error occurs
    eval_generator = torch.Generator()
    eval_generator.manual_seed(args.seed)
    eval_dataset = LeelaChessDataset(
        file_name="data/chess_games_base/test_stockfish_5000.jsonl",
        position_evaluator=position_evaluator,
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
        position_evaluator=position_evaluator,
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
    model.to(DEVICE)  # Not necessary

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
        fp16=True,
    )

    trainer = DecisionTransformerTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    if args.training:
        try:
            if args.checkpoint_path is not None:
                trainer.train(resume_from_checkpoint=args.checkpoint_path)
            else:
                trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        except ValueError:
            trainer.train()
    else:
        evaluation = trainer.evaluate()
        print(evaluation)
finally:
    if stockfish_metric is not None:
        stockfish_metric.engine.quit()
    print("Done!")

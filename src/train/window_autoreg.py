"""
Model training using self-supervised learning.
"""
import argparse

import torch
from transformers import TrainingArguments

import src.utils.translate as translate
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.utils.dataset import TwoPlayersChessDataset
from src.utils.trainer import DecisionTransformerTrainer, compute_metrics

parser = argparse.ArgumentParser("train_induction")
# Meta
parser.add_argument("--debug", type=int, default=True)
parser.add_argument("--training", type=int, default=False)
# Config
parser.add_argument("--state_dim", type=int, default=64)
parser.add_argument("--act_dim", type=int, default=4672)
parser.add_argument("--window_size", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
# Training
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--overwrite", type=int, default=False)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--logging_steps_ratio", type=float, default=0.01)
parser.add_argument("--eval_steps_ratio", type=float, default=0.1)
parser.add_argument("--train_batch_size", type=int, default=50)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--eval_batch_size", type=int, default=500)
parser.add_argument("--lr", type=float, default=1e-5)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.name is None:
    NAME = f"dt_{args.state_dim}_{args.window_size}_{args.n_epochs}_{args.lr}"
else:
    NAME = args.name
if args.debug:
    OUTPUT_DIR = "weights/debug"
    LOGGING_DIR = "logging/debug"
else:
    OUTPUT_DIR = f"weights/{NAME}"
    LOGGING_DIR = f"logging/{NAME}"


eval_generator = torch.Generator(device=DEVICE)
eval_generator.manual_seed(args.seed)
eval_dataset = TwoPlayersChessDataset(
    file_name="data/chess_games_base/test_stockfish_5000.jsonl",
    board_to_tensor=translate.board_to_64tensor,
    act_dim=args.act_dim,
    state_dim=args.state_dim,
    window_size=args.window_size,
    generator=eval_generator,
    return_ids=True,
    eval_mode=True,
)
eval_dataset_len = len(eval_dataset)

train_generator = torch.Generator(device=DEVICE)
train_generator.manual_seed(args.seed)
if args.debug:
    train_dataset_file = "data/chess_games_base/test_stockfish_5000.jsonl"
else:
    train_dataset_file = "data/chess_games_base/train_stockfish_262k.jsonl"
train_dataset = TwoPlayersChessDataset(
    file_name=train_dataset_file,
    board_to_tensor=translate.board_to_64tensor,
    act_dim=4672,
    state_dim=64,
    window_size=10,
    generator=train_generator,
    return_ids=True,
)
train_dataset_len = len(train_dataset)

conf = DecisionTransformerConfig(
    state_dim=args.state_dim,
    act_dim=args.act_dim,
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
    trainer.train()
else:
    evaluation = trainer.evaluate()
    print(evaluation)

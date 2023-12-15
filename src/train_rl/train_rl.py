"""
Model training using reinforcement learning with a basic REINFORCE algorithm.
"""
import argparse

import torch

from src.metric.stockfish import StockfishMetric
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.train_rl.utils.rl_trainer_config import RLTrainerConfig
from src.train_rl.utils.rl_trainers import DecisionTransformerREINFORCETrainer

parser = argparse.ArgumentParser("train-rl")
# Meta
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--training", action=argparse.BooleanOptionalAction, default=False)
# Config
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--layers", type=int, default=6)
parser.add_argument("--heads", type=int, default=4)
# Training
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--n-epochs", type=int, default=100)
parser.add_argument("--logging-steps-ratio", type=float, default=0.01)
parser.add_argument("--eval-steps-ratio", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--one-player", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--use-stockfish-eval", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stockfish-eval-depth", type=int, default=6)
parser.add_argument("--stockfish-gameplay-depth", type=int, default=2)
parser.add_argument("--resume-from-checkpoint", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--checkpoint-path", type=str, default=None)
parser.add_argument("--output-root", type=str, default="")
parser.add_argument("--temperature", type=float, default=1.0)
# Model
parser.add_argument("--state-dim", type=int, default=72)
parser.add_argument("--act-dim", type=int, default=4672)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.name is None:
    NAME = (
        f"train-rl{'_sf' if args.use_stockfish_eval else ''}"
        f"{'_1p' if args.one_player else '_2p'}"
        f"L{args.layers}_H{args.heads}"
        f"_{args.lr}"
    )
else:
    NAME = args.name
if args.debug:
    OUTPUT_DIR = f"{args.output_root}\\weights\\debug"
    LOGGING_DIR = f"{args.output_root}\\logging\\debug"
    FIGURES_DIR = f"{args.output_root}\\figures\\debug"
else:
    OUTPUT_DIR = f"{args.output_root}\\weights\\{NAME}"
    LOGGING_DIR = f"{args.output_root}\\logging\\{NAME}"
    FIGURES_DIR = f"{args.output_root}\\figures\\{NAME}"


if args.use_stockfish_eval:
    stockfish_metric = StockfishMetric()

try:  # To be sure to close stockfish engine if an error occurs
    conf = DecisionTransformerConfig(
        state_dim=args.state_dim,
        act_dim=args.act_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        hidden_size=64 * args.heads,
    )
    model = DecisionTransformerModel(conf)
    model.to(DEVICE)  # Not necessary

    trainer_cfg = RLTrainerConfig(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        figures_dir=FIGURES_DIR,
        overwrite_output_dir=args.debug or args.overwrite,
        logging_steps_ratio=args.logging_steps_ratio,
        eval_steps_ratio=args.eval_steps_ratio,
        save_steps_ratio=args.eval_steps_ratio,
        num_train_epochs=args.n_epochs,
        run_name=NAME,
        seed=args.seed,
        lr=args.lr,
        one_player=args.one_player,
        use_stock_fish_eval=args.use_stockfish_eval,
        stockfish_metric=stockfish_metric if args.use_stockfish_eval else None,
        stockfish_eval_depth=args.stockfish_eval_depth,
        stockfish_gameplay_depth=args.stockfish_gameplay_depth,
        resume_from_checkpoint=args.resume_from_checkpoint,
        checkpoint_path=args.checkpoint_path,
        state_dim=args.state_dim,
        act_dim=args.act_dim,
        temperature=args.temperature,
    )

    trainer = DecisionTransformerREINFORCETrainer(cfg=trainer_cfg, model=model, device=DEVICE)

    if args.training:
        try:
            trainer.train()
        except ValueError:
            trainer.train()
    else:
        evaluation = trainer.evaluate()
        print(evaluation)
finally:
    if stockfish_metric is not None:
        stockfish_metric.engine.quit()

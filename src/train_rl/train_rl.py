"""
Model training using RL and a simple REINFORCE algorithm using deprecated encodings.

This script trains a Decision Transformer model on the dataset created using the deprecated encodings.
The training can be initiated using the command line.

Arguments:
    --debug: if True, then output directories will be overwritten and signified as debug.
    --training: if True, the model is trained. Otherwise, the model is evaluated.
    --seed: the seed used to generate the training examples.
    --layers: the number of layers in the Decision Transformer model.
    --heads: the number of attention heads in the Decision Transformer model.
    --name: the name of the training. If None, the name is automatically generated.
    --overwrite: if True, the log directory is overwritten.
    --n-epochs: the number of epochs used for training.
    --logging-steps-ratio: the ratio of steps between each logging.
    --eval-steps-ratio: the ratio of steps between each evaluation.
    --lr: the learning rate.
    --one-player: if True, the training examples are generated from one player's perspective.
    --use-stockfish-eval: if True, the training examples are augmented with Stockfish evaluation shaping rewards.
    --stockfish-eval-depth: the depth of the Stockfish evaluation.
    --stockfish-gameplay-depth: the depth of the Stockfish evaluation during gameplay.
    --resume-from-checkpoint: if True, the training is resumed from the latest checkpoint.
    --checkpoint-path: the local path to the checkpoint from which the training is resumed.
    --checkpointing-steps-ratio: the ratio of steps between each time the model is saved.
    --output-root: the root folder where the weights and logs are saved.
    --temperature: the temperature used for the softmax in the policy.
    --lr-scheduler: if True, a cosine-annealing learning rate scheduler is used.

Note:
    The RL trainer currently only supports the one player mode, dense reward setting.
    Resuming from checkpoint is currently not supported. Evaluation mode is currently not supported.

Typical usage example:

```bash
>>> python src.train_rl.train_rl.py --training True --no-debug --n-epochs 10000 --lr 1e-4
```
"""
import argparse
import os

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
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--one-player", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--use-stockfish-eval", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stockfish-eval-depth", type=int, default=6)
parser.add_argument("--stockfish-gameplay-depth", type=int, default=2)
parser.add_argument("--resume-from-checkpoint", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--checkpoint-path", type=str, default=None)
parser.add_argument("--checkpointing-steps-ratio", type=float, default=0.1)
parser.add_argument("--output-root", type=str, default=os.getcwd())
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--lr-scheduler", action=argparse.BooleanOptionalAction, default=False)
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
        save_steps_ratio=args.checkpointing_steps_ratio,
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
        lr_scheduler=args.lr_scheduler,
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

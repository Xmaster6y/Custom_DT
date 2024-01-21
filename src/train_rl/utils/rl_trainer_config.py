"""
Configuration object for RL training classes.

Typical usage example:
```python
>>> from src.train_rl.utils.rl_trainer_config import RLTrainerConfig
>>> from src.models.decision_transformer import DecisionTransformer
>>> from src.train_rl.utils.rl_trainers import DecisionTransformerREINFORCETrainer
>>> model = DecisionTransformer()
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> config = RLTrainerConfig(
...     output_dir="weights/debug",
...     logging_dir="logging/debug",
...     overwrite_output_dir=True,
...     ...)
>>> trainer = DecisionTransformerREINFORCETrainer(config, model, device)
>>> trainer.train()
```
```bash
>>> # in terminal
>>> tensorboard --logdir logging
>>> # don't forget to clean up the logging and weights directories after training!
```
"""

from src.metric.stockfish import StockfishMetric


class RLTrainerConfig:
    """
    Configuration object for RL training classes.

    Attributes:
        output_dir: path to the output directory.
        logging_dir: path to the logging directory.
        overwrite_output_dir: whether to overwrite the output directory.
        logging_steps_ratio: ratio of steps to log.
        eval_steps_ratio: ratio of steps to evaluate.
        save_steps_ratio: ratio of steps to save.
        num_train_epochs: number of training epochs.
        run_name: name of the run.
        seed: random seed.
        lr: learning rate.
        one_player: whether the RL agent models one player or both players.
        which_player: which player the RL agent models. Options are 'white', 'black', or 'random'.
        use_stockfish_eval: whether to use Stockfish evaluations as shaping reward.
        stockfish_metric: metric to use for Stockfish evaluations.
        stockfish_eval_depth: depth to use for Stockfish evaluations.
        stockfish_gameplay_depth: depth to use for Stockfish gameplay.
        resume_from_checkpoint: whether to resume from a checkpoint.
        checkpoint_path: path to the checkpoint.
        state_dim: dimension of state space.
        act_dim: dimension of action space.
        temperature: temperature for softmax.
        lr_scheduler: whether to use a learning rate scheduler.
    """

    def __init__(
        self,
        output_dir: str,
        logging_dir: str,
        overwrite_output_dir: bool,
        logging_steps_ratio: float,
        eval_steps_ratio: float,
        save_steps_ratio: float,
        num_train_epochs: int,
        run_name: str,
        seed: int,
        lr: float,
        one_player: bool,
        which_player: str,
        use_stockfish_eval: bool,
        stockfish_metric: StockfishMetric,
        stockfish_eval_depth: int,
        stockfish_gameplay_depth: int,
        resume_from_checkpoint: bool,
        checkpoint_path: str,
        state_dim: int,
        act_dim: int,
        temperature: float,
        lr_scheduler: bool,
    ):
        """
        Initializes the RLTrainerConfig object.

        Args:
            output_dir: path to the output directory.
            logging_dir: path to the logging directory.
            overwrite_output_dir: whether to overwrite the output directory.
            logging_steps_ratio: ratio of steps to log.
            eval_steps_ratio: ratio of steps to evaluate.
            save_steps_ratio: ratio of steps to save.
            num_train_epochs: number of training epochs.
            run_name: name of the run.
            seed: random seed.
            lr: learning rate.
            one_player: whether the RL agent models one player or both players.
            which_player: which player the RL agent models. Options are 'white', 'black', or 'random'.
            use_stockfish_eval: whether to use Stockfish evaluations as shaping reward.
            stockfish_metric: metric to use for Stockfish evaluations.
            stockfish_eval_depth: depth to use for Stockfish evaluations.
            stockfish_gameplay_depth: depth to use for Stockfish gameplay.
            resume_from_checkpoint: whether to resume from a checkpoint.
            checkpoint_path: path to the checkpoint.
            state_dim: dimension of state space.
            act_dim: dimension of action space.
            temperature: temperature for softmax.
            lr_scheduler: whether to use a learning rate scheduler.
        """
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.overwrite_output_dir = overwrite_output_dir
        self.logging_steps_ratio = logging_steps_ratio
        self.eval_steps_ratio = eval_steps_ratio
        self.save_steps_ratio = save_steps_ratio
        self.num_train_epochs = num_train_epochs
        self.run_name = run_name
        self.seed = seed
        self.lr = lr
        self.one_player = one_player
        self.which_player = which_player
        self.use_stockfish_eval = use_stockfish_eval
        self.stockfish_metric = stockfish_metric
        self.stockfish_eval_depth = stockfish_eval_depth
        self.stockfish_gameplay_depth = stockfish_gameplay_depth
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_path = checkpoint_path
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.temperature = temperature
        self.lr_scheduler = lr_scheduler

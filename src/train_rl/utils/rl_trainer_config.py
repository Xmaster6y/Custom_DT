from src.metric.stockfish import StockfishMetric


class RLTrainerConfig:
    def __init__(
        self,
        output_dir: str,
        logging_dir: str,
        figures_dir: str,
        overwrite_output_dir: bool,
        logging_steps_ratio: float,
        eval_steps_ratio: float,
        save_steps_ratio: float,
        num_train_epochs: int,
        run_name: str,
        seed: int,
        lr: float,
        one_player: bool,
        use_stock_fish_eval: bool,
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
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.figures_dir = figures_dir
        self.overwrite_output_dir = overwrite_output_dir
        self.logging_steps_ratio = logging_steps_ratio
        self.eval_steps_ratio = eval_steps_ratio
        self.save_steps_ratio = save_steps_ratio
        self.num_train_epochs = num_train_epochs
        self.run_name = run_name
        self.seed = seed
        self.lr = lr
        self.one_player = one_player
        self.use_stock_fish_eval = use_stock_fish_eval
        self.stockfish_metric = stockfish_metric
        self.stockfish_eval_depth = stockfish_eval_depth
        self.stockfish_gameplay_depth = stockfish_gameplay_depth
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_path = checkpoint_path
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.temperature = temperature
        self.lr_scheduler = lr_scheduler

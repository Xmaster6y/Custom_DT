"""
Reinforcement learning training classes for the Decision Transformer model.
"""

import pathlib
import sys
from itertools import count
from typing import Optional, Tuple

import chess
import chess.engine
import matplotlib.pyplot as plt
import torch
import tqdm
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.utils import step_mdp

from src.models.decision_transformer.modeling_decision_transformer import DecisionTransformerModel
from src.train_rl.utils.rl_trainer_config import RLTrainerConfig
from src.utils import leela_encodings
from src.utils.leela_constants import ACT_DIM, STATE_DIM


class DecisionTransformerREINFORCETrainer:
    """
    Trainer for the Decision Transformer model using the REINFORCE algorithm.

    This trainer can be used for training and evaluation of the Decision Transformer
    using Leela chess game encodings. The trainer can save and load model checkpoints, and
    will log training and evaluation metrics to a text file.

    Attributes:
        cfg: configuration object for the trainer.
        model: the Decision Transformer model.
        device: device to use for training.
        env: environment for RL training, specifically the agent's Stockfish opponent.
        engine: Stockfish engine for evaluation.
        checkpoint_dict: dictionary containing the model state dictionary, optimizer state dictionary,
            and loss at the time of the latest checkpoint.

    Notes:
        01/21/2023: supports only one-player, dense reward setting. It is undecided whether this is temporary.
    """

    def __init__(self, cfg: RLTrainerConfig, model: DecisionTransformerModel, device: str):
        """
        Initializes the DecisionTransformerREINFORCETrainer class based on a RLTrainerConfig object.

        Args:
            cfg: RLTrainerConfig configuration object for the trainer.
            model: the Decision Transformer model.
            device: device to use for training.

        Raises:
            ValueError: an error occurred if the platform is not recognized.
        """
        self.cfg = cfg
        self.model = model
        if self.cfg.resume_from_checkpoint:
            self.checkpoint_dict = torch.load(self.cfg.checkpoint_path)
            self.model.load_state_dict(self.checkpoint_dict["model_state_dict"])

        self.device = device

        if sys.platform in ["linux"]:
            platform = "linux"
        elif sys.platform in ["win32", "cygwin"]:
            platform = "windows"
        elif sys.platform in ["darwin"]:
            platform = "macos"
        else:
            raise ValueError(f"Unknown platform {sys.platform}")

        if platform in ["linux", "macos"]:
            exec_re = "stockfish*"
        elif platform == "windows":
            exec_re = "stockfish*.exe"
        else:
            raise ValueError(f"Unknown platform {platform}")

        stockfish_root = list(pathlib.Path("stockfish-source/stockfish/").glob(exec_re))[0]
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_root)

    def train(self):
        """
        Trains the Decision Transformer model using the REINFORCE algorithm. The environment
        used for training is a Stockfish opponent in a chess game. The training loss will be
        logged to a text file, and the rolling average loss can be plotted.

        Notes:
            01/21/2023: supports only one-player, dense reward setting. It is undecided whether this is temporary.

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
        ...     figures_dir="figures/debug",
        ...     overwrite_output_dir=True,
        ...     ...)
        >>> trainer = DecisionTransformerREINFORCETrainer(config, model, device)
        >>> trainer.train()
        ```
        """
        self.env = ChessEnv(
            seed=self.cfg.seed,
            device=self.device,
            state_dim=self.cfg.state_dim,
            act_dim=self.cfg.act_dim,
            gameplay_depth=self.cfg.stockfish_gameplay_depth,
            which_player=self.cfg.which_player,
            engine=self.engine,
            stockfish_metric=self.cfg.stockfish_metric,
            stockfish_eval_depth=self.cfg.stockfish_eval_depth,
        )
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        pbar = tqdm.tqdm(range(self.cfg.num_train_epochs))

        logging_steps = int(1 / self.cfg.logging_steps_ratio)
        checkpointing_steps = int(1 / self.cfg.save_steps_ratio)
        logs = {"loss": [], "rolling_av_loss": []}

        if self.cfg.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.cfg.num_train_epochs)

        if self.cfg.resume_from_checkpoint:
            optim.load_state_dict(self.checkpoint_dict["optim_state_dict"])
            pbar = tqdm.tqdm(
                iterable=range(self.checkpoint_dict["epoch"] + 1, self.cfg.num_train_epochs),
                total=self.cfg.num_train_epochs,
                initial=self.checkpoint_dict["epoch"] + 1,
            )
            logs["loss"] = self.checkpoint_dict["loss"]
            logs["rolling_av_loss"] = self.checkpoint_dict["rolling_av_loss"]

        for i in pbar:
            rollout, distros = self.one_game_rollout()
            loss = [
                distro.log_prob(action.argmax()) * traj_return
                for distro, action, traj_return in zip(distros, rollout["action"], rollout["next"]["reward"])
            ]
            (sum(loss) / len(loss)).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
            pbar.set_description(f"loss: {loss}," f"rewards: {rollout['next']['reward']},")
            logs["loss"].append((sum(loss) / len(loss)).item())

            if i % logging_steps == logging_steps - 1:
                logs["rolling_av_loss"].append(sum(logs["loss"][-logging_steps:]) / logging_steps)

            if i % checkpointing_steps == checkpointing_steps - 1:
                self.save_checkpoint(
                    epoch=i,
                    model_state_dict=self.model.state_dict(),
                    optim_state_dict=optim.state_dict(),
                    loss=logs["loss"],
                    rolling_av_loss=logs["rolling_av_loss"],
                )

            if self.cfg.lr_scheduler:
                scheduler.step()

        self.env.engine.quit()
        self.env.stockfish_metric.engine.quit()
        self.log(logs)

    def one_game_rollout(self) -> Tuple[TensorDict, list]:
        """
        Performs a single rollout of the Decision Transformer model in a chess game against a Stockfish opponent.
        The rollout is one game of chess, with the Decision Transformer model playing as white. The game will terminate
        when the game has concluded or the model has made an illegal move. The data from the entire game is recorded
        along with the action distributions from the model at each timestep.

        Returns:
            A tuple containing a TensorDict of data from the game and the action distributions from the model
                at each timestep.
        """
        _data, curr_env_state = self.env.reset()  # dtype = float32 & shape = (cfg.state_dim,)
        data = _data.expand(1).contiguous()  # dtype = float32 & shape = (1,cfg.state_dim)
        unused_action = torch.nn.functional.one_hot(
            torch.tensor([1]), num_classes=self.cfg.act_dim
        ).float()  # dtype = float32 & shape = (1,self.cfg.act_dim)
        if _data["params", "us_them"][0]:
            starting_return_to_go = torch.ones((1, 1), dtype=torch.float32)  # dtype = float32 & shape = (1,1)
        else:
            starting_return_to_go = torch.tensor(
                [
                    [
                        1
                        - self.cfg.stockfish_metric.eval_board(
                            curr_env_state, player="black", evaluation_depth=self.cfg.stockfish_eval_depth
                        )
                    ]
                ]
            )
        action_distros = []
        for t in count():
            if t > 0:
                state_seq = torch.flatten(
                    torch.cat([data["board"][0].unsqueeze(0), data["next"]["board"]], dim=0).unsqueeze(0),
                    start_dim=2,
                    end_dim=-1,
                )  # needs dtype = float32 & shape = (1, seq_len, cfg.state_dim)
                action_seq = torch.cat([data["action"], unused_action], dim=0).unsqueeze(
                    0
                )  # needs dtype = float32 & shape = (1, seq_len, self.cfg.act_dim)
                rtg_seq = torch.cat([starting_return_to_go, data["next"]["reward"]], dim=0).unsqueeze(
                    0
                )  # needs dtype = float32 & shape = (1, seq_len, 1)
                timestep_seq = torch.arange(t + 1).unsqueeze(0)  # needs dtype = int64 & shape = (1,t)
            else:
                state_seq = torch.flatten(
                    data["board"].unsqueeze(0), start_dim=2, end_dim=-1
                )  # dtype = float32 & shape = (1, 1, cfg.state_dim)
                action_seq = unused_action.unsqueeze(0)  # dtype = float32 & shape = (1, 1, self.cfg.act_dim)
                rtg_seq = starting_return_to_go.unsqueeze(0)  # dtype = float32 & shape = (1, 1, 1)
                timestep_seq = torch.zeros((1, 1), dtype=torch.int64)  # dtype = int64 & shape = (1,1)

            _, action_preds, _ = self.model(
                states=state_seq,
                actions=action_seq,
                returns_to_go=rtg_seq,
                timesteps=timestep_seq,
                return_dict=False,
            )

            curr_action_pred = action_preds[:, -1, :]  # dtype = float32 & shape = (1, self.cfg.act_dim)
            temp_scaled_curr_action = curr_action_pred / self.cfg.temperature
            curr_action_distro = torch.distributions.categorical.Categorical(logits=temp_scaled_curr_action)
            action_distros.append(curr_action_distro)
            sampled_action = (
                torch.nn.functional.one_hot(curr_action_distro.sample(), num_classes=self.cfg.act_dim).float().squeeze()
            )

            _data["action"] = sampled_action  # dtype = float32 & shape = (self.cfg.act_dim,)

            _data, curr_env_state = self.env.step(
                _data, curr_env_state
            )  # out = dtype = float32 & shape = (cfg.state_dim,), (self.cfg.act_dim,), (1,)
            data = torch.cat([data, _data.expand(1).contiguous()], dim=0) if t > 0 else _data.expand(1).contiguous()

            _data = step_mdp(_data, keep_other=True)

            if _data["done"]:
                _data, curr_env_state = self.env.reset(_data)
                break

        return (data, action_distros)

    def evaluate(self):
        raise NotImplementedError

    def save_checkpoint(
        self, epoch: int, model_state_dict: dict, optim_state_dict: dict, loss: list, rolling_av_loss: list
    ):
        """
        Saves a model checkpoint to a file.

        Args:
            epoch: current epoch number.
            model_state_dict: state dictionary of the model.
            optim_state_dict: state dictionary of the optimizer.
            loss: loss value at the current epoch.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optim_state_dict": optim_state_dict,
                "loss": loss,
                "rolling_av_loss": rolling_av_loss,
            },
            f"{self.cfg.output_dir}_checkpt_e{epoch}.pt",
        )

    def plot(self, rolling_av_loss: list):
        """
        Plots the rolling average loss of a training run. The plot will be saved to a png file.

        Args:
            rolling_av_loss: list of rolling average loss values from a training run.

        Notes:
            This is a primitive plotting function that will be improved.
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        n_points = len(rolling_av_loss) // 200
        plt.plot(rolling_av_loss[:: n_points + 1])
        plt.title("rolling av loss")
        plt.xlabel("iteration")
        plt.savefig(f"{self.cfg.figures_dir}.png")

    def log(self, logs):
        """
        Logs the rolling average loss of a training run to a text file. Will also
        plot the rolling average loss.

        Args:
            logs: dictionary of logs from a training run.

        Notes:
            This is a primitive logging function that will be improved.
        """
        action = "w" if self.cfg.overwrite_output_dir else "a"
        with open(f"{self.cfg.logging_dir}.txt", action) as f:
            f.write(f"run name: {self.cfg.run_name}\n")
            f.write(f"lr: {self.cfg.lr}\n")
            f.write(f"number of epochs: {self.cfg.num_train_epochs}\n")
            f.write("rolling average loss: ")
            for i in range(0, len(logs["rolling_av_loss"]), 10):
                for log in logs["rolling_av_loss"][i : i + 10]:
                    f.write(str(log))
                    f.write(", ")
                f.write("\n")
            f.write("\n\n\n")
            f.write("#" * 100)
            f.write("#" * 100)
            f.write("\n\n\n")
            print("Done!")
        self.plot(logs["rolling_av_loss"])


class ChessEnv:
    """
    Environment for the Decision Transformer model to play chess against.

    This environment is a Stockfish chess player. This environment is built using the tensordict
    library: https://pytorch.org/tensordict/overview.html.

    Attributes:
        device: device to use for training.
        state_dim: dimension of state space.
        act_dim: dimension of action space.
        gameplay_depth: depth to use for Stockfish gameplay.
        which_player: the agent's perspective. choices: "white", "black", or "random".
        engine: Stockfish engine for gameplay.
        stockfish_metric: StockfishMetric object to use for Stockfish evaluations.
        stockfish_eval_depth: search depth to use for Stockfish evaluations.
        observation_spec: specification for the observation space. This is equivalent to the state
            space because this is a fully observable environment.
        state_spec: specification for the state space.
        action_spec: specification for the action space.
        reward_spec: specification for the reward space.
        done_spec: specification for the termination signal.
        rng: random number generator.
    """

    def __init__(
        self,
        td_params=None,
        seed=None,
        device="cpu",
        batch_size=None,
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        gameplay_depth=2,
        which_player="white",
        engine=None,
        stockfish_metric=None,
        stockfish_eval_depth=6,
    ):
        """
        Initializes an instance of the ChessEnv class.

        Args:
            td_params: tensordict containing the hyperparameters for the environment.
            seed: random seed.
            device: device to use for training.
            batch_size: batch size to use for training.
            state_dim: dimension of state space.
            act_dim: dimension of action space.
            gameplay_depth: depth to use for Stockfish gameplay.
            which_player: the agent's perspective. choices: "white", "black", or "random".
            engine: Stockfish engine for gameplay.
            stockfish_metric: StockfishMetric object to use for Stockfish evaluations.
            stockfish_eval_depth: search depth to use for Stockfish evaluations.

        Raises:
            ValueError: an error occurs if which_player is not "white", "black", or "random".
        """
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gameplay_depth = gameplay_depth
        if which_player not in ["white", "black", "random"]:
            raise ValueError("which_player must be 'white', 'black', or 'random'")
        self.which_player = which_player
        self.engine = engine
        self.stockfish_metric = stockfish_metric
        self.stockfish_eval_depth = stockfish_eval_depth

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)

        if td_params is None:
            td_params = ChessEnv.gen_params(
                device=self.device,
                rng=self.rng,
                batch_size=batch_size,
                gameplay_depth=self.gameplay_depth,
                which_player=self.which_player,
            )

        self._make_spec(td_params)

    @staticmethod
    def gen_params(
        device: str,
        rng: torch.Generator,
        batch_size=None,
        gameplay_depth=2,
        which_player="white",
    ) -> TensorDict:
        """
        Generates the hyperparameters for the environment.

        Args:
            device: device on which training is being conducted.
            rng: random number generator.
            batch_size: batch size to use for training.
            gameplay_depth: depth to use for Stockfish gameplay.
            which_player: choices: "white", "black", or "random".

        Returns:
            A tensordict containing the hyperparameters for the environment.
        """
        if batch_size is None:
            batch_size = []
        if which_player == "white":
            us = True
        elif which_player == "black":
            us = False
        else:
            us = bool(torch.randint(2, (1,), generator=rng).item())
        them = not us
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "env_gameplay_depth": torch.tensor([gameplay_depth], dtype=torch.int16),
                        "us_them": torch.tensor([us, them], dtype=torch.bool),
                    },
                    [],
                )
            },
            [],
            device=device,
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def _make_spec(self, td_params: TensorDict):
        """
        Creates the specifications for the observation, state, action, and reward spaces, along
        with the specification for the termination signal.

        Args:
            td_params: tensordict containing the hyperparameters for the environment.
        """
        self.observation_spec = CompositeSpec(
            board=BoundedTensorSpec(
                low=-7,
                high=32000,
                shape=(self.state_dim,),
                dtype=torch.float32,
                device=self.device,
            ),
            params=self.make_composite_from_td(td_params["params"]),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            low=0,
            high=self.act_dim,
            shape=(1,),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward_spec = BoundedTensorSpec(
            low=0,
            high=11,
            shape=(1,),
            dtype=torch.float32,
            device=self.device,
        )
        self.done_spec = BoundedTensorSpec(
            low=0,
            high=1,
            shape=(1,),
            dtype=torch.bool,
            device=self.device,
        )

    def make_composite_from_td(self, td: TensorDict) -> CompositeSpec:
        """
        Converts a tensordict into a similar spec structure of unbounded values.

        Args:
            td: tensordict to convert.

        Returns:
            A CompositeSpec object with the same structure as the input tensordict.
        """
        return CompositeSpec(
            {
                key: self.make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else UnboundedContinuousTensorSpec(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
                for key, tensor in td.items()
            },
            shape=td.shape,
        )

    def reset(self, tensordict: TensorDict = None, batch_size=None) -> Tuple[TensorDict, chess.Board]:
        """
        Resets the environment by generating a starting chess board.

        Args:
            tensordict: tensordict containing information about the environment.
            batch_size: batch size to use for training.

        Returns:
            A tensordict of the starting chess board that is ready to be modified in a rollout and the starting
            chess board object.
        """
        tensordict = self.gen_params(
            device=self.device,
            rng=self.rng,
            batch_size=batch_size,
            gameplay_depth=self.gameplay_depth,
            which_player=self.which_player,
        )

        if tensordict["params", "us_them"][0]:
            chess_obj_board = chess.Board()
            board = leela_encodings.board_to_tensor(chess_obj_board, us_them=(chess.WHITE, chess.BLACK))
        else:
            chess_obj_board = chess.Board()
            white_move = self.engine.play(
                chess_obj_board, chess.engine.Limit(depth=tensordict["params", "env_gameplay_depth"])
            ).move
            chess_obj_board.push(white_move)
            board = leela_encodings.board_to_tensor(
                chess_obj_board, us_them=(chess.BLACK, chess.WHITE)
            )  # board should come out flipped

        return (
            TensorDict(
                {
                    "board": board,  # dtype = float32 & shape = (STATE_DIM,)
                    "params": tensordict["params"],
                    "done": torch.tensor([0], dtype=torch.bool),
                },
                batch_size=tensordict.shape,
                device=self.device,
            ),
            chess_obj_board,
        )

    def step(self, tensordict: TensorDict, curr_env_state: chess.Board) -> Tuple[TensorDict, chess.Board]:
        """
        Performs a single environment step. If the action taken by the model is not an intelligble
        move (from-square is not occupied), the return-to-go will be 10 and the game will terminate.
        If the action taken by the model is an intelligible but illegal move, the return-to-go will
        be 7 and the game will terminate. If the action taken by the model is a legal move, the
        Stockfish engine will play a move and the return-to-go will be the Stockfish evaluation of
        the board position from the perspective of white.

        Args:
            tensordict: tensordict containing the current state of the environment.
            curr_env_state: current chess board object of the environment.

        Returns:
            A tuple containing a.) a tensordict containing a tensor representation of the next state of the
            environment, the hyperparameters, the reward, and the termination signal, and b.) a chess board
            object of the next state of the environment.
        """
        board_tensor, move = (
            tensordict["board"],
            tensordict["action"],
        )  # dtype = float32 & shape = (self.cfg.state_dim,), (self.cfg.act_dim,)
        env_gameplay_depth = tensordict["params", "env_gameplay_depth"]
        us_them = tensordict["params", "us_them"]
        us = chess.WHITE if us_them[0] else chess.BLACK
        them = not us

        proposed_move = leela_encodings.decode_move(torch.argmax(move).item(), us_them=(us, them), board=curr_env_state)
        # from square is not occupied by a white piece: should be punished severely
        proposed_from_piece = curr_env_state.piece_at(proposed_move.from_square)
        if proposed_from_piece is None or proposed_from_piece.color != us:
            next_board_tensor = board_tensor
            reward_tensor = torch.tensor([10], dtype=torch.float32)
            done = torch.tensor([1], dtype=torch.bool)
        # proposing a move with a white piece, but not a legal move: should be punished less severely
        elif proposed_move not in list(curr_env_state.legal_moves):
            next_board_tensor = board_tensor
            reward_tensor = torch.tensor([7], dtype=torch.float32)
            done = torch.tensor([1], dtype=torch.bool)
        # proposes legal move
        else:
            curr_env_state.push(proposed_move)

            if curr_env_state.outcome() is not None:
                done = torch.tensor([1], dtype=torch.bool)
            else:
                done = torch.tensor([0], dtype=torch.bool)
                next_move = self.engine.play(curr_env_state, chess.engine.Limit(depth=env_gameplay_depth)).move
                curr_env_state.push(next_move)
            next_board_tensor = leela_encodings.board_to_tensor(board=curr_env_state.copy(), us_them=(us, them))
            reward_tensor = torch.tensor(
                [
                    1
                    - self.stockfish_metric.eval_board(
                        curr_env_state,
                        player="white" if us == chess.WHITE else "black",
                        evaluation_depth=self.stockfish_eval_depth,
                    )
                ]
            )

            if curr_env_state.outcome() is not None:
                done = torch.tensor([1], dtype=torch.bool)

        next_td = TensorDict(
            {
                "board": next_board_tensor,  # dtype = float32 & shape = (72,)
                "params": tensordict["params"],
                "reward": reward_tensor,  # dtype = float32 & shape = (1,)
                "done": done,  # dtype = bool & shape = (1,)
            },
            tensordict.shape,
        )
        tensordict["next"] = next_td
        return tensordict, curr_env_state

    def _set_seed(self, seed: Optional[int]):
        """
        Sets the random seed for the environment.

        Args:
            seed: random seed.
        """
        rng = torch.manual_seed(seed)
        self.rng = rng

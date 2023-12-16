import pathlib
import sys
from itertools import count
from typing import Optional

import chess
import chess.engine
import matplotlib.pyplot as plt
import torch
import tqdm
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import step_mdp

from src.utils import translate


class DecisionTransformerREINFORCETrainer:
    """Trainer for the Decision Transformer model using the REINFORCE algorithm.
    Notes:
    - Only supports one-player, dense reward setting from the perspective of white, right now.
    """

    def __init__(self, cfg, model, device):
        self.cfg = cfg
        if self.cfg.resume_from_checkpoint:
            self.model = model.load_from_checkpoint(self.cfg.checkpoint_path)
        else:
            self.model = model
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
        self.env = ChessEnv(
            seed=self.cfg.seed,
            device=self.device,
            state_dim=self.cfg.state_dim,
            act_dim=self.cfg.act_dim,
            gameplay_depth=self.cfg.stockfish_gameplay_depth,
            engine=self.engine,
            stockfish_metric=self.cfg.stockfish_metric,
            stockfish_eval_depth=self.cfg.stockfish_eval_depth,
        )
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

        pbar = tqdm.tqdm(range(self.cfg.num_train_epochs))

        if self.cfg.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.cfg.num_train_epochs)

        logging_steps = int(1 / self.cfg.logging_steps_ratio)
        logs = {"loss": [], "rolling_av_loss": []}

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
            if self.cfg.lr_scheduler:
                scheduler.step()
        self.log(logs)

    def one_game_rollout(self):
        _data = self.env.reset()  # dtype = float32 & shape = (cfg.state_dim,)
        data = _data.expand(1).contiguous()  # dtype = float32 & shape = (1,cfg.state_dim)
        unused_action = torch.nn.functional.one_hot(
            torch.tensor([1]), num_classes=self.cfg.act_dim
        ).float()  # dtype = float32 & shape = (1,self.cfg.act_dim)
        starting_return_to_go = torch.ones((1, 1), dtype=torch.float32)  # dtype = float32 & shape = (1,1)
        action_distros = []
        for t in count():
            if t > 0:
                state_seq = torch.cat([data["board"][0].unsqueeze(0), data["next"]["board"]], dim=0).unsqueeze(
                    0
                )  # needs dtype = float32 & shape = (1, seq_len, cfg.state_dim)
                action_seq = torch.cat([data["action"], unused_action], dim=0).unsqueeze(
                    0
                )  # needs dtype = float32 & shape = (1, seq_len, self.cfg.act_dim)
                rtg_seq = torch.cat([starting_return_to_go, data["next"]["reward"]], dim=0).unsqueeze(
                    0
                )  # needs dtype = float32 & shape = (1, seq_len, 1)
                timestep_seq = torch.arange(t + 1).unsqueeze(0)  # needs dtype = int64 & shape = (1,t)
            else:
                state_seq = data["board"].unsqueeze(0)  # dtype = float32 & shape = (1, 1, cfg.state_dim)
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

            _data = self.env.step(_data)  # out = dtype = float32 & shape = (cfg.state_dim,), (self.cfg.act_dim,), (1,)
            data = torch.cat([data, _data.expand(1).contiguous()], dim=0) if t > 0 else _data.expand(1).contiguous()
            _data = step_mdp(_data, keep_other=True)

            if _data["done"]:
                _data = self.env.reset()
                break

        return data, action_distros

    def evaluate(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_from_checkpoint(self, checkpoint_path):
        raise NotImplementedError

    def plot(self, rolling_av_loss):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        n_points = len(rolling_av_loss) // 200
        plt.plot(rolling_av_loss[:: n_points + 1])
        plt.title("rolling av loss")
        plt.xlabel("iteration")
        plt.savefig(self.cfg.figures_dir + ".png")

    def log(self, logs):
        if self.cfg.overwrite_output_dir:
            action = "w"
        else:
            action = "a"
        with open(self.cfg.logging_dir + ".txt", action) as f:
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


class ChessEnv(EnvBase):
    metadata = {"render.modes": ["human"]}
    batch_locked = False

    def __init__(
        self,
        td_params=None,
        seed=None,
        device="cpu",
        batch_size=None,
        state_dim=72,
        act_dim=4672,
        gameplay_depth=2,
        engine=None,
        stockfish_metric=None,
        stockfish_eval_depth=6,
    ):
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gameplay_depth = gameplay_depth
        self.engine = engine
        self.stockfish_metric = stockfish_metric
        self.stockfish_eval_depth = stockfish_eval_depth

        if td_params is None:
            td_params = ChessEnv.gen_params(batch_size=batch_size, gameplay_depth=self.gameplay_depth)

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    @staticmethod
    def gen_params(batch_size=None, gameplay_depth=2) -> TensorDictBase:
        """Returns a tensordict containing the hyperparameters for the environment."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "env_gameplay_depth": torch.tensor([gameplay_depth], dtype=torch.int64),
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def _make_spec(self, td_params):
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

    def make_composite_from_td(self, td):
        """
        Custom funtion to convert a tensordict in a similar spec structure
        of unbounded values.
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

    def _reset(self, tensordict, batch_size=None):
        if tensordict is None or tensordict.is_empty():
            # if no tensordict is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input tensordict contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=batch_size)
        if self.state_dim == 72:
            board = translate.board_to_72tensor(chess.Board())
        else:
            raise RuntimeError("state_dim must be 72")

        return TensorDict(
            {
                "board": board,  # dtype = float32 & shape = (72,)
                "params": tensordict["params"],
                "done": torch.tensor([0], dtype=torch.bool),
            },
            batch_size=tensordict.shape,
        )

    def _step(self, tensordict):
        board_tensor, move = (
            tensordict["board"],
            tensordict["action"],
        )  # dtype = float32 & shape = (72,), (self.cfg.act_dim,)
        env_gameplay_depth = tensordict["params", "env_gameplay_depth"]
        board = chess.Board(fen=translate.complete_tensor_to_fen(board_tensor))  # dtype float32 & shape = (state_dim,)

        proposed_move = translate.decode_move(torch.argmax(move).item(), board)
        # from square is unoccupied - not even picking up a piece: should be punished severely
        if proposed_move is None:
            next_board_tensor = board_tensor
            reward_tensor = torch.tensor([10], dtype=torch.float32)
            done = torch.tensor([1], dtype=torch.bool)
        # proposing a move with an actual piece, but not a legal move: should be punished less severely
        elif proposed_move not in list(board.legal_moves):
            next_board_tensor = board_tensor
            reward_tensor = torch.tensor([7], dtype=torch.float32)
            done = torch.tensor([1], dtype=torch.bool)
        # proposes legal move
        else:
            board.push(proposed_move)

            if board.outcome() is not None:
                done = torch.tensor([1], dtype=torch.bool)
            else:
                done = torch.tensor([0], dtype=torch.bool)
                next_move = self.engine.play(board, chess.engine.Limit(depth=env_gameplay_depth)).move
                board.push(next_move)
            next_board_tensor = translate.board_to_72tensor(board.copy()).float()
            reward_tensor = torch.tensor(
                [
                    1
                    - self.stockfish_metric.eval_board(
                        board, player="white", evaluation_depth=self.stockfish_eval_depth
                    )
                ]
            )

            if board.outcome() is not None:
                done = torch.tensor([1], dtype=torch.bool)

        return TensorDict(
            {
                "board": next_board_tensor,  # dtype = float32 & shape = (72,)
                "params": tensordict["params"],
                "reward": reward_tensor,  # dtype = float32 & shape = (1,)
                "done": done,  # dtype = bool & shape = (1,)
            },
            tensordict.shape,
        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

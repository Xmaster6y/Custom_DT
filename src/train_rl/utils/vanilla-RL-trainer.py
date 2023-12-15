"""
DEPRECATED RL Training Loop

Notes:
 - Only supports one-player, dense reward setting from the perspective of white, right now.
"""
import os
import pathlib
import sys
import time
from collections import defaultdict
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
from torchrl.envs.utils import check_env_specs, step_mdp

from src.metric.stockfish import StockfishMetric
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.utils import translate

torch.set_printoptions(sci_mode=False)
cwd = os.getcwd()
sys.path.append(cwd)
stockfish_root = list(pathlib.Path(f"{cwd}/stockfish-source/stockfish/").glob("*.exe"))[0]
engine = chess.engine.SimpleEngine.popen_uci(stockfish_root)


def stockfish_move(board_tensor):
    board = chess.Board(fen=translate.complete_tensor_to_fen(board_tensor))
    move = engine.play(board, chess.engine.Limit(depth=5)).move
    if move is None:
        print(board)
        print(board.outcome())
    promotion = move.promotion
    if promotion is not None and promotion != chess.QUEEN:  # Underpromotion
        direction = (move.to_square % 8) - (move.from_square % 8)
        extra_index = (promotion - 2) + 3 * (direction + 1) + 9 * move.from_square
        move_idx = 4096 + extra_index
    else:
        move_idx = move.from_square + 64 * move.to_square
    return torch.nn.functional.one_hot(torch.tensor(move_idx), num_classes=4672).float()


conf = DecisionTransformerConfig(
    state_dim=72,
    act_dim=4672,
)
model = DecisionTransformerModel(conf)


class ChessEnv(EnvBase):
    metadata = {"render.modes": ["human"]}
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu", batch_size=None):
        if td_params is None:
            td_params = ChessEnv.gen_params(batch_size=batch_size)

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    @staticmethod
    def gen_params(batch_size=None) -> TensorDictBase:
        """Returns a tensordict containing the hyperparameters for the environment."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "env_search_depth": torch.tensor([2], dtype=torch.int64),
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
                shape=(72,),
                dtype=torch.float32,
                device=self.device,
            ),
            params=self.make_composite_from_td(td_params["params"]),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            low=0,
            high=4672,
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

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            # if no tensordict is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input tensordict contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

        return TensorDict(
            {
                "board": translate.board_to_72tensor(chess.Board()),  # dtype = float32 & shape = (72,)
                "params": tensordict["params"],
                "done": torch.tensor([0], dtype=torch.bool),
            },
            batch_size=tensordict.shape,
        )

    @staticmethod
    def _step(tensordict):
        board_tensor, move = tensordict["board"], tensordict["action"]  # dtype = float32 & shape = (72,), (4672,)
        env_search_depth = tensordict["params", "env_search_depth"]
        board = chess.Board(fen=translate.complete_tensor_to_fen(board_tensor))  # dtype float32 & shape = (72,)

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
                next_move = engine.play(board, chess.engine.Limit(depth=env_search_depth)).move
                board.push(next_move)
            next_board_tensor = translate.board_to_72tensor(board.copy()).float()
            sf_eval_metric = StockfishMetric()
            reward_tensor = torch.tensor([1 - sf_eval_metric.eval_board(board, player="white")])

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


env = ChessEnv()
check_env_specs(env)


def one_game_rollout(temperature=1.0):
    _data = env.reset()  # dtype = float32 & shape = (72,)
    data = _data.expand(1).contiguous()  # dtype = float32 & shape = (1,72)
    unused_action = torch.nn.functional.one_hot(
        torch.tensor([1]), num_classes=4672
    ).float()  # dtype = float32 & shape = (1,4672)
    starting_return_to_go = torch.ones((1, 1), dtype=torch.float32)  # dtype = float32 & shape = (1,1)
    action_distros = []
    for t in count():
        print("-" * 25, t, "-" * 25)
        if t > 0:
            print("legal move")
            print("|\n" * 100)

        if t > 0:
            state_seq = torch.cat([data["board"][0].unsqueeze(0), data["next"]["board"]], dim=0).unsqueeze(
                0
            )  # needs dtype = float32 & shape = (1, seq_len, 72)
            action_seq = torch.cat([data["action"], unused_action], dim=0).unsqueeze(
                0
            )  # needs dtype = float32 & shape = (1, seq_len, 4672)
            rtg_seq = torch.cat([starting_return_to_go, data["next"]["reward"]], dim=0).unsqueeze(
                0
            )  # needs dtype = float32 & shape = (1, seq_len, 1)
            timestep_seq = torch.arange(t + 1).unsqueeze(0)  # needs dtype = int64 & shape = (1,t)
        else:
            state_seq = data["board"].unsqueeze(0)  # dtype = float32 & shape = (1, 1, 72)
            action_seq = unused_action.unsqueeze(0)  # dtype = float32 & shape = (1, 1, 4672)
            rtg_seq = starting_return_to_go.unsqueeze(0)  # dtype = float32 & shape = (1, 1, 1)
            timestep_seq = torch.zeros((1, 1), dtype=torch.int64)  # dtype = int64 & shape = (1,1)

        _, action_preds, _ = model(
            states=state_seq,
            actions=action_seq,
            returns_to_go=rtg_seq,
            timesteps=timestep_seq,
            return_dict=False,
        )

        curr_action_pred = action_preds[:, -1, :]  # dtype = float32 & shape = (1, 4672)
        temp_scaled_curr_action = curr_action_pred / temperature
        curr_action_distro = torch.distributions.categorical.Categorical(logits=temp_scaled_curr_action)
        action_distros.append(curr_action_distro)
        sampled_action = torch.nn.functional.one_hot(curr_action_distro.sample(), num_classes=4672).float().squeeze()

        _data["action"] = sampled_action  # dtype = float32 & shape = (4672,)

        _data = env.step(_data)  # out = dtype = float32 & shape = (72,), (4672,), (1,)
        data = torch.cat([data, _data.expand(1).contiguous()], dim=0) if t > 0 else _data.expand(1).contiguous()
        _data = step_mdp(_data, keep_other=True)

        if _data["done"]:
            _data = env.reset()
            break

    return data, action_distros


def train(lr, steps, temp):
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    pbar = tqdm.tqdm(range(steps))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)
    logs = defaultdict(list)

    for i in pbar:
        rollout, distros = one_game_rollout(temperature=temp)
        loss = [
            distro.log_prob(action.argmax()) * traj_return
            for distro, action, traj_return in zip(distros, rollout["action"], rollout["next"]["reward"])
        ]
        (sum(loss) / len(loss)).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(f"loss: {loss}," f"rewards: {rollout['next']['reward']}," f"grad_norm: {gn}")
        logs["loss"].append((sum(loss) / len(loss)).item())
        if i % 100 == 99:
            logs["rolling_av_loss"].append(sum(logs["loss"][-100:]) / 100)
        scheduler.step()
    plot(logs)
    time.sleep(1)
    with open("src/utils/logs.txt", "a") as f:
        f.write(f"lr: {lr}\n")
        f.write(f"steps {steps}\n")
        f.write("rolling average loss: ")
        f.write(str(logs["rolling_av_loss"]))
        f.write("\n\n\n")


def plot(logs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(logs["rolling_av_loss"])
    plt.title("rolling av loss")
    plt.xlabel("iteration")
    plt.show()


train(lr=1e-3, steps=10000, temp=1.0)
